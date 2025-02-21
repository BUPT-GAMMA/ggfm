import bitsandbytes as bnb
import torch
import torch.nn.functional as F
from accelerate.hooks import remove_hook_from_module
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch import Tensor
from torch import nn
from torch_geometric.nn.pool import global_add_pool
from torch_geometric.transforms.add_positional_encoding import AddRandomWalkPE
from transformers import BitsAndBytesConfig
from transformers import (LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModel)

from gp.nn.layer.pyg import RGCNEdgeConv
from gp.nn.models.GNN import MultiLayerMessagePassing
from gp.nn.models.util_model import MLP
from gp.utils.utils import load_pretrained_state

LLM_DIM_DICT = {"ST": 768, "BERT": 768, "e5": 1024, "llama2_7b": 4096, "llama2_13b": 5120}


class TextClassModel(torch.nn.Module):
    """Text classification model that combines graph embeddings with class embeddings.
    
    Args:
        model: The underlying graph neural network model
        outdim (int): Output dimension of the model
        task_dim (int): Dimension of the task/classification output
        emb (torch.Tensor, optional): Pre-computed embeddings to use. Defaults to None.
    """
    def __init__(self, model, outdim, task_dim, emb=None):
        super().__init__()
        self.model = model
        if emb is not None:
            self.emb = torch.nn.Parameter(emb.clone())

        self.mlp = MLP([2 * outdim, 2 * outdim, outdim, task_dim])

    def forward(self, g):
        emb = self.model(g)
        class_emb = emb[g.target_node_mask]
        att_emb = class_emb.repeat_interleave(len(self.emb), dim=0)
        att_emb = torch.cat(
            [att_emb, self.emb.repeat(len(class_emb), 1)], dim=-1
        )
        res = self.mlp(att_emb).view(-1, len(self.emb))
        return res


class AdaPoolClassModel(torch.nn.Module):
    """Adaptive pooling classification model that performs weighted pooling of node embeddings.
    
    Args:
        model: The underlying graph neural network model
        outdim (int): Output dimension of the model
        task_dim (int): Dimension of the task/classification output
        emb (torch.Tensor, optional): Pre-computed embeddings to use. Defaults to None.
    """
    def __init__(self, model, outdim, task_dim, emb=None):
        super().__init__()
        self.model = model
        if emb is not None:
            self.emb = torch.nn.Parameter(emb.clone())

        self.mlp = MLP([2 * outdim, 2 * outdim, outdim, task_dim])

    def forward(self, g):
        emb = self.model(g)
        float_mask = g.target_node_mask.to(torch.float)
        target_emb = float_mask.view(-1, 1) * emb
        n_count = global_add_pool(float_mask, g.batch, g.num_graphs)
        class_emb = global_add_pool(target_emb, g.batch, g.num_graphs)
        class_emb = class_emb / n_count.view(-1, 1)
        rep_class_emb = class_emb.repeat_interleave(g.num_classes, dim=0)
        res = self.mlp(
            torch.cat([rep_class_emb, g.x[g.true_nodes_mask]], dim=-1)
        )
        return res


class SingleHeadAtt(torch.nn.Module):
    """Single-head attention layer implementation.
    
    Args:
        dim (int): Dimension of the input features
    
    Attributes:
        sqrt_dim (torch.Tensor): Square root of input dimension for scaling
        Wk (nn.Parameter): Key projection matrix
        Wq (nn.Parameter): Query projection matrix
    """
    def __init__(self, dim):
        super().__init__()
        self.sqrt_dim = torch.sqrt(torch.tensor(dim))
        self.Wk = torch.nn.Parameter(torch.zeros((dim, dim)))
        torch.nn.init.xavier_uniform_(self.Wk)
        self.Wq = torch.nn.Parameter(torch.zeros((dim, dim)))
        torch.nn.init.xavier_uniform_(self.Wq)

    def forward(self, key, query, value):
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim
        attn = torch.nn.functional.softmax(score, -1)
        context = torch.bmm(attn, value)
        return context, attn


class BinGraphModel(torch.nn.Module):
    """Binary graph classification model that combines GNN with language model features.
    
    Args:
        model: The underlying GNN model
        llm_name (str): Name of the language model to use
        outdim (int): Output dimension of the model
        task_dim (int): Dimension of the task output
        add_rwpe (int, optional): Random walk positional encoding dimension. Defaults to None.
        dropout (float, optional): Dropout rate. Defaults to 0.0.
        **kwargs: Additional keyword arguments
    
    Attributes:
        model: The GNN model
        llm_name (str): Name of the language model
        outdim (int): Output dimension
        llm_proj (nn.Linear): Projection layer for language model features
        mlp (MLP): Multi-layer perceptron for classification
        rwpe (AddRandomWalkPE, optional): Random walk positional encoding
    """
    def __init__(self, model, llm_name, outdim, task_dim, add_rwpe=None, dropout=0.0, **kwargs):
        super().__init__()
        assert llm_name in LLM_DIM_DICT.keys()
        self.model = model
        self.llm_name = llm_name
        self.outdim = outdim
        self.llm_proj = nn.Linear(LLM_DIM_DICT[llm_name], outdim)
        self.mlp = MLP([outdim, 2 * outdim, outdim, task_dim], dropout=0.0)
        if add_rwpe is not None:
            self.rwpe = AddRandomWalkPE(add_rwpe)
            self.edge_rwpe_prior = torch.nn.Parameter(
                torch.zeros((1, add_rwpe))
            )
            torch.nn.init.xavier_uniform_(self.edge_rwpe_prior)
            self.rwpe_normalization = torch.nn.BatchNorm1d(add_rwpe)
            self.walk_length = add_rwpe
        else:
            self.rwpe = None

    def initial_projection(self, g):
        g.x = self.llm_proj(g.x)
        g.edge_attr = self.llm_proj(g.edge_attr)
        return g

    def forward(self, g):
        g = self.initial_projection(g)

        if self.rwpe is not None:
            with torch.no_grad():
                rwpe_norm = self.rwpe_normalization(g.rwpe)
                g.x = torch.cat([g.x, rwpe_norm], dim=-1)
                g.edge_attr = torch.cat(
                    [
                        g.edge_attr,
                        self.edge_rwpe_prior.repeat(len(g.edge_attr), 1),
                    ],
                    dim=-1,
                )
        emb = self.model(g)
        class_emb = emb[g.true_nodes_mask]
        res = self.mlp(class_emb)
        return res

    def freeze_gnn_parameters(self):
        for p in self.model.parameters():
           p.requires_grad = False
        for p in self.mlp.parameters():
            p.requires_grad = False
        for p in self.llm_proj.parameters():
            p.requires_grad = False



class BinGraphAttModel(torch.nn.Module):
    """Binary graph classification model with attention-based pooling across layers.
    
    This model uses a single layer attention mechanism to pool node representations 
    across different layers of the GNN.
    
    Args:
        model: The underlying GNN model
        llm_name (str): Name of the language model to use
        outdim (int): Output dimension of the model
        task_dim (int): Dimension of the task output
        add_rwpe (int, optional): Random walk positional encoding dimension. Defaults to None.
        dropout (float, optional): Dropout rate. Defaults to 0.0.
        **kwargs: Additional keyword arguments
    
    Attributes:
        model: The GNN model
        llm_name (str): Name of the language model
        outdim (int): Output dimension
        llm_proj (nn.Linear): Projection layer for language model features
        mlp (MLP): Multi-layer perceptron for classification
        att (SingleHeadAtt): Single-head attention layer
    """
    def __init__(self, model, llm_name, outdim, task_dim, add_rwpe=None, dropout=0.0, **kwargs):
        super().__init__()
        assert llm_name in LLM_DIM_DICT.keys()
        self.model = model
        self.llm_name = llm_name
        self.outdim = outdim
        self.llm_proj = nn.Linear(LLM_DIM_DICT[llm_name], outdim)
        self.mlp = MLP([outdim, 2 * outdim, outdim, task_dim], dropout=0.0)
        self.att = SingleHeadAtt(outdim)
        if add_rwpe is not None:
            self.rwpe = AddRandomWalkPE(add_rwpe)
            self.edge_rwpe_prior = torch.nn.Parameter(
                torch.zeros((1, add_rwpe))
            )
            torch.nn.init.xavier_uniform_(self.edge_rwpe_prior)
            self.rwpe_normalization = torch.nn.BatchNorm1d(add_rwpe)
            self.walk_length = add_rwpe
        else:
            self.rwpe = None

    def initial_projection(self, g):
        g.x = self.llm_proj(g.x)
        g.edge_attr = self.llm_proj(g.edge_attr)
        return g

    def forward(self, g):
        g = self.initial_projection(g)
        if self.rwpe is not None:
            with torch.no_grad():
                rwpe_norm = self.rwpe_normalization(g.rwpe)
                g.x = torch.cat([g.x, rwpe_norm], dim=-1)
                g.edge_attr = torch.cat(
                    [
                        g.edge_attr,
                        self.edge_rwpe_prior.repeat(len(g.edge_attr), 1),
                    ],
                    dim=-1,
                )
        emb = torch.stack(self.model(g), dim=1)
        query = g.x.unsqueeze(1)
        emb = self.att(emb, query, emb)[0].squeeze()

        class_emb = emb[g.true_nodes_mask]
        res = self.mlp(class_emb)
        return res

    def freeze_gnn_parameters(self):
        for p in self.model.parameters():
           p.requires_grad = False
        for p in self.att.parameters():
            p.requires_grad = False
        for p in self.mlp.parameters():
            p.requires_grad = False
        for p in self.llm_proj.parameters():
            p.requires_grad = False

def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-10)


class LLMModel(torch.nn.Module):
    """Large Language Model wrapper for feature extraction and fine-tuning.
    
    This class provides functionality to load and use various language models with
    optional quantization and parameter-efficient fine-tuning (LoRA).
    
    Args:
        llm_name (str): Name of the language model to use
        quantization (bool, optional): Whether to use 4-bit quantization. Defaults to True.
        peft (bool, optional): Whether to use parameter-efficient fine-tuning. Defaults to True.
        cache_dir (str, optional): Directory to cache models. Defaults to "cache_data/model".
        max_length (int, optional): Maximum sequence length. Defaults to 500.
    
    Attributes:
        llm_name (str): Name of the language model
        quantization (bool): Whether quantization is enabled
        indim (int): Input dimension of the model
        cache_dir (str): Cache directory path
        max_length (int): Maximum sequence length
        model: The underlying language model
        tokenizer: The tokenizer for the language model
    """
    def __init__(self, llm_name, quantization=True, peft=True, cache_dir="cache_data/model", max_length=500):
        super().__init__()
        assert llm_name in LLM_DIM_DICT.keys()
        self.llm_name = llm_name
        self.quantization = quantization

        self.indim = LLM_DIM_DICT[self.llm_name]
        self.cache_dir = cache_dir
        self.max_length = max_length
        model, self.tokenizer = self.get_llm_model()
        if peft:
            self.model = self.get_lora_perf(model)
        else:
            self.model = model
        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = 'right'

    def find_all_linear_names(self, model):
        """
        find all module for LoRA fine-tuning.
        """
        cls = bnb.nn.Linear4bit if self.quantization else torch.nn.Linear
        lora_module_names = set()
        for name, module in model.named_modules():
            if isinstance(module, cls):
                names = name.split('.')
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])

        if 'lm_head' in lora_module_names:  # needed for 16-bit
            lora_module_names.remove('lm_head')
        return list(lora_module_names)

    def create_bnb_config(self):
        """
        quantization configuration.
        """
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        return bnb_config

    def get_lora_perf(self, model):
        """
        LoRA configuration.
        """
        target_modules = self.find_all_linear_names(model)
        config = LoraConfig(
            target_modules=target_modules,
            r=16,  # dimension of the updated matrices
            lora_alpha=16,  # parameter for scaling
            lora_dropout=0.2,  # dropout probability for layers
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        model = get_peft_model(model, config)

        return model

    def get_llm_model(self):
        if self.llm_name == "llama2_7b":
            model_name = "meta-llama/Llama-2-7b-hf"
            ModelClass = LlamaForCausalLM
            TokenizerClass = LlamaTokenizer

        elif self.llm_name == "llama2_13b":
            model_name = "meta-llama/Llama-2-13b-hf"
            ModelClass = LlamaForCausalLM
            TokenizerClass = LlamaTokenizer

        elif self.llm_name == "e5":
            model_name = "intfloat/e5-large-v2"
            ModelClass = AutoModel
            TokenizerClass = AutoTokenizer

        elif self.llm_name == "BERT":
            model_name = "bert-base-uncased"
            ModelClass = AutoModel
            TokenizerClass = AutoTokenizer

        elif self.llm_name == "ST":
            model_name = "sentence-transformers/multi-qa-distilbert-cos-v1"
            ModelClass = AutoModel
            TokenizerClass = AutoTokenizer

        else:
            raise ValueError(f"Unknown language model: {self.llm_name}.")
        if self.quantization:
            bnb_config = self.create_bnb_config()
            model = ModelClass.from_pretrained(model_name,
                                               quantization_config=bnb_config,
                                               #attn_implementation="flash_attention_2",
                                               #torch_type=torch.bfloat16,
                                               cache_dir=self.cache_dir)
        else:
            model = ModelClass.from_pretrained(model_name, cache_dir=self.cache_dir)
        model = remove_hook_from_module(model, recurse=True)
        model.config.use_cache = False
        tokenizer = TokenizerClass.from_pretrained(model_name, cache_dir=self.cache_dir, add_eos_token=True)
        if self.llm_name[:6] == "llama2":
            tokenizer.pad_token = tokenizer.bos_token
        return model, tokenizer

    def pooling(self, outputs, text_tokens=None):
        # if self.llm_name in ["BERT", "ST", "e5"]:
        return F.normalize(mean_pooling(outputs, text_tokens["attention_mask"]), p=2, dim=1)

        # else:
        #     return outputs[text_tokens["input_ids"] == 2] # llama2 EOS token

    def forward(self, text_tokens):
        outputs = self.model(input_ids=text_tokens["input_ids"],
                             attention_mask=text_tokens["attention_mask"],
                             output_hidden_states=True,
                             return_dict=True)["hidden_states"][-1]

        return self.pooling(outputs, text_tokens)

    def encode(self, text_tokens, pooling=False):

        with torch.no_grad():
            outputs = self.model(input_ids=text_tokens["input_ids"],
                                 attention_mask=text_tokens["attention_mask"],
                                 output_hidden_states=True,
                                 return_dict=True)["hidden_states"][-1]
            outputs = outputs.to(torch.float32)
            if pooling:
                outputs = self.pooling(outputs, text_tokens)

            return outputs, text_tokens["attention_mask"]



class TransformerModel(nn.Module):
    """Transformer encoder model implementation.
    
    Args:
        input_dim (int): Input dimension of the model
        num_layers (int): Number of transformer layers
        hidden_dim (int): Hidden dimension in transformer layers
        num_heads (int): Number of attention heads in each layer
    
    Attributes:
        encoder (nn.TransformerEncoder): The transformer encoder
    """

    def __init__(
        self, input_dim: int, num_layers: int, hidden_dim: int, num_heads: int
    ):
        super(TransformerModel, self).__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                input_dim, num_heads, hidden_dim, batch_first=True
            ),
            num_layers,
        )

    def forward(
        self,
        x: Tensor,
        mask: Tensor = None,
        src_key_padding_mask: Tensor = None,
    ) -> Tensor:
        encoded = self.encoder(
            x, mask=mask, src_key_padding_mask=src_key_padding_mask
        )
        return encoded


class PyGRGCNEdge(MultiLayerMessagePassing):
    """PyTorch Geometric implementation of Relational GCN with edge features.
    
    This model implements a multi-layer message passing neural network that can handle
    relational graphs with edge features.
    
    Args:
        num_layers (int): Number of message passing layers
        num_rels (int): Number of relation types
        inp_dim (int): Input feature dimension
        out_dim (int): Output feature dimension
        drop_ratio (float, optional): Dropout ratio. Defaults to 0
        JK (str, optional): Jumping knowledge mode. Defaults to "last"
        batch_norm (bool, optional): Whether to use batch normalization. Defaults to True
    
    Attributes:
        num_rels (int): Number of relation types
    """
    def __init__(
        self,
        num_layers: int,
        num_rels: int,
        inp_dim: int,
        out_dim: int,
        drop_ratio=0,
        JK="last",
        batch_norm=True,
    ):
        super().__init__(
            num_layers, inp_dim, out_dim, drop_ratio, JK, batch_norm
        )
        self.num_rels = num_rels
        self.build_layers()

    def build_input_layer(self):
        return RGCNEdgeConv(self.inp_dim, self.out_dim, self.num_rels)

    def build_hidden_layer(self):
        return RGCNEdgeConv(self.inp_dim, self.out_dim, self.num_rels)

    def build_message_from_input(self, g):
        return {
            "g": g.edge_index,
            "h": g.x,
            "e": g.edge_type,
            "he": g.edge_attr,
        }

    def build_message_from_output(self, g, h):
        return {"g": g.edge_index, "h": h, "e": g.edge_type, "he": g.edge_attr}

    def layer_forward(self, layer, message):
        return self.conv[layer](
            message["h"], message["he"], message["g"], message["e"]
        )

