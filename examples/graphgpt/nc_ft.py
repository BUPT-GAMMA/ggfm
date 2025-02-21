import copy
import dataclasses
import json
import logging
import os
from dataclasses import dataclass, field
from enum import auto, Enum
from typing import Dict, Optional, Sequence, List

import torch
import torch.nn as nn
import transformers
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import FSDPStrategy
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch_geometric.data import Data
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from ggfm.models.graphgpt import GraphGPT_pl

# TODO: import and use code from ../data/dataset.py

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
DEFAULT_GRAPH_TOKEN = "<graph>"
DEFAULT_GRAPH_PATCH_TOKEN = "<g_patch>"
DEFAULT_G_START_TOKEN = "<g_start>"
DEFAULT_G_END_TOKEN = "<g_end>"


#### tring bash cmd example ####

# to fill in the following path to run the first stage of our GraphGPT!
# cd ~/py/graphgpt/
# export PYTHONPATH=/home/csy/gammalab/HiGPT:$PYTHONPATH
# model_path=~/py/graphgpt/data/vicuna/vicuna-7b-v1.5-16k
# instruct_ds=~/py/graphgpt/data/graph_matching/train_instruct_graphmatch.json
# graph_data_path=~/py/graphgpt/data/graph_data/graph_data_all.pt
# pretra_gnn=clip_gt_arxiv_pub
# output_model=~/py/graphgpt/model/s1
#
# python graphgpt/train/train_light.py \
#     --gpus 0,1,2,3 \
#     --model_name_or_path ${model_path} \
#     --version v1 \
#     --data_path ${instruct_ds} \
#     --graph_content ./arxiv_ti_ab.json \
#     --graph_data_path ${graph_data_path} \
#     --graph_tower ${pretra_gnn} \
#     --tune_graph_mlp_adapter True \
#     --graph_select_layer -2 \
#     --use_graph_start_end True \
#     --bf16 False \
#     --output_dir ${output_model} \
#     --num_train_epochs 3 \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 1 \
#     --real_batch_size 1 \
#     --gradient_accumulation_steps 1 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 2400 \
#     --save_total_limit 1 \
#     --learning_rate 2e-5 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 2048 \
#     --gradient_checkpointing True \
#     --lazy_preprocess True \
#     --report_to wandb \
#     --fp16 False

### tring code ###


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_graph_mlp_adapter: bool = field(default=False)
    graph_tower: Optional[str] = field(default=None)
    graph_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    pretrain_graph_mlp_adapter: Optional[str] = field(default=None)
    use_graph_start_end: bool = field(default=False)
    model_save_name: Optional[str] = field(default="model_{epoch}-{step}")


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_graph: bool = False
    sep_graph_conv_front: bool = False
    graph_token_len: int = 0
    graph_content: Optional[str] = field(default=None)
    graph_data_path: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'


@dataclass
class TrainingArguments:
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_graph_mlp_adapter: bool = field(default=False)
    force_fsdp: bool = field(default=False)
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    strategy: str = field(
        default='fsdp'
    )
    real_batch_size: int = field(default=1)

    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    disable_tqdm: bool = False

    gpus: Optional[str] = field(default='0,1')
    resume: Optional[str] = field(default=None)

    adam_epsilon: float = field(default=1e-8)
    warmup_steps: int = field(default=1000)
    num_workers: int = field(default=16)

    bf16: bool = field(default=False)
    fp16: bool = field(default=False)
    output_dir: str = field(default='./checkpoints/graphchat-gt-graphmatch-7b')
    num_train_epochs: int = field(default=3)
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=1)
    evaluation_strategy: str = field(default='no')
    save_strategy: str = field(default='steps')
    save_steps: int = field(default=2400)
    save_total_limit: int = field(default=1)
    learning_rate: float = field(default=2e-5)
    weight_decay: float = field(default=0.)
    warmup_ratio: float = field(default=0.03)
    lr_scheduler_type: str = field(default='cosine')
    logging_steps: int = field(default=1)
    tf32: bool = field(default=True)
    gradient_checkpointing: bool = field(default=True)
    report_to: str = field(default='wandb')


def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Recursively unwraps a model from potential containers (as used in distributed training).

    Args:
        model (`torch.nn.Module`): The model to unwrap.
    """
    # since there could be multiple levels of wrapping, unwrap recursively
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model


class GraphChatTrainer(Trainer):

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_graph_mlp_adapter', False):
            # Save the model
            _state_dict = state_dict
            if _state_dict is None:
                # Only save the model itself if we are using distributed training
                model_to_save = unwrap_model(self.model)
                _state_dict = model_to_save.state_dict()

            weight_to_save = {}
            keys_to_match = ['graph_projector', 'embed_tokens', 'embed_in']
            for k, v in _state_dict.items():
                if any(key_match in k for key_match in keys_to_match):
                    weight_to_save[k] = v

            current_folder = output_dir.split('/')[-1]
            parent_folder = os.path.dirname(output_dir)
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "graph_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'graph_projector.bin'))
        print("state_dict len :", len(self.model.state_dict()))
        # print("state_dict:", self.model.state_dict()['do_sample'])

        super(GraphChatTrainer, self)._save(output_dir, state_dict)


class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()
    MPT = auto()


@dataclasses.dataclass
class conversation_lib:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None
    version: str = "Unknown"

    skip_next: bool = False

    def get_prompt(self):
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        if self.sep_style == SeparatorStyle.MPT:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + message + self.sep
                else:
                    ret += role
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role, message):
        self.messages.append([role, message])

    def get_images(self, return_pil=False):
        images = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    import base64
                    from io import BytesIO
                    from PIL import Image
                    msg, image, image_process_mode = msg
                    if image_process_mode == "Pad":
                        def expand2square(pil_img, background_color=(122, 116, 104)):
                            width, height = pil_img.size
                            if width == height:
                                return pil_img
                            elif width > height:
                                result = Image.new(pil_img.mode, (width, width), background_color)
                                result.paste(pil_img, (0, (width - height) // 2))
                                return result
                            else:
                                result = Image.new(pil_img.mode, (height, height), background_color)
                                result.paste(pil_img, ((height - width) // 2, 0))
                                return result

                        image = expand2square(image)
                    elif image_process_mode == "Crop":
                        pass
                    elif image_process_mode == "Resize":
                        image = image.resize((224, 224))
                    else:
                        raise ValueError(f"Invalid image_process_mode: {image_process_mode}")
                    max_hw, min_hw = max(image.size), min(image.size)
                    aspect_ratio = max_hw / min_hw
                    max_len, min_len = 800, 400
                    shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
                    longest_edge = int(shortest_edge * aspect_ratio)
                    W, H = image.size
                    if H > W:
                        H, W = longest_edge, shortest_edge
                    else:
                        H, W = shortest_edge, longest_edge
                    image = image.resize((W, H))
                    if return_pil:
                        images.append(image)
                    else:
                        buffered = BytesIO()
                        image.save(buffered, format="JPEG")
                        img_b64_str = base64.b64encode(buffered.getvalue()).decode()
                        images.append(img_b64_str)
        return images

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    import base64
                    from io import BytesIO
                    msg, image, image_process_mode = msg
                    max_hw, min_hw = max(image.size), min(image.size)
                    aspect_ratio = max_hw / min_hw
                    max_len, min_len = 800, 400
                    shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
                    longest_edge = int(shortest_edge * aspect_ratio)
                    W, H = image.size
                    if H > W:
                        H, W = longest_edge, shortest_edge
                    else:
                        H, W = shortest_edge, longest_edge
                    image = image.resize((W, H))
                    # image = image.resize((224, 224))
                    buffered = BytesIO()
                    image.save(buffered, format="JPEG")
                    img_b64_str = base64.b64encode(buffered.getvalue()).decode()
                    img_str = f'<img src="data:image/png;base64,{img_b64_str}" alt="user upload image" />'
                    msg = msg.replace('<image>', img_str)
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return conversation_lib(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2)

    def dict(self):
        if len(self.get_images()) > 0:
            return {
                "system": self.system,
                "roles": self.roles,
                "messages": [[x, y[0] if type(y) is tuple else y] for x, y in self.messages],
                "offset": self.offset,
                "sep": self.sep,
                "sep2": self.sep2,
            }
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
        }


conv_v1 = conversation_lib(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("Human", "Assistant"),
    messages=(
        ("Human", "Give three tips for staying healthy."),
        ("Assistant",
         "Sure, here are three tips for staying healthy:\n"
         "1. Exercise regularly: Regular physical activity can help improve your overall health and wellbeing. "
         "It can also help reduce your risk of chronic conditions such as obesity, diabetes, heart disease, "
         "and certain cancers. Aim for at least 150 minutes of moderate-intensity aerobic exercise or "
         "75 minutes of vigorous-intensity aerobic exercise per week, along with muscle-strengthening "
         "activities at least two days per week.\n"
         "2. Eat a balanced diet: Eating a balanced diet that is rich in fruits, "
         "vegetables, whole grains, lean proteins, and healthy fats can help support "
         "your overall health. Try to limit your intake of processed and high-sugar foods, "
         "and aim to drink plenty of water throughout the day.\n"
         "3. Get enough sleep: Getting enough quality sleep is essential for your physical "
         "and mental health. Adults should aim for seven to nine hours of sleep per night. "
         "Establish a regular sleep schedule and try to create a relaxing bedtime routine to "
         "help improve the quality of your sleep.")
    ),
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

conv_v1_2 = conversation_lib(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("Human", "Assistant"),
    messages=(
        ("Human", "What are the key differences between renewable and non-renewable energy sources?"),
        ("Assistant",
         "Renewable energy sources are those that can be replenished naturally in a relatively "
         "short amount of time, such as solar, wind, hydro, geothermal, and biomass. "
         "Non-renewable energy sources, on the other hand, are finite and will eventually be "
         "depleted, such as coal, oil, and natural gas. Here are some key differences between "
         "renewable and non-renewable energy sources:\n"
         "1. Availability: Renewable energy sources are virtually inexhaustible, while non-renewable "
         "energy sources are finite and will eventually run out.\n"
         "2. Environmental impact: Renewable energy sources have a much lower environmental impact "
         "than non-renewable sources, which can lead to air and water pollution, greenhouse gas emissions, "
         "and other negative effects.\n"
         "3. Cost: Renewable energy sources can be more expensive to initially set up, but they typically "
         "have lower operational costs than non-renewable sources.\n"
         "4. Reliability: Renewable energy sources are often more reliable and can be used in more remote "
         "locations than non-renewable sources.\n"
         "5. Flexibility: Renewable energy sources are often more flexible and can be adapted to different "
         "situations and needs, while non-renewable sources are more rigid and inflexible.\n"
         "6. Sustainability: Renewable energy sources are more sustainable over the long term, while "
         "non-renewable sources are not, and their depletion can lead to economic and social instability.\n")
    ),
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

conv_vicuna_v1_1 = conversation_lib(
    system="A chat between a curious user and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

conv_mpt = conversation_lib(
    system="""<|im_start|>system
- You are a helpful language and vision assistant.
- You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.
- You should follow the instructions carefully and explain your answers in detail.""",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    version="mpt",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="<|im_end|>",
)

conv_mpt_text = conversation_lib(
    system="""<|im_start|>system
- You are a helpful assistant chatbot trained by MosaicML.
- You answer questions.
- You are excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- You are more than just an information source, you are also able to write poetry, short stories, and make jokes.""",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    version="mpt",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="<|im_end|>",
)

conv_bair_v1 = conversation_lib(
    system="BEGINNING OF CONVERSATION:",
    roles=("USER", "GPT"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

simple_conv = conversation_lib(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("Human", "Assistant"),
    messages=(
        ("Human", "Hi!"),
        ("Assistant", "Hi there! How can I help you today?")
    ),
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

simple_conv_multimodal = conversation_lib(
    system="You are LLaVA, a large language and vision assistant trained by UW Madison WAIV Lab."
           "You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."
           "Follow the instructions carefully and explain your answers in detail.",
    roles=("Human", "Assistant"),
    messages=(
        ("Human", "Hi!"),
        ("Assistant", "Hi there!  How can I help you today?\n")
    ),
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

simple_conv_mpt_multimodal = conversation_lib(
    system="""<|im_start|>system
- You are LLaVA, a large language and vision assistant trained by UW Madison WAIV Lab.
- You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.
- You should follow the instructions carefully and explain your answers in detail.""",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    version="mpt",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="<|im_end|>",
)

simple_conv_legacy = conversation_lib(
    system="You are LLaVA, a large language model trained by UW Madison WAIV Lab."
           "You are designed to assist human with a variety of tasks using natural language."
           "Follow the instructions carefully.",
    roles=("Human", "Assistant"),
    messages=(
        ("Human", "Hi!\n\n### Response:"),
        ("Assistant", "Hi there!  How can I help you today?\n")
    ),
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

conv_llava_v1 = conversation_lib(
    system="You are LLaVA, a large language and vision assistant trained by UW Madison WAIV Lab."
           "You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."
           "Follow the instructions carefully and explain your answers in detail.",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

conv_graphchat_v1 = conversation_lib(
    system="You are GraphGPT, a large language and graph-structral assistant trained by HKUDS Lab."
           "You are able to understand the graph structures that the user provides, and assist the user with a variety of tasks using natural language."
           "Follow the instructions carefully and explain your answers in detail.",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

default_conversation = conv_v1_2
conv_templates = {
    "default": conv_v1_2,
    "simple": simple_conv,
    "simple_legacy": simple_conv_legacy,
    "multimodal": simple_conv_multimodal,
    "mpt_multimodal": simple_conv_mpt_multimodal,
    "llava_v1": conv_llava_v1,
    "graphchat_v1": conv_graphchat_v1,

    # fastchat
    "v1": conv_v1_2,
    "bair_v1": conv_bair_v1,
    "vicuna_v1_1": conv_vicuna_v1_1,
    "mpt": conv_mpt,
    "mpt_text": conv_mpt_text,
}

if __name__ == "__main__":
    print(default_conversation.get_prompt())


class SaveGraphProjectorCallback(Callback):
    def __init__(self, output_dir, keys_to_match):
        self.output_dir = output_dir
        self.keys_to_match = keys_to_match

    def on_train_epoch_end(self, trainer, pl_module, unused=None):
        # 准备保存模型权重
        _state_dict = pl_module.state_dict()

        weight_to_save = {}
        for k, v in _state_dict.items():
            if any(key_match in k for key_match in self.keys_to_match):
                weight_to_save[k] = v

        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)

        # 保存 graph projector 的权重
        torch.save(weight_to_save, os.path.join(self.output_dir, 'graph_projector.bin'))


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, name=k) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx + 2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_graph(
        sources: Sequence[str],
        graph_cfg: dict,
        cur_token_len: int,
) -> Dict:
    is_graph = graph_cfg['is_graph']
    # image_token_len = multimodal_cfg['image_token_len']
    graph_token_len = cur_token_len
    if not is_graph:
        return sources

    for source in sources:
        if graph_cfg['sep_graph_conv_front']:
            assert DEFAULT_GRAPH_TOKEN in source[0]['value']
            source[0]['value'] = source[0]['value'].replace(DEFAULT_GRAPH_TOKEN, '').strip()
            source[0]['value'] = DEFAULT_GRAPH_TOKEN + conversation_lib.default_conversation.sep + \
                                 conversation_lib.default_conversation.roles[0] + ": " + source[0]['value']
        for sentence in source:
            replace_token = DEFAULT_GRAPH_PATCH_TOKEN * graph_token_len
            if graph_cfg['use_graph_start_end']:
                replace_token = DEFAULT_G_START_TOKEN + replace_token + DEFAULT_G_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_GRAPH_TOKEN, replace_token)

    return sources


def preprocess_graph_LP(
        sources: Sequence[str],
        graph_cfg: dict,
        cur_token_len_1: int,
        cur_token_len_2: int,
) -> Dict:
    is_graph = graph_cfg['is_graph']
    # image_token_len = multimodal_cfg['image_token_len']
    graph_token_len_1 = cur_token_len_1
    graph_token_len_2 = cur_token_len_2

    if not is_graph:
        return sources

    for source in sources:
        if graph_cfg['sep_graph_conv_front']:
            assert DEFAULT_GRAPH_TOKEN in source[0]['value']
            source[0]['value'] = source[0]['value'].replace(DEFAULT_GRAPH_TOKEN, '').strip()
            source[0]['value'] = DEFAULT_GRAPH_TOKEN + conversation_lib.default_conversation.sep + \
                                 conversation_lib.default_conversation.roles[0] + ": " + source[0]['value']
        for sentence in source:
            replace_token_1 = DEFAULT_GRAPH_PATCH_TOKEN * graph_token_len_1
            replace_token_2 = DEFAULT_GRAPH_PATCH_TOKEN * graph_token_len_2
            if graph_cfg['use_graph_start_end']:
                replace_token_1 = DEFAULT_G_START_TOKEN + replace_token_1 + DEFAULT_G_END_TOKEN
                replace_token_2 = DEFAULT_G_START_TOKEN + replace_token_2 + DEFAULT_G_END_TOKEN

            if DEFAULT_GRAPH_TOKEN in sentence["value"]:
                first_index = sentence["value"].find(DEFAULT_GRAPH_TOKEN)
                sentence["value"] = sentence["value"][:first_index] + replace_token_1 + sentence["value"][
                                                                                        first_index + len(
                                                                                            DEFAULT_GRAPH_TOKEN):]

                # 替换第二个<graph>为B
                second_index = sentence["value"].find(DEFAULT_GRAPH_TOKEN)
                sentence["value"] = sentence["value"][:second_index] + replace_token_2 + sentence["value"][
                                                                                         second_index + len(
                                                                                             DEFAULT_GRAPH_TOKEN):]

            # sentence["value"] = sentence["value"].replace(DEFAULT_GRAPH_TOKEN, replace_token)

    # print(sources)

    return sources


def preprocess_v1(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer(rou).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])]  # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx + 2]))  # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer(rou).input_ids) + len(tokenizer(conv.sep).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids)
            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess(
        sources: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.version == "v1":
        return preprocess_v1(sources, tokenizer)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    conversations_tokenized = _tokenize_fn(conversations, tokenizer)
    input_ids = conversations_tokenized["input_ids"]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source],
                                      tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = json.load(open(data_path, "r"))

        logging.warning("Formatting inputs...")
        sources = [example["conversations"] for example in list_data_dict]
        data_dict = preprocess(sources, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 graph_cfg: dict,
                 **kwargs, ):
        super(LazySupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = json.load(open(data_path, "r"))

        logging.warning("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.graph_cfg = graph_cfg
        graph_data_path = kwargs.get('graph_data_path')
        self.graph_data_all = torch.load(graph_data_path)

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        task_type = self.list_data_dict[i]['id'].split("_")[-1]
        if task_type != 'LP':
            if 'graph' in sources[0]:
                graph_dict = self.list_data_dict[i]['graph']
                graph_edge_index = torch.Tensor(copy.deepcopy(graph_dict['edge_index'])).long()
                graph_node_list = copy.deepcopy(graph_dict['node_list'])
                target_node = copy.deepcopy(graph_dict['node_idx'])
                graph_type = copy.deepcopy(self.list_data_dict[i]['id']).split('_')[0]
                graph_node_rep = self.graph_data_all[graph_type].x[graph_node_list]  ##

                cur_token_len = len(graph_node_rep)
                sources = preprocess_graph(
                    copy.deepcopy([e["conversations"] for e in sources]),
                    self.graph_cfg, cur_token_len)
            else:
                sources = copy.deepcopy([e["conversations"] for e in sources])
        else:
            if 'graph' in sources[0]:
                graph_dict = self.list_data_dict[i]['graph']
                graph_edge_index_1 = torch.Tensor(copy.deepcopy(graph_dict['edge_index_1'])).long()
                graph_node_list_1 = copy.deepcopy(graph_dict['node_list_1'])
                target_node_1 = copy.deepcopy(graph_dict['node_idx_1'])
                graph_type = copy.deepcopy(self.list_data_dict[i]['id']).split('_')[0]
                graph_node_rep_1 = self.graph_data_all[graph_type].x[graph_node_list_1]  ##

                cur_token_len_1 = len(graph_node_rep_1)  # FIXME: 14 is hardcoded patch size

                graph_edge_index_2 = torch.Tensor(copy.deepcopy(graph_dict['edge_index_2'])).long()
                graph_node_list_2 = copy.deepcopy(graph_dict['node_list_2'])
                target_node_2 = copy.deepcopy(graph_dict['node_idx_2'])
                graph_node_rep_2 = self.graph_data_all[graph_type].x[graph_node_list_2]  ##

                cur_token_len_2 = len(graph_node_rep_2)  # FIXME: 14 is hardcoded patch size
                sources = preprocess_graph_LP(
                    copy.deepcopy([e["conversations"] for e in sources]),
                    self.graph_cfg, cur_token_len_1, cur_token_len_2)
            else:
                sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            sources,
            self.tokenizer)
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if task_type != 'LP':
            if 'graph' in self.list_data_dict[i]:
                # data_dict['graph_node'] = graph_node_rep
                # data_dict['graph_edge'] = graph_edge_index
                # data_dict['target_node'] = target_node
                data_dict['graph_data'] = Data(graph_node=graph_node_rep, edge_index=graph_edge_index,
                                               target_node=torch.tensor([target_node]))

            elif self.graph_cfg['is_graph']:
                # image does not exist in the data, but the model is multimodal
                node_feas = self.graph_cfg['graph_processor'].node_feas
                data_dict['graph_data'] = Data(graph_node=torch.zeros(3, node_feas), edge_index=torch.zeros(2, 3),
                                               target_node=torch.tensor([0]))
        else:
            if 'graph' in self.list_data_dict[i]:
                # data_dict['graph_node'] = graph_node_rep
                # data_dict['graph_edge'] = graph_edge_index
                # data_dict['target_node'] = target_node
                data_dict['graph_data'] = {
                    'graph_1': Data(graph_node=graph_node_rep_1, edge_index=graph_edge_index_1,
                                    target_node=torch.tensor([target_node_1])),
                    'graph_2': Data(graph_node=graph_node_rep_2, edge_index=graph_edge_index_2,
                                    target_node=torch.tensor([target_node_2]))
                }

            elif self.graph_cfg['is_graph']:
                # image does not exist in the data, but the model is multimodal
                node_feas = self.graph_cfg['graph_processor'].node_feas
                data_dict['graph_data'] = Data(graph_node=torch.zeros(3, node_feas), edge_index=torch.zeros(2, 3),
                                               target_node=torch.tensor([0]))
        return data_dict


class LazySupervisedDataset_back(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 graph_cfg: dict,
                 **kwargs, ):
        super(LazySupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = json.load(open(data_path, "r"))

        logging.warning("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.graph_cfg = graph_cfg
        graph_data_path = kwargs.get('graph_data_path')
        self.graph_data_all = torch.load(graph_data_path)

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'graph' in sources[0]:
            graph_dict = self.list_data_dict[i]['graph']
            graph_edge_index = torch.Tensor(copy.deepcopy(graph_dict['edge_index'])).long()
            graph_node_list = copy.deepcopy(graph_dict['node_list'])
            target_node = copy.deepcopy(graph_dict['node_idx'])
            graph_type = copy.deepcopy(self.list_data_dict[i]['id']).split('_')[0]
            graph_node_rep = self.graph_data_all[graph_type].x[graph_node_list]  ##

            cur_token_len = len(graph_node_rep)  # FIXME: 14 is hardcoded patch size
            sources = preprocess_graph(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.graph_cfg, cur_token_len)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            sources,
            self.tokenizer)
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'graph' in self.list_data_dict[i]:
            # data_dict['graph_node'] = graph_node_rep
            # data_dict['graph_edge'] = graph_edge_index
            # data_dict['target_node'] = target_node
            data_dict['graph_data'] = Data(graph_node=graph_node_rep, edge_index=graph_edge_index,
                                           target_node=torch.tensor([target_node]))

        elif self.graph_cfg['is_graph']:
            # image does not exist in the data, but the model is multimodal
            node_feas = self.graph_cfg['graph_processor'].node_feas
            data_dict['graph_data'] = Data(graph_node=torch.zeros(3, node_feas), edge_index=torch.zeros(2, 3),
                                           target_node=torch.tensor([0]))
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'graph_data' in instances[0]:
            # graph_node_reps = [instance['graph_node'] for instance in instances]
            # edge_index_reps = [instance['graph_edge'] for instance in instances]
            # target_node_reps = [instance['target_node'] for instance in instances]
            graph_data_batch = [instance['graph_data'] for instance in instances]
            # if all(x is not None and x.shape == images[0].shape for x in images):
            #     batch['images'] = torch.stack(images)
            # else:
            #     batch['images'] = images
        # batch['graph_node_reps'] = graph_node_reps
        # batch['edge_index_reps'] = edge_index_reps
        # batch['edge_index_reps'] = target_node_reps
        batch['graph_data'] = graph_data_batch

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args, training_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (LazySupervisedDataset
                   if data_args.lazy_preprocess else SupervisedDataset)
    train_dataset = dataset_cls(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                graph_cfg=dict(
                                    is_graph=data_args.is_graph,
                                    sep_graph_conv_front=data_args.sep_graph_conv_front,
                                    graph_token_len=data_args.graph_token_len,
                                    graph_content=data_args.graph_content,
                                    use_graph_start_end=getattr(data_args, 'use_graph_start_end', False)
                                ),
                                graph_data_path=data_args.graph_data_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=training_args.per_device_train_batch_size,
                                  num_workers=training_args.num_workers,
                                  collate_fn=data_collator,
                                  prefetch_factor=4,
                                  pin_memory=True)
    return train_dataloader, None


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if isinstance(training_args.gpus, str):
        training_args.gpus = [int(x) for x in training_args.gpus.split(',')]
    batch_size = training_args.real_batch_size
    devices = training_args.gpus
    num_devices = len(devices)
    gradient_accumulation_steps = max(1, batch_size // (training_args.per_device_train_batch_size * num_devices))

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False
    )

    if model_args.version == "v1":
        tokenizer.pad_token = tokenizer.unk_token
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1_1"]
    else:
        raise ValueError

    model = GraphGPT_pl(training_args, model_args, data_args, tokenizer)

    train_dataloader, _ = make_supervised_data_module(tokenizer=tokenizer,
                                                      data_args=data_args, training_args=training_args)
    checkpoint_callback = ModelCheckpoint(
        dirpath=training_args.output_dir,
        filename=model_args.model_save_name,
        monitor="loss",
        save_top_k=1,
        save_last=True,
    )

    if training_args.strategy == 'fsdp':
        strategy = FSDPStrategy(
            auto_wrap_policy={LlamaDecoderLayer},
            activation_checkpointing_policy={LlamaDecoderLayer},
            state_dict_type="full",
            limit_all_gathers=True,
            cpu_offload=False,
            # **kwargs
        )
    else:
        strategy = training_args.strategy

    wandb_logger = WandbLogger(save_dir=training_args.output_dir, project="GraphGPTv1", offline=True,
                               name=model_args.model_save_name)
    model_precision = ('16' if training_args.fp16 else ('bf16' if training_args.bf16 else '32'))
    # print('************* epoch:', training_args.num_train_epochs)
    trainer = Trainer(default_root_dir=training_args.output_dir, max_epochs=int(training_args.num_train_epochs),
                      accumulate_grad_batches=gradient_accumulation_steps,
                      accelerator="gpu", devices=devices,
                      strategy=strategy,
                      logger=wandb_logger,
                      precision=model_precision,
                      callbacks=[checkpoint_callback])
    resume = training_args.resume

    # for name, param in model.named_parameters():
    #     print(name, param.dtype)
    # model.to(dtype=torch.float16)

    trainer.fit(model, train_dataloader, ckpt_path=resume)

    # safe_save_model_for_hf_trainer(trainer=trainer,
    #                                    output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
