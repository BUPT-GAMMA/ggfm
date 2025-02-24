import argparse
import os
import sys
sys.path.append('/home/cwx/workspace/ggfm')
from ggfm.data.text_pair_datasets import TextPairDataset
from ggfm.models.translator_model.translator_qformer import TranslatorQformer
from ggfm.models.translator_model.translator_chatglm import TranslatorCHATGLMIMDB
import time
import logging
import datetime
import os
import math
from tqdm import tqdm
import time
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np

from torch_geometric.data import Data
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import SAGEConv

from transformers import AutoTokenizer, AutoModel
import pickle
from dgl.data.utils import load_graphs
import dgl

from warnings import filterwarnings
filterwarnings("ignore")

# lr_scheduler

def train_GNN(args):
    """
    Train a Graph Neural Network (GNN) using link prediction for pretraining.
    Args:
        args: A namespace object containing the following attributes:
            - data_dir (str): Directory containing the dataset.
            - graph_files (str): Filename of the graph data.
            - graph_node_name_file (str): Filename of the node attributes.
            - bert_dir (str): Directory of the pretrained BERT model.
            - device (str): Device to run the training on (e.g., 'cpu' or 'cuda').
            - node_embedding_output_path (str): Path to save the node embeddings.
    Functions:
        open_pkl_file(file_path):
            Opens a pickle file and returns its content.
        load_dblp_data():
            Loads the DBLP dataset, including the graph, node names, and labels.
        load_imdb_data():
            Loads the IMDB dataset, including the graph, node names, and labels.
        process_data(inputs, prompt='Actor', model_name=args.bert_dir):
            Processes input data using a pretrained BERT model to generate embeddings.
        train():
            Trains the GNN model using link prediction.
        compute_accuracy(outputs, labels):
            Computes the accuracy of the model predictions.
        test():
            Tests the GNN model and computes validation accuracy.
    Returns:
        None
    """
    def open_pkl_file(file_path):
        with open(file_path, 'rb') as file:
            file_content = pickle.load(file)
            return file_content

    def load_dblp_data():
        data_dir = args.data_dir
        def open_pkl_file(file_path):
            with open(file_path, 'rb') as file:
                file_content = pickle.load(file)
                return file_content

        # load graph
        glist, _ = load_graphs(data_dir + args.graph_files)
        g = glist[0]

        # label
        label = g.nodes['author'].data['label']
        for ntype in g.ntypes:
            if (ntype != 'author'):
                g.nodes[ntype].data['has_label'] = torch.full((g.num_nodes(ntype),), False)
            else:
                g.nodes[ntype].data['has_label'] = torch.full((g.num_nodes(ntype),), True)

        # read node attributes (name)
        graph_node_name = open_pkl_file(data_dir + args.graph_node_name_file)
        return g, graph_node_name, label


    def load_imdb_data():
        data_dir = args.data_dir

        # load graph
        glist, _ = load_graphs(data_dir + args.graph_files)
        g = glist[0]

        # label
        label = g.nodes['movie'].data['label']
        for ntype in g.ntypes:
            if (ntype != 'movie'):
                g.nodes[ntype].data['has_label'] = torch.full((g.num_nodes(ntype),), False)
            else:
                g.nodes[ntype].data['has_label'] = torch.full((g.num_nodes(ntype),), True)

        # read node attributes (name)
        graph_node_name = open_pkl_file(data_dir + args.graph_node_name_file)
        return g, graph_node_name, label

    def process_data(inputs, prompt='Actor', model_name=args.bert_dir):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        # Add prompt to enhance semantics
        prompted_texts = [f"{prompt}: {name}" for name in inputs]
        
        inputs = tokenizer(
            prompted_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=32
        )
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Use the average of all tokens (more suitable for short texts than [CLS])
        embeddings = torch.mean(outputs.last_hidden_state, dim=1).numpy()
        return embeddings


    g, graph_node_name, labels = load_dblp_data()
    for etype in g.ntypes:
        emb = process_data(graph_node_name[etype], etype, args.bert_dir)
        g.nodes[etype].data['emb'] = torch.tensor(emb)

    # Convert the graph to a homogeneous graph
    homo_g = dgl.to_homogeneous(g, ndata=['emb','has_label'])
    bert_node_embeddings = homo_g.ndata['emb']
    edge_index = torch.stack(homo_g.edges())
    idx = torch.nonzero(homo_g.ndata['has_label']).squeeze()
    data = Data(x=bert_node_embeddings, edge_index=edge_index, y=labels, train_idx=idx)
    
    # Create a DataLoader for link prediction
    train_loader = LinkNeighborLoader(
        data,
        batch_size=65536,
        shuffle=True,
        neg_sampling_ratio=1.0,
        num_neighbors=[10, 10],
    )

    device = torch.device(args.device)

    data = data.to(device)

    class Net(nn.Module):
        def __init__(self, in_dim, hid_dim, out_dim):
            super(Net, self).__init__()
            self.conv1 = SAGEConv(in_dim, hid_dim)
            self.conv2 = SAGEConv(hid_dim, out_dim)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            x = self.conv2(x, edge_index)

            return x


    model = Net(768, 1024, 768).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    def train():
        model.train()

        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            h = model(batch.x, batch.edge_index)
            h_src = h[batch.edge_label_index[0]]
            h_dst = h[batch.edge_label_index[1]]
            pred = (h_src * h_dst).sum(dim=-1)
            loss = F.binary_cross_entropy_with_logits(pred, batch.edge_label)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * pred.size(0)

        return total_loss / data.num_nodes


    class LogisticRegression(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(LogisticRegression, self).__init__()
            self.linear1 = nn.Linear(input_dim, 512)
            self.linear2 = nn.Linear(512, output_dim) 
            self.act = nn.ReLU()

        def forward(self, x):
            out = self.linear1(x)
            out = self.act(out)
            out = self.linear2(out)
            return out


    def compute_accuracy(outputs, labels):
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == labels).sum().item()
        total = labels.size(0)
        accuracy = correct / total
        return accuracy


    def test():
        with torch.no_grad():
            model.eval()
            out = model(data.x, data.edge_index)

        for epoch in range(1, 501):
            LR_model.train()
            optimizer.zero_grad()
            pred = LR_model(out[data.train_idx])

            label = F.one_hot(data.y, max(data.y)+1).float()

            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()

        LR_model.eval()
        val_outputs = LR_model(out[data.train_idx])
        val_acc = compute_accuracy(val_outputs, data.y)

        return val_acc


    times = []
    best_acc = 0
    for epoch in range(10):
        start = time.time()
        input_dim = 768
        output_dim = torch.max(data.y) + 1
        LR_model = LogisticRegression(input_dim, output_dim).to(device)
        optimizer = torch.optim.Adam(LR_model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        loss = train()
        acc = test()
        print("loss:", loss)

    print(acc)

    out = model(data.x, data.edge_index)[homo_g.ndata['has_label']]
    torch.save(out, args.node_embedding_output_path)


def produce(args):
    """
    This function generates text descriptions for nodes and their neighboring nodes, 
    specifically for classification purposes.
    Args:
        args: A namespace object containing various arguments and configurations 
              required for data loading, model initialization, and inference.
    Returns:
        None
    """
    def load_data(dataset='dblp'):
        def open_pkl_file(file_path):
            with open(file_path, 'rb') as file:
                file_content = pickle.load(file)
                return file_content

        data_dir = args.data_dir

        # load graph
        glist, label_dict = load_graphs(data_dir + args.graph_path)
        g = glist[0]

        # 读取节点属性（name）
        graph_node_name = open_pkl_file(data_dir + args.graph_node_name_file)
        print(g)

        return g, graph_node_name

    def get_src_to_dst_df(edge_index):
        # 转换为DataFrame
        edge_df = pd.DataFrame({
            'src_node': edge_index[0].numpy(),
            'dst_node': edge_index[1].numpy()
        })
        
        # 使用groupby获取邻居节点列表
        src_to_dst_dict = edge_df.groupby('src_node')['dst_node'].apply(list).to_dict()
        
        return src_to_dst_dict

    class LLM(nn.Module):
        def __init__(self, args, **kwargs):
            super().__init__()
            self._args = args
            # tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self._args.llm_checkpoint, trust_remote_code=True)
            # model
            self.llm = AutoModel.from_pretrained(self._args.llm_checkpoint, trust_remote_code=True).half().to(device)

        def update_summary(self, node_word_input, neighbor_word_input, node_id, title, summary, chat_src=True, chat_dst=True):
            try:
                if chat_src:
                    response_node, _ = self.llm.chat(self.tokenizer,
                                                            node_word_input,
                                                            history=[])
                else:
                    response_node = node_word_input
                if chat_dst:
                    response_neighbor, _ = self.llm.chat(self.tokenizer,
                                                                neighbor_word_input,
                                                                history=[])
                else:
                    response_neighbor = neighbor_word_input
                summary.append({
                    'node_id': node_id,
                    'title': title,
                    'response_node': response_node,
                    'response_neighbor': response_neighbor
                })
                print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))} paper {node_id+1} title: \"{title}\"")
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    print("CUDA out of memory error detected, skipping this batch")
                    return 
                else:
                    return 
        def inference_chatglm_dblp(self, g, graph_node_name):
            self.llm.eval()

            author_data, paper_data, conf_data, term_data = graph_node_name['author'], graph_node_name['paper'], graph_node_name['conf'], graph_node_name['term']
            author_to_paper, paper_to_author = get_src_to_dst_df(g.edges(etype='write')), get_src_to_dst_df(g.edges(etype='was written by'))
            paper_to_term, term_to_paper = get_src_to_dst_df(g.edges(etype='was published in')), get_src_to_dst_df(g.edges(etype='publish'))
            paper_to_conf, conf_to_paper = get_src_to_dst_df(g.edges(etype='was received by')), get_src_to_dst_df(g.edges(etype='receive'))

            summary = []
            for i, author_title in tqdm(enumerate(author_data)):
                node_id = i
                src_prompt_pre = "The name of this author is as follows: "
                # src_prompt = '\n please summarize this author and list five key words of this author. All answers are in English and No Chinese in your answer'
                node_word_input = src_prompt_pre + author_title
                # if len(node_word_input[0]) > 3000- len(src_prompt):
                #     node_word_input = node_word_input[:3000-len(src_prompt)]

                dst_prompt_pre = '\n The authors\' name are provided as follows: '
                dst_prompt = "\n Please summarize the topic and content of these papers. All answers are in English and No Chinese in your answer"
                dst_title_abstract = ''
                for neighbor_paper_id in author_to_paper[i]:
                    conf_id = paper_to_conf[neighbor_paper_id][0]
                    term_ids = paper_to_term[neighbor_paper_id]
                    terms = [term_data[term_id] for term_id in term_ids]
                    term_str = ', '.join(terms)
                    dst_title_abstract += conf_data[conf_id] +': '+ paper_data[neighbor_paper_id] + ': ' + term_str + '\n'

                neighbor_word_input = dst_prompt_pre + dst_title_abstract
                if len(neighbor_word_input[0]) > 3000-len(dst_prompt):
                    neighbor_word_input = neighbor_word_input[:3000-len(dst_prompt)]
                neighbor_word_input += dst_prompt
                self.update_summary(node_word_input, neighbor_word_input, node_id, author_title, summary, chat_src=False, chat_dst=True)
    
            summary_df = pd.DataFrame(summary)
            embeddings = torch.load(args.node_embedding_output_path).to('cpu')
            new_data = []
            for _, row in summary_df.iterrows():
                node_id = int(row['node_id'])
                embedding = np.array(embeddings[node_id].detach())
                str_array = [str(num) for num in embedding]
                str_representation = ", ".join(str_array)
                title = row['title']

                new_data.append({
                    'node_id': node_id,
                    'embedding':str_representation ,
                    'paper_summary':row['response_node'],
                    'citepapers_summary':row['response_neighbor'],
                    'title':title
                    })
            summary_embeddings = pd.DataFrame(new_data)
            summary_embeddings.to_csv(args.produce_output_csv,index=False)

        def inference_chatglm_imdb(self, g, graph_node_name):
            self.llm.eval()

            movie_data, actor_data, director_data = graph_node_name['movie'], graph_node_name['actor'], graph_node_name['director']

            movie_to_actor, movie_to_director = get_src_to_dst_df(g.edges(etype='was acted by')), get_src_to_dst_df(g.edges(etype='was directed by'))
            actor_to_movie, director_to_movie = get_src_to_dst_df(g.edges(etype='acted in')), get_src_to_dst_df(g.edges(etype='directed'))

            node_id = -1
            summary = []

            for i, movie_title in tqdm(enumerate(movie_data)):
                node_id += 1
                src_prompt_pre = "The title of this movie is as follows: "
                src_prompt = '\n please summarize this movie and list three key words of this movie. All answers are in English and No Chinese in your answer'
                node_word_input = src_prompt_pre + movie_title
                if len(node_word_input[0]) > 3000- len(src_prompt):
                    node_word_input = node_word_input[:3000-len(src_prompt)]
                node_word_input += src_prompt

                dst_prompt_pre = '\n The actors and dirctors are provided as follows: '
                
                # dst_prompt = "\n Please summarize the topic and content of these movies. All answers are in English and No Chinese in your answer"
                for neighbor_actor_id in movie_to_actor[i]:
                    dst_title_abstract = 'actors: ' + actor_data[neighbor_actor_id] + '\n'
                
                for neighbor_director_id in movie_to_director[i]:
                    dst_title_abstract = 'directors: ' + director_data[neighbor_director_id] + '\n'

                neighbor_word_input = dst_prompt_pre + dst_title_abstract

                self.update_summary(node_word_input, neighbor_word_input, node_id, movie_title, summary, chat_src=True, chat_dst=False)

            summary_df = pd.DataFrame(summary)
            embeddings = torch.load(args.embed).to('cpu')
            new_data = []
            for _, row in summary_df.iterrows():
                node_id = int(row['node_id'])
                embedding = np.array(embeddings[node_id].detach())
                str_array = [str(num) for num in embedding]
                str_representation = ", ".join(str_array)
                title = row['title']

                new_data.append({
                    'node_id': node_id,
                    'embedding':str_representation ,
                    'paper_summary':row['response_node'],
                    'citepapers_summary':row['response_neighbor'],
                    'title':title
                    })
            summary_embeddings = pd.DataFrame(new_data)
            summary_embeddings.to_csv(args.produce_output_csv,index=False)

    logging.info("Main arguments:")
    for k, v in args.__dict__.items():
        logging.info("{}={}".format(k, v))
    logging.info("device type: {}".format(torch.cuda.get_device_name(torch.cuda.current_device())))

    # load model
    model = LLM(args)
    logging.info('start inference')
    g, graph_node_name= load_data('dblp')
    model.inference_chatglm_dblp(g, graph_node_name)

class LinearWarmupStepLRScheduler:
    def __init__(
        self,
        optimizer,
        max_epoch,
        min_lr,
        init_lr,
        decay_rate=1,
        warmup_start_lr=-1,
        warmup_steps=0,
        **kwargs
    ):
        self.optimizer = optimizer

        self.max_epoch = max_epoch
        self.min_lr = min_lr

        self.decay_rate = decay_rate

        self.init_lr = init_lr
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr if warmup_start_lr >= 0 else init_lr

    def step(self, cur_epoch, cur_step):
        if cur_epoch == 0:
            warmup_lr_schedule(
                step=cur_step,
                optimizer=self.optimizer,
                max_step=self.warmup_steps,
                init_lr=self.warmup_start_lr,
                max_lr=self.init_lr,
            )
        else:
            step_lr_schedule(
                epoch=cur_epoch,
                optimizer=self.optimizer,
                init_lr=self.init_lr,
                min_lr=self.min_lr,
                decay_rate=self.decay_rate,
            )


class LinearWarmupCosineLRScheduler:
    def __init__(
        self,
        optimizer,
        max_epoch,
        min_lr,
        init_lr,
        warmup_steps=0,
        warmup_start_lr=-1,
        **kwargs
    ):
        self.optimizer = optimizer

        self.max_epoch = max_epoch
        self.min_lr = min_lr

        self.init_lr = init_lr
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr if warmup_start_lr >= 0 else init_lr

    def step(self, cur_epoch, cur_step):
        # assuming the warmup iters less than one epoch
        if cur_epoch == 0:
            warmup_lr_schedule(
                step=cur_step,
                optimizer=self.optimizer,
                max_step=self.warmup_steps,
                init_lr=self.warmup_start_lr,
                max_lr=self.init_lr,
            )
        else:
            cosine_lr_schedule(
                epoch=cur_epoch,
                optimizer=self.optimizer,
                max_epoch=self.max_epoch,
                init_lr=self.init_lr,
                min_lr=self.min_lr,
            )


def cosine_lr_schedule(optimizer, epoch, max_epoch, init_lr, min_lr):
    """Decay the learning rate"""
    lr = (init_lr - min_lr) * 0.5 * (
        1.0 + math.cos(math.pi * epoch / max_epoch)
    ) + min_lr
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def warmup_lr_schedule(optimizer, step, max_step, init_lr, max_lr):
    """Warmup the learning rate"""
    lr = min(max_lr, init_lr + (max_lr - init_lr) * step / max(max_step, 1))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def step_lr_schedule(optimizer, epoch, init_lr, min_lr, decay_rate):
    """Decay the learning rate"""
    lr = max(min_lr, init_lr * (decay_rate**epoch))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def _load_checkpoint(self, filename):
        """
        Resume from a checkpoint.
        """

        if os.path.isfile(filename):
            checkpoint = torch.load(filename, map_location=self.device)
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]
        self.unwrap_dist_model(self.model).load_state_dict(state_dict)

        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scaler and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])

        self.start_epoch = checkpoint["epoch"] + 1
        logging.info("Resume checkpoint from {}".format(filename))


def _train_inner_loop(
        epoch,
        iters_per_epoch,
        model,
        data_loader,
        optimizer,
        scaler,
        lr_scheduler,
        start_iters=None,
        accum_grad_iters=1,
    ):
        """


        An inner training loop compatible with both epoch-based and iter-based training.

        When using epoch-based, training stops after one epoch; when using iter-based,
        training stops after #iters_per_epoch iterations.
        """
        use_amp = scaler is not None

        if not hasattr(data_loader, "__next__"):
            # convert to iterator if not already
            data_loader = iter(data_loader)

        # if iter-based runner, schedule lr based on inner epoch.
        logging.info(
            "Start training epoch {}, {} iters per inner epoch.".format(
                epoch, iters_per_epoch
            )
        )
        if start_iters is None:
            # epoch-based runner
            inner_epoch = epoch
        else:
            # In iter-based runner, we schedule the learning rate based on iterations.
            inner_epoch = start_iters // iters_per_epoch

        current_step = 0
        for network_input in tqdm(data_loader):
            lr_scheduler.step(cur_epoch=inner_epoch, cur_step=current_step) # TODO：换一个torch自带的lr_scheduler

            with torch.cuda.amp.autocast():
                loss = model(network_input)["loss"]

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # 然后检查梯度是否溢出
            # for param in model.parameters():
            #     if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
            #         print("NaN or Inf detected in gradients")
            #         break

            # update gradients every accum_grad_iters iterations
            if (current_step + 1) % accum_grad_iters == 0:
                if use_amp:
                    # 混合精度相关操作
                    scaler.unscale_(optimizer)  # 梯度裁剪前必须unscale
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # 常规训练流程
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                optimizer.zero_grad()

            current_step += 1


        return loss.item()


def train_epoch(model, epoch, iters_per_epoch, data_loader, optimizer, scaler, lr_scheduler, accum_grad_iters):
    _train_inner_loop(
        epoch=epoch,
        iters_per_epoch=iters_per_epoch,
        model=model,
        data_loader=data_loader,
        optimizer=optimizer,
        scaler=scaler,
        lr_scheduler=lr_scheduler,
        accum_grad_iters=accum_grad_iters,
    )


def _save_checkpoint(model, optimizer, scaler, output_dir, cur_epoch, is_best=False):
    """
    Save the checkpoint at the current epoch.
    """
    model_no_ddp = model
    param_grad_dic = {
        k: v.requires_grad for (k, v) in model_no_ddp.named_parameters()
    }
    state_dict = model_no_ddp.state_dict()
    for k in list(state_dict.keys()):
        if k in param_grad_dic.keys() and not param_grad_dic[k]:
            # delete parameters that do not require gradient
            del state_dict[k]
    save_obj = {
        "model": state_dict,
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler else None,
        "epoch": cur_epoch,
    }
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_to = os.path.join(
        output_dir,
        "checkpoint_{}.pth".format("best" if is_best else cur_epoch),
    )
    logging.info("Saving checkpoint at epoch {} to {}.".format(cur_epoch, save_to))
    torch.save(save_obj, save_to)


def train_stage_1(args):
    """
    Train the model for stage 1 using the provided arguments.
    Args:
        args: An object containing the following attributes:
            - lr_sched_stage_1: Learning rate scheduler type for stage 1.
            - init_lr_stage_1: Initial learning rate for stage 1.
            - min_lr_stage_1: Minimum learning rate for stage 1.
            - warmup_lr_stage_1: Warmup learning rate for stage 1.
            - weight_decay_stage_1: Weight decay for stage 1.
            - max_epoch_stage_1: Maximum number of epochs for stage 1.
            - batch_size_train_stage_1: Batch size for training in stage 1.
            - warmup_steps_stage_1: Number of warmup steps for stage 1.
            - accum_grad_iters_stage_1: Number of gradient accumulation iterations for stage 1.
            - max_length_stage_1: Maximum sequence length for stage 1.
            - vocab_size_stage_1: Vocabulary size for stage 1.
            - output_dir_stage_1: Output directory for stage 1.
            - datasets_dir_stage_1: Directory containing datasets for stage 1.
            - device: Device to use for training (e.g., 'cuda' or 'cpu').
    Returns:
        None
    """
    print('train_stage_1')
    lr_sched = args.lr_sched_stage_1
    init_lr = args.init_lr_stage_1
    min_lr = args.min_lr_stage_1
    warmup_lr = args.warmup_lr_stage_1
    weight_decay = args.weight_decay_stage_1
    max_epoch = args.max_epoch_stage_1
    batch_size_train = args.batch_size_train_stage_1
    warmup_steps = args.warmup_steps_stage_1
    accum_grad_iters = args.accum_grad_iters_stage_1
    max_length = args.max_length_stage_1
    vocab_size = args.vocab_size_stage_1
    output_dir = args.output_dir_stage_1
    mode = 'train'
    datasets_dir = args.datasets_dir_stage_1
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")


    # create dataset
    train_set = TextPairDataset(datasets_dir, max_length, vocab_size, mode)

    # create datalaoder
    dataloader = DataLoader(train_set,batch_size=batch_size_train,pin_memory=True,sampler=None,drop_last=True)

    # create model
    model = TranslatorQformer(num_features=768,
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
        max_txt_len=32,)
    model = model.to(device)
    
    # optimizer
    p_wd, p_non_wd = [], []
    num_parameters = 0 

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue  # frozen weights
        if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
            p_non_wd.append(p)
        else:
            p_wd.append(p)
    num_parameters += p.data.nelement()
    optim_params = [
            {
                "params": p_wd,
                "weight_decay": float(weight_decay),
            },
            {"params": p_non_wd, "weight_decay": 0},
        ]
    optimizer = torch.optim.AdamW(
        optim_params,
        lr = float(init_lr),
        weight_decay=float(weight_decay),
        betas=(0.9, 0.999),
    )

    # scaler
    scaler = torch.cuda.amp.GradScaler()

    # lr_scheduler
    if lr_sched == "linear_warmup_cosine_lr":
        lr_scheduler = LinearWarmupCosineLRScheduler(
            optimizer=optimizer,
            max_epoch=max_epoch,
            min_lr=min_lr,
            init_lr=init_lr,
            warmup_steps=warmup_steps,
            warmup_start_lr=warmup_lr,
        )
    elif lr_sched == "linear_warmup_step_lr":
        lr_scheduler = LinearWarmupStepLRScheduler(
            optimizer=optimizer,
            max_epoch=max_epoch,
            min_lr=min_lr,
            init_lr=init_lr,
            decay_rate=0.1,
            warmup_steps=warmup_steps,
            warmup_start_lr=warmup_lr,
        )
    else:
        raise ValueError("Unknown lr scheduler {}".format(lr_sched))

    # train_model
    start_time = time.time()
    best_agg_metric = 0
    best_epoch = 0
    mode = 'train'  # Mode for dataset, replace with actual mode if needed
    evaluate_only = False
    resume_ckpt_path = None
    start_epoch = 0
    valid_splits = []  # List of validation splits, replace with actual splits if needed
    test_splits = []
    iters_per_epoch = train_set.row_count // batch_size_train


    # resume from checkpoint if specified
    if not evaluate_only and resume_ckpt_path is not None:
        _load_checkpoint(resume_ckpt_path)

    for cur_epoch in tqdm(range(start_epoch, max_epoch)):
        # training phase
        if not evaluate_only:
            logging.info("Start training")
            train_loss = train_epoch(model, cur_epoch, iters_per_epoch, dataloader, optimizer, scaler, lr_scheduler, accum_grad_iters)

        # evaluation phase
        if len(valid_splits) > 0:
            for split_name in valid_splits:
                logging.info("Evaluating on {}.".format(split_name))

                val_log = eval_epoch(
                    split_name=split_name, cur_epoch=cur_epoch
                )
                if val_log is not None:

                    agg_metrics = val_log["agg_metrics"]
                    if agg_metrics > best_agg_metric and split_name == "val":
                        best_epoch, best_agg_metric = cur_epoch, agg_metrics

                        _save_checkpoint(model, optimizer, scaler, output_dir, cur_epoch, is_best=True)

                    val_log.update({"best_epoch": best_epoch})

        else:
            # if no validation split is provided, we just save the checkpoint at the end of each epoch.
            if not evaluate_only:
                _save_checkpoint(model, optimizer, scaler, output_dir, cur_epoch, is_best=False)# TODO:改为保存模型

        if evaluate_only:
            break

    # testing phase
    test_epoch = "best" if len(valid_splits) > 0 else cur_epoch

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info("Training time {}".format(total_time_str))


def train_stage_2(args):
    """
    Train the model for stage 2 using the provided arguments.
    Args:
        args: An object containing the following attributes:
            - lr_sched_stage_2: Learning rate scheduler type for stage 2.
            - init_lr_stage_2: Initial learning rate for stage 2.
            - min_lr_stage_2: Minimum learning rate for stage 2.
            - warmup_lr_stage_2: Warmup learning rate for stage 2.
            - weight_decay_stage_2: Weight decay for stage 2.
            - max_epoch_stage_2: Maximum number of epochs for stage 2.
            - batch_size_train_stage_2: Batch size for training in stage 2.
            - batch_size_eval_stage_2: Batch size for evaluation in stage 2.
            - warmup_steps_stage_2: Number of warmup steps for stage 2.
            - accum_grad_iters_stage_2: Number of gradient accumulation iterations for stage 2.
            - max_length_stage_2: Maximum sequence length for stage 2.
            - vocab_size_stage_2: Vocabulary size for stage 2.
            - output_dir_stage_2: Output directory for stage 2.
            - datasets_dir_stage_2: Directory containing datasets for stage 2.
            - bert_dir_stage_2: Directory of the pretrained BERT model for stage 2.
            - llm_dir_stage_2: Directory of the pretrained LLM model for stage 2.
            - device: Device to use for training (e.g., 'cuda' or 'cpu').
    Returns:
        None
    """
    print('train_stage_2')
    lr_sched = args.lr_sched_stage_2
    init_lr = args.init_lr_stage_2
    min_lr = args.min_lr_stage_2
    warmup_lr = args.warmup_lr_stage_2
    bert_dir = args.bert_dir_stage_2
    llm_dir = args.llm_dir_stage_2
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    

    # create dataset
    train_set = TextPairDataset(datasets_dir, max_length, vocab_size, mode)

    # create datalaoder
    dataloader = DataLoader(train_set,batch_size=batch_size_train,pin_memory=True,sampler=None,drop_last=True)

    # create model
    model = TranslatorCHATGLMIMDB(
        llm_dir=llm_dir,
        bert_dir=bert_dir,
        num_features=768,
        num_query_token=32,
        chatglm2_model="",
        max_txt_len=2048,)
    model = model.to(device)
    
    # optimizer
    p_wd, p_non_wd = [], []
    num_parameters = 0 

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue  # frozen weights
        if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
            p_non_wd.append(p)
        else:
            p_wd.append(p)
    num_parameters += p.data.nelement()
    optim_params = [
            {
                "params": p_wd,
                "weight_decay": float(weight_decay),
            },
            {"params": p_non_wd, "weight_decay": 0},
        ]
    optimizer = torch.optim.AdamW(
        optim_params,
        lr = float(init_lr),
        weight_decay=float(weight_decay),
        betas=(0.9, 0.999),
    )

    # scaler
    scaler = torch.cuda.amp.GradScaler()

    # lr_scheduler
    if lr_sched == "linear_warmup_cosine_lr":
        lr_scheduler = LinearWarmupCosineLRScheduler(
            optimizer=optimizer,
            max_epoch=max_epoch,
            min_lr=min_lr,
            init_lr=init_lr,
            warmup_steps=warmup_steps,
            warmup_start_lr=warmup_lr,
        )
    elif lr_sched == "linear_warmup_step_lr":
        lr_scheduler = LinearWarmupStepLRScheduler(
            optimizer=optimizer,
            max_epoch=max_epoch,
            min_lr=min_lr,
            init_lr=init_lr,
            decay_rate=0.1,
            warmup_steps=warmup_steps,
            warmup_start_lr=warmup_lr,
        )
    else:
        raise ValueError("Unknown lr scheduler {}".format(lr_sched))

    # train_model
    start_time = time.time()
    best_agg_metric = 0
    best_epoch = 0
    mode = 'train'  # Mode for dataset, replace with actual mode if needed
    evaluate_only = False
    resume_ckpt_path = None
    start_epoch = 0
    valid_splits = []  # List of validation splits, replace with actual splits if needed
    iters_per_epoch = train_set.row_count // batch_size_train


    # resume from checkpoint if specified
    if not evaluate_only and resume_ckpt_path is not None:
        _load_checkpoint(resume_ckpt_path)

    for cur_epoch in range(start_epoch, max_epoch):
        # training phase
        if not evaluate_only:
            logging.info("Start training")
            train_loss = train_epoch(model, cur_epoch, iters_per_epoch, dataloader, optimizer, scaler, lr_scheduler, accum_grad_iters)

        # evaluation phase
        if len(valid_splits) > 0:
            for split_name in valid_splits:
                logging.info("Evaluating on {}.".format(split_name))

                val_log = eval_epoch(
                    split_name=split_name, cur_epoch=cur_epoch
                )
                if val_log is not None:

                    agg_metrics = val_log["agg_metrics"]
                    if agg_metrics > best_agg_metric and split_name == "val":
                        best_epoch, best_agg_metric = cur_epoch, agg_metrics

                        _save_checkpoint(model, optimizer, scaler, output_dir, cur_epoch, is_best=True)

                    val_log.update({"best_epoch": best_epoch})

        else:
            # if no validation split is provided, we just save the checkpoint at the end of each epoch.
            if not evaluate_only:
                _save_checkpoint(model, optimizer, scaler, output_dir, cur_epoch, is_best=False)# TODO:改为保存模型

        if evaluate_only:
            break

    # testing phase
    test_epoch = "best" if len(valid_splits) > 0 else cur_epoch
    # evaluate(test_splits, cur_epoch=test_epoch, skip_reload=evaluate_only)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info("Training time {}".format(total_time_str))


def argsparser():
    parser = argparse.ArgumentParser(description="Training Script for Graph Neural Networks and Models")

    # =========train_GNN=========
    parser.add_argument('--data_dir', type=str, default="./data/new_data/dblp/", help="Directory for dataset")
    parser.add_argument('--graph_file', type=str, default="graph.bin", help="Graph file name")
    parser.add_argument('--graph_node_name_file', type=str, default='graph_node.pkl', help="Node name file")
    parser.add_argument('--node_embedding_output_path', type=str, default="./data/graphsage_node_embeddings_dblp.pt", help="Path to save output embeddings")
    
    # =========produce=========
    parser.add_argument('--produce_dataset', type=str, default='dblp', help="Dataset name for produce function")
    parser.add_argument('--produce_output_csv', type=str, default='/home/cwx/workspace/ggfm/data/summary_embeddings_dblp.csv', help="Output CSV path for produce function")
    
    # =========train_stage_1=========
    # Arguments for stage 1 (TextPairDataset and TranslatorQformer)
    parser.add_argument('--lr_sched_stage_1', type=str, default="linear_warmup_cosine_lr", choices=["linear_warmup_cosine_lr", "linear_warmup_step_lr"], help="LR scheduler type for stage 1")
    parser.add_argument('--init_lr_stage_1', type=float, default=1e-4, help="Initial learning rate for stage 1")
    parser.add_argument('--min_lr_stage_1', type=float, default=1e-5, help="Minimum learning rate for stage 1")
    parser.add_argument('--warmup_lr_stage_1', type=float, default=1e-6, help="Warmup learning rate for stage 1")
    parser.add_argument('--weight_decay_stage_1', type=float, default=0.05, help="Weight decay for stage 1")
    parser.add_argument('--max_epoch_stage_1', type=int, default=10, help="Maximum number of epochs for stage 1")
    parser.add_argument('--batch_size_train_stage_1', type=int, default=8, help="Batch size for training in stage 1")
    parser.add_argument('--warmup_steps_stage_1', type=int, default=5000, help="Warmup steps for stage 1")
    parser.add_argument('--accum_grad_iters_stage_1', type=int, default=32, help="Gradient accumulation steps for stage 1")
    parser.add_argument('--max_length_stage_1', type=int, default=512, help="Maximum sequence length for stage 1")
    parser.add_argument('--vocab_size_stage_1', type=int, default=100000, help="Vocabulary size for stage 1")
    parser.add_argument('--output_dir_stage_1', type=str, default='output', help="Output directory for checkpoints in stage 1")
    parser.add_argument('--datasets_dir_stage_1', type=str, default="/home/cwx/workspace/ggfm/data/summary_embeddings_dblp_author_only.csv", help="Path to datasets directory for stage 1")

    # =========train_stage_2=========
    # Arguments for stage 2 (TranslatorCHATGLMIMDB)
    parser.add_argument('--lr_sched_stage_2', type=str, default="linear_warmup_cosine_lr", choices=["linear_warmup_cosine_lr", "linear_warmup_step_lr"], help="LR scheduler type for stage 2")
    parser.add_argument('--init_lr_stage_2', type=float, default=1e-4, help="Initial learning rate for stage 2")
    parser.add_argument('--min_lr_stage_2', type=float, default=1e-5, help="Minimum learning rate for stage 2")
    parser.add_argument('--warmup_lr_stage_2', type=float, default=1e-6, help="Warmup learning rate for stage 2")
    parser.add_argument('--weight_decay_stage_2', type=float, default=0.05, help="Weight decay for stage 2")
    parser.add_argument('--max_epoch_stage_2', type=int, default=1, help="Maximum number of epochs for stage 2")
    parser.add_argument('--batch_size_train_stage_2', type=int, default=1, help="Batch size for training in stage 2")
    parser.add_argument('--batch_size_eval_stage_2', type=int, default=64, help="Batch size for evaluation in stage 2")
    parser.add_argument('--warmup_steps_stage_2', type=int, default=5000, help="Warmup steps for stage 2")
    parser.add_argument('--accum_grad_iters_stage_2', type=int, default=32, help="Gradient accumulation steps for stage 2")
    parser.add_argument('--max_length_stage_2', type=int, default=1024, help="Maximum sequence length for stage 2")
    parser.add_argument('--vocab_size_stage_2', type=int, default=100000, help="Vocabulary size for stage 2")
    parser.add_argument('--output_dir_stage_2', type=str, default='output', help="Output directory for checkpoints in stage 2")
    parser.add_argument('--datasets_dir_stage_2', type=str, default="/home/cwx/workspace/ggfm/data/summary_embeddings_dblp.csv", help="Path to datasets directory for stage 2")
    parser.add_argument('--llm_dir_stage_2', type=str, default='/home/cwx/workspace/ggfm/GraphTranslator/Translator/models/chatglm2-6b', help="Path to LLM model directory for stage 2")
    parser.add_argument('--bert_dir_stage_2', type=str, default='/home/cwx/workspace/ggfm/GraphTranslator/Translator/models/bert-base-uncased', help="Path to BERT model directory for stage 2")

    # =========General Arguments=========
    parser.add_argument('--device', type=str, default="cuda:3", help="Device to use for training (e.g., 'cuda:0')")
    parser.add_argument('--log_freq', type=int, default=50, help="Frequency of logging during training")
    parser.add_argument('--resume_ckpt_path', type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument('--evaluate_only', action='store_true', help="Whether to evaluate only (no training)")


    return parser.parse_args()

if __name__ == '__main__':
    args = argsparser()
    train_GNN()
    produce()
    train_stage_1()
    train_stage_2()
    