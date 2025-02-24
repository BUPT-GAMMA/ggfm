import re
import pandas as pd

import numpy as np
import argparse
import random
import sys
import os

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
sys.path.append('/home/cwx/workspace/ggfm')
from ggfm.models.translator_model.translator_chatglm import TranslatorCHATGLMIMDB

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))
from ggfm.data.text_pair_datasets import TextPairDataset
from dgl.data.utils import load_graphs
from ggfm.models import *
from tqdm import tqdm
from sklearn.metrics import f1_score


torch.backends.cuda.matmul.allow_tf32 = True


def setup_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True
        

def generate(model, train_set, data_loader, generate_prompt, pred_dir, scaler):
    """
    An inner training loop compatible with both epoch-based and iter-based training.

    When using epoch-based, training stops after one epoch; when using iter-based,
    training stops after #iters_per_epoch iterations.
    """
    use_amp = scaler is not None
    if not hasattr(data_loader, "__next__"):
        data_loader = iter(data_loader)

    pred_txt = open(pred_dir, 'w')

    for network_input in tqdm(data_loader):
        with torch.cuda.amp.autocast(enabled=use_amp):
            ChatGLM_response = model.generate(network_input, generate_prompt)

        for i in range(len(ChatGLM_response)):
            id = str(network_input[0][i].detach().cpu().numpy())
            ori_desc = network_input[2][i].replace('\n', '\\n').replace('\t', '\\t')
            pred = ChatGLM_response[i].replace('\n', '\\n').replace('\t', '\\t')
            pred_txt.write(id+'\t'+ori_desc+'\t'+pred+'\n')

    pred_txt.close()

def translator_generate(args):
    batch_size_eval = args.batch_size
    datasets_dir = f"/home/cwx/workspace/ggfm/data/summary_embeddings_{args.datasets}_{args.typee}.csv"
    max_length = 2048
    bert_dir = '/home/cwx/workspace/ggfm/ggfm/models/bert-base-uncased'
    llm_dir = '/home/cwx/workspace/ggfm/ggfm/models/chatglm2-6b'
    mode = "train"
    vocab_size = 100000

    if args.datasets == 'dblp':
        examples = [
            'Example 1:\n{"category":"0","reason":"Consistent publications in CVPR with computer vision focus"}\n',
            'Example 2:\n{"category":"1","reason":"Multiple ACL papers on language modeling"}\n',
            'Example 3:\n{"category":"2","reason":"NeurIPS papers centered on reinforcement learning"}\n',
            'Example 4:\n{"category":"0","reason":"ICML works on image generation techniques"}\n',
            'Example 5:\n{"category":"1","reason":"EMNLP research on semantic parsing"}\n'
        ]
        if args.num_example>0:
            examples = '\n'.join(examples[:args.num_example])
        else:
            examples = ''
        generate_prompt = [
            '\nQuestion: Analyze the research focus of author <{}> based on their papers:',
            '\nPaperlist format: conf: paper_title: term\n{}',
            '\nSelect the single best category from: <0, 1, 2>',
            f'Output only 1 numeric category with reasoning. Examples: {examples}. Use JSON format with keys: category, reason.',
            'Round 0:\n\nQuestion: Analyze author {} with papers:\n{}\n\nAnswer:{} \n\nRound 1:\n{}'
        ]
    elif args.datasets == 'imdb':
        examples = [
            'Example 1:\n{"category":"action","reason":"Dominant combat sequences"}\n',
            'Example 2:\n{"category":"drama","reason":"Emotional relationship focus"}\n',
            'Example 3:\n{"category":"comedy","reason":"Meta humor throughout"}\n',
            'Example 4:\n{"category":"action","reason":"Continuous survival sequences"}\n',
            'Example 5:\n{"category":"action","reason":"Bullet-time combat scenes"}\n'
        ]
        if args.num_example>0:
            examples = '\n'.join(examples[:args.num_example])
        else:
            examples = ''
        generate_prompt = [
            '\nQuestion: Please summarize the topic and content of the paper and its citations in English. \nAnswer:',
            '\nQuestion: Based on the summary of the above movie titled <{}>, carefully analyze its core elements and select the single best match category from these options:',
            'categories: <comedy, action, drama>',
            f'Output only 1 category with your reasoning. Answer example: {examples}. Use JSON format with keys: category, reason. Use only English and the specified categories.\n\nAnswer:',
            'Round 0:\n\nQuestion:We are exploring the movie titled {}. \nSummarize the movie\'s content and citations in English \n\nAnswer:{} \n\nRound 1:\n{}'
        ]
    pred_dir = f"/home/cwx/workspace/ggfm/data/pred_{args.output_dir}_{args.datasets}.txt"
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # create dataset
    train_set = TextPairDataset(datasets_dir, max_length, vocab_size, mode)

    # create datalaoder
    dataloader = DataLoader(train_set,batch_size=batch_size_eval,pin_memory=True,sampler=None,drop_last=True)
    # create model
    model = TranslatorCHATGLMIMDB(
        llm_dir=llm_dir,
        bert_dir=bert_dir,
        num_features=768,
        num_query_token=32,
        chatglm2_model="",
        max_txt_len=2048,)
    model = model.to(device)

    model.eval()

    scaler = torch.cuda.amp.GradScaler()

    generate(model, train_set, dataloader, generate_prompt, pred_dir, scaler)


def get_topk_predictions(node_dict, k):
    topk_predictions = []
    for node, probabilities in node_dict.items():
        try:
            topk_predictions.append(probabilities[0])
        except:
            topk_predictions.append(-1)
    return topk_predictions


def legality_rate(node2pred, args):

    if args.datasets == 'dblp':
        patterns = ["0", "1", "2"]
        label_map = {"0": 0, "1": 1, "2": 2}
    elif args.datasets == 'imdb':
        patterns = [ "action","comedy", "drama", ]
        label_map = {"action": 0, "comedy": 1, "drama": 2}
    print("Total class number:", len(patterns))
    assert len(patterns) == len(label_map), "patterns and label_map should have the same size"

    count = 0
    node2digitallabel = {}
    for node, pred_list in node2pred.items():
        matches = []
        for pred in pred_list:
            for pattern in patterns:
                match_label = re.findall(pattern, pred)
                if len(match_label) > 2:
                    match_label = list(set(match_label))
                    label1 = label_map[match_label[0]]
                    matches.append(label1)
                elif len(match_label) == 2:
                    label1 = label_map[match_label[0]]
                    label2 = label_map[match_label[1]]
                    if label1 != label2:
                        print("error")
                    else:
                        matches.append(label1)
                elif len(match_label) == 1:
                    label1 = label_map[match_label[0]]
                    matches.append(label1)
        matches = list(set(matches))
        node2digitallabel[int(node)] = list(set(matches))
        if len(matches)>0:
            count += 1

    print(f"Total sample number: {count}")
    print(f"Legality rate: {round(100*count/len(node2pred),2) if len(node2pred) > 0 else 0.0}%")

    return node2digitallabel, count


def read_data(label_file, pred_file, args):
    # df_node2label = pd.read_csv(label_file)
    graph, label_dict = load_graphs(label_file)
    g = graph[0]
    if args.datasets == 'dblp':
        ty = 'author'
    elif args.datasets == 'imdb':
        ty = 'movie'
    node2label = g.nodes[ty].data['label'][torch.range(0, 1000).long()]
    # node2label = dict(zip(df_node2label['node_id'], df_node2label['digital_label']))
    df_pred = pd.read_csv(pred_file, sep='\t', names=['node', 'summary', 'pred'])
    node2pred = {}
    for _, row in df_pred.iterrows():
        node = int(row.iloc[0])
        node2pred[node] = row.iloc[2].split("\n")
        if _ == 1000:
            break
    return node2label, node2pred


def evaluation(label_file, pred_file, args):
    node2label, node2pred = read_data(label_file, pred_file, args)
    node2digitallabel, count = legality_rate(node2pred, args)

    # for k in [1,3,5]:
    k=1
    acc_count = 0
    node2digitallabel_k = get_topk_predictions(node2digitallabel, k)
    # for node, pred_list in enumerate(node2digitallabel_k):
    #     label = node2label[node]
    #     if len(pred_list) > 0 and label in pred_list:
    #         acc_count += 1
    macro = f1_score(node2digitallabel_k, node2label.tolist()[:len(node2digitallabel_k)], average='macro')
    micro = f1_score(node2digitallabel_k, node2label.tolist()[:len(node2digitallabel_k)], average='micro')
    print('macro: ', macro)
    print('micro: ', micro)
    # print(f"Top@{k} Accuracy: {round(100*acc_count/count,2) if count > 0 else 0.0}%")


def parse_args():
    parser = argparse.ArgumentParser(description="Training Script")
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=4,
                      help="Batch size for training (default: 8)")
    
    # 设备参数
    parser.add_argument("--gpu", type=int, default=3,
                      help="GPU device ID (default: 0)")

    # 实验参数
    parser.add_argument("--output_dir", type=str, default="example_1",
                      help="Output directory for checkpoints")

    parser.add_argument("--num_example", type=int, default=5)

    parser.add_argument("--datasets", type=str,default="imdb")

    parser.add_argument("--typee", type=str,default="movie_only")
    
    return parser.parse_args()

if __name__ == '__main__':
    args=parse_args()
    label_file = f"/home/cwx/workspace/ggfm/data/new_data/{args.datasets}/graph.bin"
    pred_file = f"/home/cwx/workspace/ggfm/data/pred_{args.output_dir}_{args.datasets}.txt"
    translator_generate(args)
    evaluation(label_file, pred_file, args)
