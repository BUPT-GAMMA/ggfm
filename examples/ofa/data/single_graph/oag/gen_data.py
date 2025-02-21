import os
import pickle
import numpy as np
import torch
import torch_geometric as pyg
from torch_geometric.data import Data
from data.single_graph.oag.graph import Graph, renamed_load
from data.ofa_data import OFAPygDataset

__all__ = ['OAGOFADataset', 'OAGSplitter']

def load_oag_data(data_dir="/local/yjy/"):
    """加载 OAG 数据集"""
    with open(os.path.join(data_dir, "graph.pk"), 'rb') as f:
        graph = renamed_load(f)
    
    # 加载划分信息
    splits = {}
    for split in ["train", "valid", "test"]:
        with open(os.path.join(data_dir, f"{split}_ids.pkl"), "rb") as f:
            splits[split] = pickle.load(f)
    
    # 转换为 PyG Data 对象
    paper_features = []
    for node in graph.node_feature['paper']:
        if isinstance(node, dict) and 'emb' in node:
            paper_features.append(node['emb'])
        else:
            paper_features.append(np.zeros(768))
    
    # 收集所有边
    paper_paper_edges = []  # 论文-论文边
    paper_field_edges = []  # 论文-领域边
    field_labels = []      # 领域标签
    
    # 1. 论文-论文引用关系
    if 'paper' in graph.edge_list and 'paper' in graph.edge_list['paper']:
        for relation_type in ['PP_cite']:  # 只使用直接引用关系
            if relation_type in graph.edge_list['paper']['paper']:
                for target_id in graph.edge_list['paper']['paper'][relation_type]:
                    for source_id in graph.edge_list['paper']['paper'][relation_type][target_id]:
                        paper_paper_edges.append([source_id, target_id])
    
    # 2. 论文-领域关系 (用于节点分类)
    if 'paper' in graph.edge_list and 'field' in graph.edge_list['paper']:
        for relation_type in ['rev_PF_in_L0']:  # 使用最顶层领域
            if relation_type in graph.edge_list['paper']['field']:
                for paper_id in graph.edge_list['paper']['field'][relation_type]:
                    for field_id in graph.edge_list['paper']['field'][relation_type][paper_id]:
                        paper_field_edges.append([paper_id, field_id])
                        field_labels.append(field_id)
    
    # 构建 PyG Data 对象
    data = Data(
        x=torch.FloatTensor(paper_features),
        edge_index=torch.LongTensor(paper_paper_edges).t() if paper_paper_edges else torch.empty((2, 0), dtype=torch.long),
        y=torch.LongTensor(field_labels) if field_labels else None,
        node_text=[str(node) for node in graph.node_feature['paper']],
        edge_text=['citation'] * len(paper_paper_edges) if paper_paper_edges else []
    )
    
    # 添加其他属性
    data.paper_field_edges = torch.LongTensor(paper_field_edges) if paper_field_edges else torch.empty((0, 2), dtype=torch.long)
    data.paper_nodes = list(range(len(paper_features)))
    data.field_indices = list(set(field_id for _, field_id in paper_field_edges)) if paper_field_edges else []
    data.num_nodes = len(paper_features)
    
    # 对于链接预测任务，添加 L2 领域信息
    paper_field_l2_edges = []
    if 'paper' in graph.edge_list and 'field' in graph.edge_list['paper']:
        if 'rev_PF_in_L2' in graph.edge_list['paper']['field']:
            for paper_id in graph.edge_list['paper']['field']['rev_PF_in_L2']:
                for field_id in graph.edge_list['paper']['field']['rev_PF_in_L2'][paper_id]:
                    paper_field_l2_edges.append([paper_id, field_id])
    
    data.paper_field_l2_edges = torch.LongTensor(paper_field_l2_edges) if paper_field_l2_edges else torch.empty((0, 2), dtype=torch.long)
    
    return data, splits

def process_oag_graph(graph_data, task="node_classification"):
    """处理 OAG 图数据
    
    Args:
        graph_data: 原始图数据
        task: 任务类型，'node_classification' 或 'link_prediction'
        
    Returns:
        pyg_data: PyG Data对象
    """
    if task == "node_classification":
        # 使用 paper-field 边获取 paper 的标签
        paper_field_edges = graph_data.paper_field_edges
        paper_nodes = graph_data.paper_nodes
        field_labels = graph_data.field_labels
        
        # 将 field 标签传递给相连的 paper
        paper_labels = torch.zeros(len(paper_nodes), dtype=torch.long)
        for paper_idx, field_idx in paper_field_edges:
            paper_labels[paper_idx] = field_labels[field_idx]
            
        # 移除 field 节点和相关边
        mask = torch.ones(graph_data.num_nodes, dtype=torch.bool)
        mask[graph_data.field_indices] = False
        
        # 构建新的图
        edge_index = graph_data.edge_index
        edge_mask = mask[edge_index[0]] & mask[edge_index[1]]
        
        pyg_data = Data(
            x=graph_data.x[mask],
            edge_index=edge_index[:, edge_mask],
            y=paper_labels,
            node_text=graph_data.node_text[mask],
            edge_text=graph_data.edge_text[edge_mask]
        )
        
    elif task == "link_prediction":
        # 将 paper-field-L2 链路预测转换为 17750 类的节点分类
        paper_field_l2_edges = graph_data.paper_field_l2_edges
        paper_nodes = graph_data.paper_nodes
        
        # 使用 L2-field 作为标签
        paper_labels = torch.zeros(len(paper_nodes), dtype=torch.long)
        for paper_idx, field_l2_idx in paper_field_l2_edges:
            paper_labels[paper_idx] = field_l2_idx
            
        # 移除 field 节点和相关边
        mask = torch.ones(graph_data.num_nodes, dtype=torch.bool)
        mask[graph_data.field_indices] = False
        
        edge_index = graph_data.edge_index
        edge_mask = mask[edge_index[0]] & mask[edge_index[1]]
        
        pyg_data = Data(
            x=graph_data.x[mask],
            edge_index=edge_index[:, edge_mask],
            y=paper_labels,
            node_text=graph_data.node_text[mask],
            edge_text=graph_data.edge_text[edge_mask]
        )
    
    return pyg_data

class OAGOFADataset(OFAPygDataset):
    def __init__(self, name, load_texts=True, root=None, encoder=None):
        self.task = "node_classification" if "node" in name else "link_prediction"
        # 先设置 encoder
        self.encoder = encoder
        # 设置数据目录为固定路径
        self.data_dir = "/local/yjy/"
        super().__init__(name, load_texts, self.data_dir, encoder)
        
    def gen_data(self):
        # 加载数据
        graph_data, splits = load_oag_data(self.data_dir)
        
        # 准备文本特征
        if self.task == "node_classification":
            prompt_text = {
                "ML": "Papers in Machine Learning field",
                "DM": "Papers in Data Mining field", 
                "CV": "Papers in Computer Vision field",
                "NLP": "Papers in Natural Language Processing field",
                "DB": "Papers in Database field"
            }
        else:  # link_prediction
            prompt_text = {str(i): f"Papers in field category {i}" for i in range(17750)}
        
        texts = [
            graph_data.node_text,  # 节点文本特征
            graph_data.edge_text,  # 边文本特征
            list(prompt_text.values()),  # 标签文本
            ["prompt edge."],  # 提示边文本
            ["prompt node. paper field classification"]  # 提示节点文本
        ]
        
        # 准备额外数据
        prompt_text_map = {
            "e2e_graph": {
                "noi_node_text_feat": ["noi_node_text_feat", [0]],
                "class_node_text_feat": ["class_node_text_feat", torch.arange(len(texts[2]))],
                "prompt_edge_text_feat": ["prompt_edge_text_feat", [0]]
            },
            "e2e_node": {
                "noi_node_text_feat": ["noi_node_text_feat", [0]],
                "class_node_text_feat": ["class_node_text_feat", torch.arange(len(texts[2]))],
                "prompt_edge_text_feat": ["prompt_edge_text_feat", [0]]
            },
            "e2e_link": {
                "noi_node_text_feat": ["noi_node_text_feat", [0]],
                "class_node_text_feat": ["class_node_text_feat", torch.arange(len(texts[2]))],
                "prompt_edge_text_feat": ["prompt_edge_text_feat", [0]]
            }
        }
        
        return [graph_data], texts, [splits, prompt_text_map]

    def add_raw_texts(self, data_list, texts):
        data, slices = self.collate(data_list)
        data.node_embs = np.array(texts[0])
        data.edge_embs = np.array(texts[1])
        data.class_node_text_feat = np.array(texts[2])
        data.prompt_edge_text_feat = np.array(texts[3])
        data.noi_node_text_feat = np.array(texts[4])
        return data, slices

    def add_text_emb(self, data_list, text_emb):
        data, slices = self.collate(data_list)
        data.node_embs = text_emb[0]
        data.edge_embs = text_emb[1]
        data.class_node_text_feat = text_emb[2]
        data.prompt_edge_text_feat = text_emb[3]
        data.noi_node_text_feat = text_emb[4]
        return data, slices

    def get_idx_split(self):
        return self.side_data[0]

    def get_task_map(self):
        return self.side_data[1]

    def get_edge_list(self, mode="e2e"):
        if mode == "e2e_graph":
            return {"f2n": [1, [0]], "n2f": [3, [0]], "n2c": [2, [0]]}
        elif mode == "lr_graph":
            return {"f2n": [1, [0]], "n2f": [3, [0]]}

import os

import pandas as pd
import torch
import torch_geometric as pyg


def get_logic_label(ordered_txt):
    or_labeled_text = []
    not_and_labeled_text = []
    for i in range(len(ordered_txt)):
        for j in range(len(ordered_txt)):
            c1 = ordered_txt[i]
            c2 = ordered_txt[j]
            txt = "prompt node. literature category and description: not " + c1[0] + ". " + c1[1][0] + " and not " + c2[
                0] + ". " + c2[1][0]
            not_and_labeled_text.append(txt)
            txt = "prompt node. literature category and description: either " + c1[0] + ". " + c1[1][0] + " or " + c2[
                0] + ". " + c2[1][0]
            or_labeled_text.append(txt)
    return or_labeled_text + not_and_labeled_text



def OAGSplitter(dataset):
    """OAG数据集的分割器
    
    Args:
        dataset: OAGOFADataset 实例
        
    Returns:
        dict: 包含训练/验证/测试集索引的字典
    """
    # 获取标签
    labels = dataset.data.y
    num_nodes = len(labels)
    
    # 为每个类别保持平衡采样
    unique_labels = torch.unique(labels)
    train_idx, val_idx, test_idx = [], [], []
    
    for label in unique_labels:
        # 获取当前类别的所有节点索引
        label_idx = (labels == label).nonzero(as_tuple=True)[0]
        num_label_nodes = len(label_idx)
        
        # 随机打乱索引
        perm = torch.randperm(num_label_nodes)
        label_idx = label_idx[perm]
        
        # 80-10-10 分割
        train_size = int(0.8*num_label_nodes)
        val_size = int(0.1*num_label_nodes)
        
        train_idx.append(label_idx[:train_size])
        val_idx.append(label_idx[train_size:train_size + val_size])
        test_idx.append(label_idx[train_size + val_size:])
    
    # 合并所有类别的索引
    train_idx = torch.cat(train_idx)
    val_idx = torch.cat(val_idx)
    test_idx = torch.cat(test_idx)
    
    # 再次随机打乱
    train_idx = train_idx[torch.randperm(len(train_idx))]
    val_idx = val_idx[torch.randperm(len(val_idx))]
    test_idx = test_idx[torch.randperm(len(test_idx))]
    
    split = {
        "train": train_idx,
        "valid": val_idx,
        "test": test_idx
    }
    
    # 保存分割信息到数据集
    if hasattr(dataset.data, 'train_mask'):
        dataset.data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        dataset.data.train_mask[train_idx] = True
    if hasattr(dataset.data, 'val_mask'):
        dataset.data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        dataset.data.val_mask[val_idx] = True
    if hasattr(dataset.data, 'test_mask'):
        dataset.data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        dataset.data.test_mask[test_idx] = True
        
    return split 