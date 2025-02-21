import os
import time
from warnings import filterwarnings

import numpy as np
import torch
from numpy.random import randint
from torch import nn

from examples.gpt_gnn.lp_ft import to_torch, load_gnn
from ggfm.data import args_print, sample_subgraph, ndcg_at_k, mean_reciprocal_rank
from ggfm.data.graph import *
from ggfm.models import *
from ggfm.models.pt_hgnn import GNN

filterwarnings("ignore")

import argparse


def parse_args():
    global parser
    parser = argparse.ArgumentParser(description='Fine-Tuning on OAG Paper-Field (L2) classification task')
    '''
    Dataset arguments
'''
    parser.add_argument('--data_dir', type=str, default='/local/gzy/pthgnn/dataset/new_graph_CS.pk',
                        help='The address of preprocessed graph.')
    parser.add_argument('--use_pretrain', type=str, default=True, help='Whether to use pre-trained model')
    parser.add_argument('--pretrain_model_dir', type=str, default='/home/gzy/py/pthgnn/models/test',
                        help='The address for pretrained model.')
    parser.add_argument('--model_dir', type=str, default='models',
                        help='The address for storing the models and optimization results.')
    parser.add_argument('--task_name', type=str, default='paper_11_classification',
                        help='The name of the stored models and optimization results.')
    parser.add_argument('--cuda', type=int, default=2,
                        help='Avaiable GPU ID')
    parser.add_argument('--domain', type=str, default='_CS',
                        help='CS, Medicion or All: _CS or _Med or (empty)')
    parser.add_argument('--sample_depth', type=int, default=6,
                        help='How many numbers to sample the graph')
    parser.add_argument('--sample_width', type=int, default=128,
                        help='How many nodes to be sampled per layer per type')
    '''
   Model arguments 
'''
    parser.add_argument('--conv_name', type=str, default='hgt',
                        choices=['hgt', 'gcn', 'gat', 'rgcn', 'han', 'hetgnn'],
                        help='The name of GNN filter. By default is Heterogeneous Graph Transformer (hgt)')
    parser.add_argument('--n_hid', type=int, default=400,
                        help='Number of hidden dimension')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='Number of attention head')
    parser.add_argument('--n_layers', type=int, default=3,
                        help='Number of GNN layers')
    parser.add_argument('--prev_norm', help='Whether to add layer-norm on the previous layers', action='store_true')
    parser.add_argument('--last_norm', help='Whether to add layer-norm on the last layers', action='store_true')
    parser.add_argument('--dropout', type=int, default=0.2,
                        help='Dropout ratio')
    '''
    Optimization arguments
'''
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adamw', 'adam', 'sgd', 'adagrad'],
                        help='optimizer to use.')
    parser.add_argument('--scheduler', type=str, default='cycle',
                        help='Name of learning rate scheduler.', choices=['cycle', 'cosine'])
    parser.add_argument('--data_percentage', type=int, default=0.1,
                        help='Percentage of training and validation data to use')
    parser.add_argument('--n_epoch', type=int, default=50,
                        help='Number of epoch to run')
    parser.add_argument('--n_pool', type=int, default=8,
                        help='Number of process to sample subgraph')
    parser.add_argument('--n_batch', type=int, default=16,
                        help='Number of batch (sampled graphs) for each epoch')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Number of output nodes for training')
    parser.add_argument('--clip', type=int, default=0.5,
                        help='Gradient Norm Clipping')
    args = parser.parse_args()
    return args


args = parse_args()

args_print(args)

if args.cuda != -1:
    device = torch.device("cuda:" + str(args.cuda))
else:
    device = torch.device("cpu")

print('Start Loading Graph Data...')
graph = renamed_load(open(args.data_dir, 'rb'))
print('Finish Loading Graph Data!')

target_type = 'paper'

types = graph.get_types()


# train_pairs[target_id] = [[], _time]
def node_classification_sample(seed, pairs, time_range):
    '''
        sub-graph sampling and label preparation for node classification:
        (1) Sample batch_size number of output nodes (papers), get their time.
    '''
    np.random.seed(seed)
    target_ids = np.random.choice(list(pairs.keys()), args.batch_size, replace=False)
    target_info = []
    for target_id in target_ids:
        _, _time = pairs[target_id]
        target_info += [[target_id, _time]]
    '''
        (2) Based on the seed nodes, sample a subgraph with 'sampled_depth' and 'sampled_number'
    '''
    feature, times, edge_list, _, _ = sample_subgraph(graph, time_range, inp={'paper': np.array(target_info)}, \
                                                      sampled_depth=args.sample_depth, sampled_number=args.sample_width)

    '''
        (3) Mask out the edge between the output target nodes (paper) with output source nodes (L2 field)
        其实就是简单的删除掉节点之间的连边
    '''
    masked_edge_list = []
    for i in edge_list['paper']['field']['rev_PF_in_L2']:
        if i[0] >= args.batch_size:  # 因为节点id全部都重新编码了，所以当前节点id如果大于batch_size，那么一定是 非mask_ids
            masked_edge_list += [i]
    edge_list['paper']['field']['rev_PF_in_L2'] = masked_edge_list

    masked_edge_list = []
    for i in edge_list['field']['paper']['PF_in_L2']:
        if i[1] >= args.batch_size:
            masked_edge_list += [i]
    edge_list['field']['paper']['PF_in_L2'] = masked_edge_list
    '''
        (4) Transform the subgraph into torch Tensor (edge_index is in format of pytorch_geometric)
    '''
    node_feature, node_type, edge_time, edge_index, edge_type, node_dict, edge_dict = to_torch(feature, times,
                                                                                               edge_list, graph)
    '''
        (5) Prepare the labels for each output target node (paper), and their index in sampled graph.
            (node_dict[type][0] stores the start index of a specific type of nodes)
    '''
    ylabel = np.zeros([args.batch_size, output_num])
    for x_id, target_id in enumerate(target_ids):
        if target_id not in pairs:
            print('error 1' + str(target_id))
        for source_id in pairs[target_id][0]:
            if source_id not in cand_list:
                print('error 2' + str(target_id))
            ylabel[x_id][field_labels[source_id]] = 1

    ylabel /= ylabel.sum(axis=1).reshape(-1, 1)
    x_ids = np.arange(args.batch_size) + node_dict['paper'][0]
    return node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel


def prepare_data():
    '''
    Sampled and prepare training and validation data without parallelization.
    '''
    train_data = []
    for _ in np.arange(args.n_batch):  # 16个任务
        train_data.append(node_classification_sample(randint(), sel_train_pairs, train_range))
    valid_data = node_classification_sample(randint(), sel_valid_pairs, valid_range)

    return train_data, valid_data


def init():
    global cand_list, field_labels, output_num, criterion, t, train_range, valid_range, test_range, test_pairs, sel_train_pairs, sel_valid_pairs
    '''
    cand_list stores all the L2 fields, which is the classification domain.
'''
    # target_id
    # length = 17750
    cand_list = list(graph.edge_list['field']['paper']['PF_in_L2'].keys())  # field_id
    # print(f"length of cand_list: {len(cand_list)}")
    field_labels = list(graph.node_feature['field']['label'])  # 除L2-field外其余field的类别设置为0（实际不参与计算过程，所以设置成什么都无所谓）
    output_num = max(field_labels) + 1  # 11分类
    print(f"We are training paper {output_num} classification task...")
    '''
    Use KL Divergence here, since each paper can be associated with multiple fields.
    Thus this task is a multi-label classification.
    '''
    criterion = nn.KLDivLoss(reduction='batchmean')
    pre_range = {t: True for t in graph.times if t != None and t < 2014}
    train_range = {t: True for t in graph.times if t != None and t >= 2014 and t <= 2016}
    valid_range = {t: True for t in graph.times if t != None and t > 2016 and t <= 2017}
    test_range = {t: True for t in graph.times if t != None and t > 2017}
    train_pairs = {}
    valid_pairs = {}
    test_pairs = {}
    '''
        Prepare all the souce nodes (L2 field) associated with each target node (paper) as dict
    '''
    for target_id in graph.edge_list['paper']['field']['rev_PF_in_L2']:  # paper_id
        for source_id in graph.edge_list['paper']['field']['rev_PF_in_L2'][target_id]:  # field_id
            _time = graph.edge_list['paper']['field']['rev_PF_in_L2'][target_id][source_id]  # time
            if _time in train_range:
                if target_id not in train_pairs:
                    train_pairs[target_id] = [[], _time]
                train_pairs[target_id][0] += [source_id]
            elif _time in valid_range:
                if target_id not in valid_pairs:
                    valid_pairs[target_id] = [[], _time]
                valid_pairs[target_id][0] += [source_id]
            else:
                if target_id not in test_pairs:
                    test_pairs[target_id] = [[], _time]
                test_pairs[target_id][0] += [source_id]
    # print("Here!!!")
    np.random.seed(43)
    '''
        Only train and valid with a certain percentage of data, if necessary.
    '''
    sel_train_pairs = {p: train_pairs[p] for p in
                       np.random.choice(list(train_pairs.keys()), int(len(train_pairs) * args.data_percentage),
                                        replace=False)}
    sel_valid_pairs = {p: valid_pairs[p] for p in
                       np.random.choice(list(valid_pairs.keys()), int(len(valid_pairs) * args.data_percentage),
                                        replace=False)}
    # print("Done!!!")


def train():
    global gnn, model, node_feature, edge_index
    '''
    Initialize GNN (model is specified by conv_name) and Classifier
'''
    # print("Initializing GNN!!!")
    gnn = GNN(conv_name=args.conv_name, in_dim=len(graph.node_feature[target_type]['emb'].values[0]) + 401,
              n_hid=args.n_hid, \
              n_heads=args.n_heads, n_layers=args.n_layers, dropout=args.dropout, num_types=len(types), \
              num_relations=len(graph.get_meta_graph()) + 1, prev_norm=args.prev_norm, last_norm=args.last_norm)
    if args.use_pretrain:
        # print("use pretrain Here!!!")
        gnn.load_state_dict(load_gnn(torch.load(args.pretrain_model_dir)), strict=False)
        print('Load Pre-trained Model from (%s)' % args.pretrain_model_dir)
    # classifier = Classifier(args.n_hid, len(cand_list))
    classifier = Classifier(args.n_hid, output_num)
    model = nn.Sequential(gnn, classifier).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    stats = []
    res = []
    best_val = 0
    train_step = 0
    st = time.time()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 500, eta_min=1e-6)
    for epoch in np.arange(args.n_epoch) + 1:
        '''
            Prepare Training and Validation Data
        '''

        train_data, valid_data = prepare_data()  # 直接调用，无需并行
        print("train_data and valid_data done!!!")
        et = time.time()
        print('Data Preparation: %.1fs' % (et - st))

        '''
            Train (2014 <= time <= 2016)
        '''
        model.train()
        train_losses = []
        for node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel in train_data:
            node_rep = gnn.forward(node_feature.to(device), node_type.to(device), \
                                   edge_time.to(device), edge_index.to(device), edge_type.to(device))
            res = classifier.forward(node_rep[x_ids])
            loss = criterion(res, torch.FloatTensor(ylabel).to(device))

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            train_losses += [loss.cpu().detach().tolist()]
            train_step += 1
            scheduler.step(train_step)
            del res, loss
        '''
            Valid (2017 <= time <= 2017)
        '''
        model.eval()
        with torch.no_grad():
            node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel = valid_data
            node_rep = gnn.forward(node_feature.to(device), node_type.to(device), \
                                   edge_time.to(device), edge_index.to(device), edge_type.to(device))
            res = classifier.forward(node_rep[x_ids])
            loss = criterion(res, torch.FloatTensor(ylabel).to(device))

            '''
                Calculate Valid NDCG. Update the best model based on highest NDCG score.
            '''
            valid_res = []
            for ai, bi in zip(ylabel, res.argsort(descending=True)):
                valid_res += [ai[bi.cpu().numpy()]]
            valid_ndcg = np.average([ndcg_at_k(resi, len(resi)) for resi in valid_res])

            # 计算分类准确率
            pred_labels = torch.argmax(res, dim=1)  # 取预测结果的最大值所在的索引作为类别标签
            correct = (pred_labels == torch.argmax(torch.tensor(ylabel), dim=1).to(device)).sum().item()  # 计算正确预测的数量
            accuracy = correct / len(ylabel)  # 计算准确率
            # print(f"accuracy: {accuracy}")

            if accuracy > best_val:
                best_val = accuracy
                torch.save(model, os.path.join(args.model_dir, args.task_name + '_' + args.conv_name))
                valid_mrr = mean_reciprocal_rank(valid_res)
                print('Best Test MRR:  %.4f' % np.average(valid_mrr))
                print('UPDATE!!!')

            st = time.time()
            print(
                ("Epoch: %d (%.1fs)  LR: %.5f Train Loss: %.2f  Valid Loss: %.2f  Valid NDCG: %.4f,  Accuracy: %.4f") % \
                (epoch, (st - et), optimizer.param_groups[0]['lr'], np.average(train_losses), \
                 loss.cpu().detach().tolist(), valid_ndcg, accuracy))
            stats += [[np.average(train_losses), loss.cpu().detach().tolist()]]
            del res, loss
        del train_data, valid_data
    '''
        Evaluate the trained model via test set (time >= 2018)
    '''
    best_model = torch.load(os.path.join(args.model_dir, args.task_name + '_' + args.conv_name))
    best_model.eval()
    gnn, classifier = best_model
    with torch.no_grad():
        test_res = []
        for _ in range(10):
            node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel = \
                node_classification_sample(randint(), test_pairs, test_range)
            paper_rep = gnn.forward(node_feature.to(device), node_type.to(device), \
                                    edge_time.to(device), edge_index.to(device), edge_type.to(device))[x_ids]
            res = classifier.forward(paper_rep)
            for ai, bi in zip(ylabel, res.argsort(descending=True)):
                test_res += [ai[bi.cpu().numpy()]]
        test_ndcg = [ndcg_at_k(resi, len(resi)) for resi in test_res]
        print('Best Test NDCG: %.4f' % np.average(test_ndcg))
        test_mrr = mean_reciprocal_rank(test_res)
        print('Best Test MRR:  %.4f' % np.average(test_mrr))

        # 计算分类准确率
        pred_labels = torch.argmax(res, dim=1)  # 取预测结果的最大值所在的索引作为类别标签
        correct = (pred_labels == torch.argmax(torch.tensor(ylabel), dim=1).to(device)).sum().item()  # 计算正确预测的数量
        accuracy = correct / len(ylabel)  # 计算准确率
        print(f"accuracy: {accuracy}")


if __name__ == '__main__':
    init()
    train()
