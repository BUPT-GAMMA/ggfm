# import dgl
import torch
import random
from .utils import save_pkl_file, open_pkl_file
# from dgl.data.utils import save_graphs, load_graphs


def construct_graph(data_dir, graph, src_dst2edge_type):
    r"""

    Construct dgl.heterograph from ggfm.data.graph.

    Parameters
    ----------
    data_dir: str
        Data directory for saving dgl.heterograph object, which is saved as data_dir/graph.bin.

    graph: class:`ggfm.data.Graph`
        Target graph.
    
    src_dst2edge_type: dict
        The edge types corresponding to (src, dst) types.

    """

    edges = graph.edge_list

    single_edges = {}
    
    for target_type in edges:
            for source_type in edges[target_type]:
                for relation_type in edges[target_type][source_type]:
                    srcs, dsts = [], []
                    single_edges[(source_type, target_type, relation_type)] = [[], []]
                    for target_id in edges[target_type][source_type][relation_type]:
                        for source_id in edges[target_type][source_type][relation_type][target_id]:
                            srcs.append(source_id)
                            dsts.append(target_id)
                    single_edges[(source_type, target_type, relation_type)][0].extend(srcs)
                    single_edges[(source_type, target_type, relation_type)][1].extend(dsts)
    
    merged_edges = {}
    for key, values in single_edges.items():
        src_type, dst_type = key[0], key[1]
        srcs, dsts = values[0], values[1]
        if src_type == "paper" and dst_type == "paper":
            cur_relation = key[2]
            if "rev" in cur_relation:
                cur_edge_type = (src_type, "cited by", dst_type)
                if cur_edge_type not in merged_edges:
                    merged_edges[cur_edge_type] = [[], []]
                merged_edges[cur_edge_type][0].extend(srcs)
                merged_edges[cur_edge_type][1].extend(dsts)
            else:
                cur_edge_type = (src_type, "cites", dst_type)
                if cur_edge_type not in merged_edges:
                    merged_edges[cur_edge_type] = [[], []]
                merged_edges[cur_edge_type][0].extend(srcs)
                merged_edges[cur_edge_type][1].extend(dsts)
        elif src_type == "field" and dst_type == "field":
            cur_relation = key[2]
            if "rev" in cur_relation:  # contains
                cur_edge_type = (src_type, "contains", dst_type)
                if cur_edge_type not in merged_edges:
                    merged_edges[cur_edge_type] = [[], []]
                merged_edges[cur_edge_type][0].extend(srcs)
                merged_edges[cur_edge_type][1].extend(dsts)
            else:
                cur_edge_type = (src_type, "in", dst_type)
                if cur_edge_type not in merged_edges:
                    merged_edges[cur_edge_type] = [[], []]
                merged_edges[cur_edge_type][0].extend(srcs)
                merged_edges[cur_edge_type][1].extend(dsts)
        else:
            cur_edge_type = (src_type, src_dst2edge_type[(src_type, dst_type)], dst_type)
            if cur_edge_type not in merged_edges:
                merged_edges[cur_edge_type] = [[], []]
            merged_edges[cur_edge_type][0].extend(srcs)
            merged_edges[cur_edge_type][1].extend(dsts)
    
    for key, values in merged_edges.items():
        merged_edges[key] = (torch.tensor(values[0]), torch.tensor(values[1]))

    # g = dgl.heterograph(merged_edges)
    # save_graphs(data_dir + "graph.bin", g)
    # print("graph has been saved!")
    return merged_edges


def construct_graph_node_name(data_dir, graph):
    r"""

    Construct graph_node_name.pkl for ggfm.data.graph.

    Parameters
    ----------
    data_dir: str
        Data directory for saving dgl.heterograph object, which is saved as data_dir/graph_node_name.pkl.

    graph: class:`ggfm.data.Graph`
        Target graph.

    """
    graph_node_name = {}
    graph_node_type = graph.get_types()
    for i in range(len(graph_node_type)):
        attr = "name"
        if graph_node_type[i] == "paper": attr = "title"
        graph_node_name[graph_node_type[i]] = graph.node_feature[graph_node_type[i]][attr].tolist()
    save_pkl_file(data_dir + "graph_node_name.pkl", graph_node_name)


def metapath_based_corpus_construction(g, graph_node_name, target_type, metapaths, relation, mid_types, labeled_node_idxs, k=2):

    r"""

    Metapath-based corpus construction.

    Parameters
    ----------
    g: dgl.heterograph
        Target graph.
    
    graph_node_name: dict
        Node names for each node type.
    
    target_type: str
        Sampled node type.
    
    metapaths: list
        Sampled metapaths for each node type.
    
    relation: list
        Relations for each node type.
    
    mid_types: list
        Midtypes of each node type's metapaths.
    
    labeled_node_idxs: list
        Sampled node indexes for the target node type.
    
    k: int, optional
        Number of samples retained for each metapath sampling。
        (default: :obj:`2`)

    """

    # glist, label_dict = load_graphs(data_dir + "graph.bin")
    # g = glist[0]

    # graph_node_name = open_pkl_file(data_dir + 'graph_node_name.pkl')
    metapath = metapaths[target_type]
    relation = relation[target_type]

    sampling_time = 10
    num_nodes = len(labeled_node_idxs)
        
    all_path_for_sampling_times = [[] for _ in range(num_nodes)]
    print("---------------------------------------")
    print(f"Sampling {target_type} type nodes...")

    for p, path in enumerate(metapath):
        path_for_sampling_times = [[] for _ in range(num_nodes)]
        print(f"Sampling the {p}-th path...")
        for st in range(sampling_time):
            traces, types = dgl.sampling.random_walk(g=g, nodes=torch.tensor(labeled_node_idxs), metapath=path)
            traces = traces.tolist()
            length = len(traces)
            print(f"Performing the {st}-th sampling...")
            for i in range(length):
                if i % 10000 == 0:
                    print(f"Sampled {i} nodes...")
                path_i = []
                if traces[i][1] != -1: 
                    path_i.append(target_type.replace('_', ' '))
                    path_i.append(graph_node_name[target_type][traces[i][0]].replace('_', ' '))
                    path_i.append(relation[p][0])
                    mid_type = mid_types[target_type][p][0]
                    path_i.append(mid_type.replace('_', ' '))
                    path_i.append(graph_node_name[mid_type][traces[i][1]].replace('_', ' '))
                    path_i.append(relation[p][1])
                    if len(traces[i]) <= 3:  # apa pa
                        path_i.append(target_type.replace('_', ' '))
                        path_i.append(graph_node_name[target_type][traces[i][2]].replace('_', ' '))
                    else:  # apcpa pcpa
                        mid_type = mid_types[target_type][p][1]
                        path_i.append(mid_type.replace('_', ' '))
                        path_i.append(graph_node_name[mid_type][traces[i][2]].replace('_', ' '))
                        path_i.append(relation[p][2])

                        mid_type = mid_types[target_type][p][2]
                        path_i.append(mid_type.replace('_', ' '))
                        path_i.append(graph_node_name[mid_type][traces[i][3]].replace('_', ' '))
                        path_i.append(relation[p][3])

                        mid_type = target_type
                        path_i.append(mid_type.replace('_', ' '))
                        path_i.append(graph_node_name[mid_type][traces[i][4]].replace('_', ' '))
                else:
                    path_i.append(target_type.replace('_', ' '))
                    path_i.append(graph_node_name[target_type][traces[i][0]].replace('_', ' '))
                
                path_i.append("</s>")
                path_i = " ".join(path_i)
                path_for_sampling_times[i].append(path_i) 
        path_for_sampling_times = [list(set(item)) for item in path_for_sampling_times if item]
        
        path_for_sampling_times = [random.sample(item, min(k, len(item))) for item in path_for_sampling_times if item]
        for i, item in enumerate(path_for_sampling_times):
            path_for_sampling_times[i] = " ".join(item)
            all_path_for_sampling_times[i].append(path_for_sampling_times[i])
    all_path_for_sampling_times = [list(set(item)) for item in all_path_for_sampling_times if item]
    for i, item in enumerate(all_path_for_sampling_times):
        all_path_for_sampling_times[i] = " ".join(item)
        all_path_for_sampling_times[i].rstrip(' </s> ')

    print(all_path_for_sampling_times[0])
    print(f"length: {len(all_path_for_sampling_times)}")
    return all_path_for_sampling_times
