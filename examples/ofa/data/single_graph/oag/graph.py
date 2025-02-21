import dill
from collections import defaultdict


class Graph():
    r"""The Heterogeneous Graph Transformer (HGT) operator from the
    `"Heterogeneous Graph Transformer" 
    <https://arxiv.org/abs/2003.01332>`_ paper.
    
    A Graph object describe a homogeneous/heterogeneous graph. 
    The node-related information is included in node_feature: a DataFrame containing all features
    The link-related information is included in edge_list: index the adjacancy matrix (time) by 
        <target_type, source_type, relation_type, target_id, source_id>
    """

    def __init__(self):
        super(Graph, self).__init__()
        '''
            node_forward and bacward are only used when building the data. 
            Afterwards will be transformed into node_feature by DataFrame
            
            node_forward: name -> node_id
            node_bacward: node_id -> feature_dict
            node_feature: a DataFrame containing all features
        '''
        self.node_forward = defaultdict(lambda: {})
        self.node_bacward = defaultdict(lambda: [])
        self.node_feature = defaultdict(lambda: [])

        '''
            edge_list: index the adjacancy matrix (time) by 
            <target_type, source_type, relation_type, target_id, source_id>
        '''
        self.edge_list = defaultdict( #target_type
                            lambda: defaultdict(  #source_type
                                lambda: defaultdict(  #relation_type
                                    lambda: defaultdict(  #target_id
                                        lambda: defaultdict( # source_id
                                            lambda: int # time
                                        )))))
        self.times = {}
        self.node_text = None  # 节点文本
        self.edge_text = None  # 边文本
        self.edge_index = None  # 边的连接关系
        self.x = None  # 节点特征
        self.y = None  # 节点标签
        self.paper_field_edges = None  # 论文-领域边
        self.paper_nodes = None  # 论文节点
        self.field_labels = None  # 领域标签
        self.field_indices = None  # 领域节点索引
        self.paper_field_l2_edges = None  # 论文-二级领域边
        self.num_nodes = 0  # 节点总数

    def add_node(self, node):
        nfl = self.node_forward[node['type']]
        if node['id'] not in nfl:
            self.node_bacward[node['type']] += [node]  # type: [node(dict), node(dict),]
            ser = len(nfl)
            nfl[node['id']] = ser  # id -> cur_length
            return ser
        return nfl[node['id']]
    def add_edge(self, source_node, target_node, time = None, relation_type = None, directed = True):
        edge = [self.add_node(source_node), self.add_node(target_node)]
        '''
            Add bi-directional edges with different relation type
        '''
        self.edge_list[target_node['type']][source_node['type']][relation_type][edge[1]][edge[0]] = time
        if directed:
            self.edge_list[source_node['type']][target_node['type']]['rev_' + relation_type][edge[0]][edge[1]] = time
        else:
            self.edge_list[source_node['type']][target_node['type']][relation_type][edge[0]][edge[1]] = time
        self.times[time] = True
        
    def update_node(self, node):
        nbl = self.node_bacward[node['type']]
        ser = self.add_node(node)
        for k in node:
            if k not in nbl[ser]:
                nbl[ser][k] = node[k]

    def get_meta_graph(self):
        types = self.get_types()
        metas = []
        for target_type in self.edge_list:
            for source_type in self.edge_list[target_type]:
                for r_type in self.edge_list[target_type][source_type]:
                    metas += [(target_type, source_type, r_type)]
        return metas
    
    def get_types(self):
        return list(self.node_feature.keys())


class RenameUnpickler(dill.Unpickler):
    def find_class(self, module, name):
        renamed_module = module
        if module == "data":
            renamed_module = "data.single_graph.oag.graph"
        return super(RenameUnpickler, self).find_class(renamed_module, name)


def renamed_load(file_obj):
    """Load Graph object."""
    return RenameUnpickler(file_obj).load()