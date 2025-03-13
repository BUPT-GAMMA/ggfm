def get_data(dataset):
    """获取 Cora 数据集
    
    Args:
        dataset: SingleGraphOFADataset 实例
        
    Returns:
        tuple: (data_list, texts, task_map)
    """
    # 加载图数据
    data = load_cora_data()
    
    # 准备文本数据
    texts = [
        data.node_text,  # 节点文本特征
        data.edge_text,  # 边文本特征
        ["prompt node. node classification on citation network"],  # NOI节点文本
        [f"Papers in category {i}" for i in range(7)],  # 类别节点文本
        ["prompt edge."]  # 提示边文本
    ]
    
    # 准备任务映射字典
    task_map = {
        "e2e_node": {
            "noi_node_text_feat": ["noi_node_text_feat", [0]],
            "class_node_text_feat": ["class_node_text_feat", torch.arange(7)],  # 7个类别
            "prompt_edge_text_feat": ["prompt_edge_text_feat", [0]]
        },
        "e2e_link": {
            "noi_node_text_feat": ["noi_node_text_feat", [0]],
            "class_node_text_feat": ["class_node_text_feat", torch.arange(2)],  # 链接预测是二分类
            "prompt_edge_text_feat": ["prompt_edge_text_feat", [0]]
        }
    }
    
    return [data], texts, task_map