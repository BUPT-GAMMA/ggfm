import random
import json
import torch
# import dgl
# from dgl.data.utils import load_graphs, save_graphs
from pathlib import Path
from tqdm import tqdm
import pickle
from torch_geometric.data import HeteroData
from sentence_transformers import SentenceTransformer
from .graph import renamed_load


def dgl_to_pyg(dgl_graph):
    """
    Convert DGL graph to PyG format.
    
    Converts a DGL heterogeneous graph to PyTorch Geometric format while preserving
    node features and edge indices.
    
    Args:
        dgl_graph: DGL heterogeneous graph
        
    Returns:
        HeteroData: Converted PyG heterogeneous graph with:
            - Node features (x) initialized as zero tensors
            - Edge indices preserved for each edge type
            - Node mapping from DGL to PyG format
    """
    pyg_data = HeteroData()
    
    for ntype in dgl_graph.ntypes:
        num_nodes = dgl_graph.num_nodes(ntype)
        pyg_data[ntype].x = torch.zeros((num_nodes, 768))
        pyg_data[ntype].dgl_to_pyg = torch.arange(num_nodes)
    
    for etype in dgl_graph.canonical_etypes:
        src_type, rel_type, dst_type = etype
        src, dst = dgl_graph.edges(etype=etype)
        edge_index = torch.stack([src, dst], dim=0)
        pyg_data[src_type, rel_type, dst_type].edge_index = edge_index
    
    return pyg_data

def sample_subgraph(g, seed_nid, num_hops=2, fanout=10):
    """Sample subgraph based on neighbor sampling"""
    nodes = {ntype: [] for ntype in g.ntypes}
    nodes['paper'].append(seed_nid)
    
    for hop in range(num_hops):
        current_nodes = {ntype: torch.tensor(nodes[ntype], dtype=torch.int64) 
                        for ntype in nodes if len(nodes[ntype]) > 0}
        
        for etype in g.canonical_etypes:
            src_type, rel_type, dst_type = etype
            if dst_type in current_nodes:
                frontier = dgl.sampling.sample_neighbors(
                    g,
                    {dst_type: current_nodes[dst_type]},
                    fanout,
                    edge_dir="in",
                    replace=False
                )
                src_nodes = frontier.edges(etype=etype)[0]
                if src_type not in nodes:
                    nodes[src_type] = []
                nodes[src_type].extend(src_nodes.tolist())
    
    for ntype in nodes:
        nodes[ntype] = list(set(nodes[ntype]))
    
    subg = dgl.node_subgraph(g, nodes)
    
    pyg_graph = dgl_to_pyg(subg)
    
    for ntype in pyg_graph.node_types:
        if not hasattr(pyg_graph[ntype], 'x'):
            num_nodes = pyg_graph[ntype].num_nodes
            pyg_graph[ntype].x = torch.zeros((num_nodes, 768), dtype=torch.float32)

    return pyg_graph

def higpt_prompt_generation():
    """
    Main function to execute all dataset processing steps.
    
    Performs three main steps:
    1. Assigns paper labels based on L2 level field connections
    2. Generates node and edge type embeddings using BERT
    3. Prepares training data by:
        - Sampling subgraphs
        - Creating conversations
        - Saving in required format
    
    The processed data is saved in the following structure:
    - Labeled graph: ggfm/datasets/labeled_field_hg.bin
    - Label mapping: ggfm/datasets/label_to_field.json
    - Type embeddings: ggfm/models/meta_hgt/meta_dict/oag/
    - Training data: ggfm/datasets/stage2_data/OAG-all/
    """
    print("Starting dataset processing...")
    

    print("\n=== Step 1: Assigning paper labels ===")
    print("Loading original dataset...")
    original_graph_path = "ggfm/datasets/new_graph_CS.pk"
    with open(original_graph_path, 'rb') as f:
        original_g = renamed_load(f)
    
    print("Loading processed dataset...")
    processed_graph_path = "ggfm/datasets/reduced_hg.bin"
    processed_glist, _ = load_graphs(processed_graph_path)
    processed_g = processed_glist[0]

    field_features = original_g.node_feature['field']
    paper_field_edges = original_g.edge_list['paper']['field']
    l2_edges = paper_field_edges.get('rev_PF_in_L2', {})
    

    l2_fields = field_features[field_features['attr'] == 'L2']

    num_papers = processed_g.num_nodes('paper')
    paper_labels = torch.full((num_papers,), -1, dtype=torch.long)
    unlabeled_papers = []
    
    for paper_id in range(num_papers):
        if paper_id in l2_edges:
            fields = [f for f in l2_edges[paper_id] if f in l2_fields.index]
            if fields:
                chosen_field = random.choice(fields)
                field_label = l2_fields.loc[chosen_field, 'label']
                paper_labels[paper_id] = field_label
            else:
                unlabeled_papers.append(paper_id)
        else:
            unlabeled_papers.append(paper_id)

    processed_g.nodes['paper'].data['label'] = paper_labels

    labeled_graph_path = "ggfm/datasets/labeled_field_hg.bin"
    save_graphs(labeled_graph_path, [processed_g])
    print(f"Saved labeled graph to: {labeled_graph_path}")
    

    label_to_field = {}
    unique_labels = sorted(l2_fields['label'].unique())
    for label in unique_labels:
        fields_with_label = l2_fields[l2_fields['label'] == label]
        field_infos = []
        for _, field_info in fields_with_label.iterrows():
            field_infos.append({
                'field_id': int(field_info.name),
                'name': field_info['name'],
                'type': field_info['type'],
                'attr': field_info['attr']
            })
        label_to_field[str(label)] = field_infos

    label_to_field['-1'] = [{
        'field_id': -1,
        'name': 'Unlabeled',
        'type': 'special',
        'attr': 'none'
    }]

    mapping_path = "ggfm/datasets/label_to_field.json"
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(label_to_field, f, indent=2, ensure_ascii=False)
    print(f"Saved label mapping to: {mapping_path}")

    print("\n=== Step 2: Generating node and edge type embeddings ===")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print("Loading sentence-bert model...")
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    model = model.to(device)
    

    print("Loading graph data...")
    graph_list, _ = load_graphs(labeled_graph_path)
    g = graph_list[0]

    save_dir = Path("ggfm/models/meta_hgt/meta_dict/oag/")
    save_dir.mkdir(parents=True, exist_ok=True)

    node_type_embeddings = generate_node_type_embeddings(g.ntypes, model, device)
    torch.save(node_type_embeddings, save_dir / "node_type.pt")

    edge_type_embeddings = generate_edge_type_embeddings(g.canonical_etypes, model, device)
    torch.save(edge_type_embeddings, save_dir / "edge_type.pt")

    print("\n=== Step 3: Preparing training data ===")

    base_dir0 = Path("ggfm/datasets/stage2_data/OAG-all")
    base_dir = base_dir0 / "instruct_ds_oag"
    for split in ['train', 'test', 'val']:
        (base_dir / 'ann').mkdir(parents=True, exist_ok=True)
        (base_dir / 'graph_data' / split).mkdir(parents=True, exist_ok=True)

    try:
        with open("ggfm/datasets/graph_node_name.pkl", 'rb') as f:
            node_names = pickle.load(f)
    except:
        node_names = {}

    splits = {}
    for split_name in ['train', 'test', 'val']:
        split_file = f"oag_{split_name}_pairs.pkl"  
        with open(f"ggfm/datasets/{split_file}", 'rb') as f:
            splits[split_name] = pickle.load(f)

    for split_name, paper_ids in splits.items():
        print(f"\nProcessing {split_name} split...")
        json_data = []
        
        for idx, paper_id in enumerate(tqdm(paper_ids)):

            pyg_graph = sample_subgraph(g, paper_id)

            graph_file = f"OAG_{split_name}_{paper_id}.pt"
            graph_path = base_dir / 'graph_data' / split_name / graph_file
            torch.save(pyg_graph, graph_path)

            label = g.nodes['paper'].data['label'][paper_id].item()
            
            entry = {
                "id": f"OAG_{split_name}_{paper_id}",
                "graph_id": f"OAG_{split_name}_{paper_id}",
                "graph": {
                    "node_idx": paper_id,
                    "graph_idx": idx,
                    "graph": f"instruct_ds_oag/graph_data/{split_name}/{graph_file}",
                    "keys_order": ["paper", "author", "affiliation", "venue"],
                    "label": label
                },
                "conversations": create_conversation(pyg_graph, paper_id, node_names, label)
            }
            json_data.append(entry)

        json_file = base_dir / 'ann' / f'OAG_{split_name}_std_0_{len(paper_ids)}.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        print(f"Completed {split_name} split processing, total samples: {len(paper_ids)}")
    
    print("\nDataset processing completed!")

def generate_node_type_embeddings(node_types, model, device):
    """Generate embeddings for node types"""
    print("Generating node type embeddings...")
    embeddings = {}
    for node_type in tqdm(node_types):
        descriptions = generate_node_descriptions(node_type)
        if descriptions:
            embeddings[node_type] = get_embedding(descriptions, model, device)
    return embeddings

def generate_edge_type_embeddings(edge_types, model, device):
    """Generate embeddings for edge types"""
    print("Generating edge type embeddings...")
    embeddings = {}
    for edge_type in tqdm(edge_types):
        descriptions = generate_edge_descriptions(edge_type)
        if descriptions:
            embeddings[edge_type] = get_embedding(descriptions, model, device)
    return embeddings

def generate_node_descriptions(node_type):
    """Generate descriptions for node types"""
    descriptions = {
        'paper': [
            "This node represents a paper",
            "This is a paper node",
            "A paper node in the graph",
            "This node is a paper",
            "The node type is paper"
        ],
        'author': [
            "This node represents an author",
            "This is an author node",
            "An author node in the graph",
            "This node is an author",
            "The node type is author"
        ],
        'affiliation': [
            "This node represents an affiliation",
            "This is an affiliation node",
            "An affiliation node in the graph",
            "This node is an affiliation",
            "The node type is affiliation"
        ],
        'venue': [
            "This node represents a venue",
            "This is a venue node",
            "A venue node in the graph",
            "This node is a venue",
            "The node type is venue"
        ]
    }
    return descriptions.get(node_type, [])

def generate_edge_descriptions(edge_type):
    """
    Generate natural language descriptions for edge types.
    
    Provides multiple paraphrased descriptions for each edge type in the
    heterogeneous academic graph.
    
    Args:
        edge_type (tuple): Edge type as (source_type, relation, target_type)
        
    Returns:
        list: List of string descriptions for the edge type. Returns empty list
            if edge type is not recognized.
    """
    descriptions = {
        ('paper', 'cites', 'paper'): [
            "This paper cites that paper",
            "A paper cites another paper",
            "Paper to paper citation",
            "This is a citation relationship",
            "A citation between papers"
        ],
        ('paper', 'cited by', 'paper'): [
            "This paper is cited by that paper",
            "A paper is cited by another paper",
            "Paper to paper citation",
            "This is a citation relationship",
            "A citation between papers"
        ],
        ('author', 'writes', 'paper'): [
            "Author writes paper",
            "This is a writing relationship",
            "Author to paper writing",
            "Writing relationship",
            "Author writes"
        ],
        ('paper', 'is writen by', 'author'): [
            "Paper is written by author",
            "Paper to author relationship",
            "This is a writing relationship",
            "Paper has author as writer",
            "Author writes this paper"
        ],
        ('author', 'is affiliated with', 'affiliation'): [
            "Author is affiliated with affiliation",
            "Author to affiliation relationship",
            "This is an affiliation relationship",
            "Author belongs to affiliation",
            "Affiliation has this author"
        ],
        ('affiliation', 'employs', 'author'): [
            "Affiliation employs author",
            "Affiliation to author relationship",
            "This is an employment relationship",
            "Affiliation has this author",
            "Author belongs to affiliation"
        ],
        ('paper', 'published in', 'venue'): [
            "Paper is published in venue",
            "Paper to venue relationship",
            "This is a publication relationship",
            "Paper appears in venue",
            "Venue contains this paper"
        ],
        ('venue', 'publishes', 'paper'): [
            "Venue publishes paper",
            "Venue to paper relationship",
            "This is a publication relationship",
            "Venue contains paper",
            "Paper appears in venue"
        ]
    }
    return descriptions.get(edge_type, [])

def get_embedding(descriptions, model, device):
    """Get embedding vector for descriptions"""
    embeddings = model.encode(descriptions, convert_to_tensor=True)
    embeddings = embeddings.to(device)
    return embeddings.mean(dim=0).cpu()

def create_conversation(g, paper_id, node_names, label):
    """
    Create conversation content for a paper node.
    
    Generates a structured conversation with:
    1. A prompt describing the graph structure
    2. Paper-specific information (title, authors)
    3. A question about paper category
    4. The ground truth label
    
    Args:
        g (HeteroData): PyG heterogeneous graph
        paper_id (int): ID of target paper node
        node_names (dict): Mapping of node IDs to actual names
        label (int): Ground truth label for paper category
        
    Returns:
        list: List of conversation turns as dictionaries with:
            - "from": Speaker identifier ("human" or "gpt")
            - "value": Content of the turn
    """

    paper_title = node_names['paper'][paper_id] if 'paper' in node_names else f"Paper_{paper_id}"

    authors = []
    if ('paper', 'is writen by', 'author') in g.edge_types:
        edge_index = g['paper', 'is writen by', 'author'].edge_index
        paper_idx = 0  
        author_mask = edge_index[0] == paper_idx
        author_ids = edge_index[1][author_mask].tolist()
        authors.extend([node_names['author'][aid] if 'author' in node_names else f"Author_{aid}" 
                      for aid in author_ids])

    prompt = (
        "Given a heterogeneous academic network graph about computer science, there are four types of nodes, "
        "namely: paper, author, affiliation, venue. The relationships (meta paths) between different nodes include: "
        "[('affiliation', 'employs', 'author'), ('author', 'is affiliated with', 'affiliation'), "
        "('author', 'writes', 'paper'), ('paper', 'cited by', 'paper'), ('paper', 'cites', 'paper'), "
        "('paper', 'is writen by', 'author'), ('paper', 'published in', 'venue'), ('venue', 'publishes', 'paper')]. "
        "By performing random sampling of 2-hop 10 neighbors centered on the target paper node, a heterogeneous "
        "subgraph is obtained. In the subgraph, \"paper\" nodes: <graph>, where the 0-th node is the central node "
        f"that represents a paper with the following information: \nTitle: {paper_title} \n"
        f"Published author lists: {authors} \n"
        "\"author\" nodes: <graph>; \"affiliation\" nodes: <graph>; \"venue\" nodes: <graph>. \n"
        "Question: Which of the following areas does this paper belong to: one of the 11 categories "
        "represented by integer labels ranging from 0 to 10? \n"
        "Give likely categories directly. "
    )
    
    return [
        {"from": "human", "value": prompt},
        {"from": "gpt", "value": str(label)}
    ]

if __name__ == "__main__":
    higpt_prompt_generation() 
