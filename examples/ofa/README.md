# Code for *One for All: Towards Training One Graph Model for All Classification Tasks*

Paper: [https://arxiv.org/abs/2310.00149](https://arxiv.org/abs/2310.00149)

Authors: Hao Liu, Jiarui Feng, Lecheng Kong, Ningyue Liang, Dacheng Tao, Yixin Chen, Muhan Zhang

![OFA Pipeline ](ofapipeline.png)

OFA is a general Graph Classification Framework that can solves a wide range of graph classification tasks with a single
model and a single set of parameters. The tasks are cross-domain (e.g. citation network, molecular graph,...) and
cross-tasks (e.g. few-shot, zero-shot, graph-level, node-leve,...)

OFA use natural languages to describe all graphs, and use a LLM to embed all description in the same embedding space,
which enable cross-domain training using a single model.

OFA propose a prompting paradiagm that all task information are converted to prompt graph. So subsequence model is able
to read tasks information and predict relavent target accordingly, without having to adjust model parameters and
architecture. Hence, a single model can be cross-task.

OFA curated a list of graph datasets from a different sources and domains and describe nodes/edges in the graphs with a
systematical decription protocol. We thank previous works
including, [OGB](https://ogb.stanford.edu/), [GIMLET](https://github.com/zhao-ht/GIMLET/tree/master), [MoleculeNet](https://arxiv.org/abs/1703.00564), [GraphLLM](https://arxiv.org/pdf/2307.03393.pdf),
and [villmow](https://github.com/villmow/datasets_knowledge_embedding/tree/master) for providing wonderful raw
graph/text data that make our work possible.

## 🔥News
### Update 04/14
- Multi-GPU training is employed, it uses all visible GPU to train the model.
- Fixed bug in [#11](https://github.com/LechengKong/OneForAll/issues/11), few-shot should be correct.
- Update ArXiv split. So baseline and OFA both use the same split.
- Fix prompt edge connection to align with the paper.
If you cloned our repo earlier, please first update and reproduce our results.

### Old
OneForAll underwent a major revision, where we cleaned up the code and fixed several reported bugs. The major updates
are:

- Use yaml configs to specify tasks, see [Configuration Section](#configuration-explained) for details.
- Updated graph prompting logic, where users can design their own prompting more freely.
- Use only one Few-shot dataset for few-shot prompting of different levels of tasks.

If you previously used our repository, please pull and delete the old generated feature/text files and regenerate. Sorry
for the inconvenience.

## Requirements

To install requirement for the project using conda:

```
conda env create -f environment.yml
```

## E2E experiments

For joint end-to-end experiments on all collected dataset, run

```
python run_cdm.py --override e2e_all_config.yaml
```

All arguments can be changed by space separated values such as

```
python run_cdm.py --override e2e_all_config.yaml num_layers 7 batch_size 512 dropout 0.15 JK none
```

Users can modify the `task_names` variable in `./e2e_all_config.yaml` to control which datasets are included during
training. The length of `task_names`, `d_multiple`, and `d_min_ratio` should be the same. They can also be specified in
command line arguments by comma separated values.

e.g.

```
python run_cdm.py task_names cora_link,arxiv d_multiple 1,1 d_min_ratio 1,1
```

OFA-ind can be specified by

```
python run_cdm.py task_names cora_link d_multiple 1 d_min_ratio 1
```

## Low resource experiments

To run the few-shot and zero-shot experiments

```
python run_cdm.py --override lr_all_config.yaml
```

## Configuration explained

We define configurations for each task, each task configurations contains several datasets configurations.

Task configurations are stored in `./configs/task_config.yaml`. A task usually consists several splits of datasets (not
necessarily same datasets). For example, a regular end-to-end Cora node classification task will have the train split of
the Cora dataset as the train dataset, the valid split of the Cora dataset as one of the valid dataset, and likewise for
the test split. You can also have more validation/test by specifying the train split of the Cora as one of the
validation/test datasets. Specifically, a task configuration looks like

```yaml
arxiv:
  eval_pool_mode: mean
  dataset: arxiv             # dataset name
  eval_set_constructs:
    - stage: train           # a task should have one and only one train stage dataset
      split_name: train
    - stage: valid
      split_name: valid
      dataset: cora          # replace the default dataset for zero-shot tasks
    - stage: valid
      split_name: valid
    - stage: test
      split_name: test
    - stage: test
      split_name: train      # test the train split
```

Dataset configurations are stored in `./configs/task_config.yaml`. A dataset configuration defines how a dataset is
constructed. Specifically,

```yaml
arxiv:
  task_level: e2e_node
  preprocess: null                       # name of the preprocess function defined in task_constructor.py
  construct: ConstructNodeCls            # name of the dataset construction function defined in task_constructor.py
  args: # additional arguments to construct function
    walk_length: null
    single_prompt_edge: True
  eval_metric: acc                       # evaluation metric
  eval_func: classification_func         # evaluation function that process model output and batch to input to evaluator
  eval_mode: max                         # evaluation mode (min/max)
  dataset_name: arxiv                    # name of the OFAPygDataset
  dataset_splitter: ArxivSplitter        # splitting function defined in task_constructor.py
  process_label_func: process_pth_label  # name of process label function that transform original label to the binary labels
  num_classes: 40 
```

## Add your own datasets

If you are implementing a dataset like Cora/pubmed/Arxiv, we recommend adding a directory of your data \$customized_data \$ under data/single_graph/$customized_data$ and implement gen_data.py under the directory, you can use data/Cora/gen_data.py as an example.


After the data is constructed, you need to register you dataset name in [here](https://github.com/LechengKong/OneForAll/blob/e73f799cabb07e5c6138ba7e8f71881c4e5dd87f/task_constructor.py#L25) , and implement a **splitter** like [here](https://github.com/LechengKong/OneForAll/blob/e73f799cabb07e5c6138ba7e8f71881c4e5dd87f/task_constructor.py#L35). If you are doing zero-shot/few-shot tasks, you can constructor zero-shot/few-shot split here too.

Lastly, register a config entry in configs/data_config.yaml. For example, for end-to-end node classification

```yaml
$data_name$:
  <<: *E2E-node
  dataset_name: $data_name$
  dataset_splitter: $splitter$
  process_label_func: ... # usually processs_pth_label should work
  num_classes: $number of classes$
```
process_label_func converts the target label to binary label, and transform class embedding if the task is zero-shot/few-shot, where the number of class node is not fixed. A list of avalailable process_label_func is [here](https://github.com/LechengKong/OneForAll/blob/e73f799cabb07e5c6138ba7e8f71881c4e5dd87f/task_constructor.py#L280). It takes in all classes embedding and the correct label. The output is a tuple : (label, class_node_embedding, binary/one-hot label).

If you want more flexibility, then adding customized datasets requires implementation of a customized subclass of [OFAPygDataset](https://github.com/LechengKong/OneForAll/blob/e73f799cabb07e5c6138ba7e8f71881c4e5dd87f/data/ofa_data.py#L31) .A template is here:

```python
class CustomizedOFADataset(OFAPygDataset):
    def gen_data(self):
        """
        Returns a tuple of the following format
        (data, text, extra) 
        data: a list of Pyg Data, if you only have a one large graph, you should still wrap it with the list.
        text: a list of list of texts. e.g. [node_text, edge_text, label_text] this is will be converted to pooled vector representation.
        extra: any extra data (e.g. split information) you want to save.
        """

    def add_text_emb(self, data_list, text_emb):
        """
        This function assigns generated embedding to member variables of the graph

        data_list: data list returned in self.gen_data.
        text_emb: list of torch text tensor corresponding to the returned text in self.gen_data. text_emb[0] = llm_encode(text[0])

        
        """
        data_list[0].node_text_feat = ...     # corresponding node features
        data_list[0].edge_text_feat = ...      # corresponding edge features
        data_list[0].class_node_text_feat = ...      # class node features
        data_list[0].prompt_edge_text_feat = ...     # edge features used in prompt node
        data_list[0].noi_node_text_feat = ...       # noi node features, refer to the paper for the definition
        return self.collate(data_list)

    def get_idx_split(self):
        """
        Return the split information required to split the dataset, this optional, you can further split the dataset in task_constructor.py
        
        """

    def get_task_map(self):
        """
        Because a dataset can have multiple different tasks that requires different prompt/class text embedding. This function returns a task map that maps a task name to the desired text embedding. Specifically, a task map is of the following format.

        prompt_text_map = {task_name1: {"noi_node_text_feat": ["noi_node_text_feat", [$Index in data[0].noi_node_text_feat$]],
                                    "class_node_text_feat": ["class_node_text_feat",
                                                             [$Index in data[0].class_node_text_feat$]],
                                    "prompt_edge_text_feat": ["prompt_edge_text_feat", [$Index in data[0].prompt_edge_text_feat$]]},
                       task_name2: similar to task_name 1}
        Please refer to examples in data/ for details.
        """
        return self.side_data[-1]

    def get_edge_list(self, mode="e2e"):
        """
        Defines how to construct prompt graph
        f2n: noi nodes to noi prompt node
        n2f: noi prompt node to noi nodes
        n2c: noi prompt node to class nodes
        c2n: class nodes to noi prompt node
        For different task/mode you might want to use different prompt graph construction, you can do so by returning a dictionary. For example
        {"f2n":[1,0], "n2c":[2,0]} means you only want f2n and n2c edges, f2n edges have edge type 1, and its text embedding feature is data[0].prompt_edge_text_feat[0]
        """
        if mode == "e2e_link":
            return {"f2n": [1, 0], "n2f": [3, 0], "n2c": [2, 0], "c2n": [4, 0]}
        elif mode == "lr_link":
            return {"f2n": [1, 0], "n2f": [3, 0]}
```
