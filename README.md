# Project Directory Structure
```
├── ggfm/
│   ├── __init__.py
│   ├── ggfm/
│   │   ├── __init__.py
│   │   ├── conv/
│   │   │   ├── hgt_conv.py
│   │   │   ├── __init__.py
│   │   ├── data/
│   │   │   ├── metapath.py
│   │   │   ├── hgsampling.py
│   │   │   ├── graph.py
│   │   │   ├── utils.py
│   │   │   ├── lm_generate_embs.py
│   │   │   ├── __init__.py
│   │   │   ├── random_walk.py
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── gpt_gnn.py
│   │   │   ├── utils.py
│   │   ├── pretrained_model/
│   │   ├── fine_tuned_model/
│       ├── datasets/
│       │   ├── oag_valid_pairs.pkl
│       │   ├── oag_test_pairs.pkl
│       │   ├── oag_train_pairs.pkl
    ├── examples/
    │   ├── __init__.py
    │   ├── gpt_gnn/
    │   │   ├── nc_ft.py
    │   │   ├── pretrain.py
    │   │   ├── lp_ft.py
    │   ├── lmch/
    │   │   ├── nc_ft.py
    │   │   ├── lp_ft.py
        ├── walklm/
        │   ├── lp_ft.py
        │   ├── nc_ft.py
        │   ├── pretrain.py
```
