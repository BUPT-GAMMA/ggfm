import argparse
import os
from types import SimpleNamespace

import torch
from torchmetrics import AUROC
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import utils
from gp.lightning.data_template import DataModule
from gp.lightning.metric import (
    flat_binary_func,
    EvalKit,
)
from gp.lightning.module_template import ExpConfig
from gp.lightning.training import lightning_fit
from gp.utils.utils import (
    load_yaml,
    combine_dict,
    merge_mod,
    setup_exp,
    set_random_seed,
)
from lightning_model import GraphPredLightning
from models.model import BinGraphModel, BinGraphAttModel
from models.model import PyGRGCNEdge
from task_constructor import LinkPredictionTaskConstructor
from utils import SentenceEncoder

def main(params):
    """
    0. Check GPU setting.
    """
    device, gpu_ids = utils.get_available_devices()
    gpu_size = len(gpu_ids)

    """
    1. Load pretrained model and task configuration
    """
    # Load pretrained model
    pretrained_path = params.pretrained_path
    if pretrained_path:
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        pretrained_state = checkpoint['state_dict']
    
    # Initialize sentence encoder
    encoder = SentenceEncoder(params.llm_name, batch_size=params.llm_b_size)
    
    # Load configurations
    task_config = load_yaml(os.path.join(os.path.dirname(__file__), "configs", "lp_task_config.yaml"))
    data_config = load_yaml(os.path.join(os.path.dirname(__file__), "configs", "data_config.yaml"))

    # Initialize link prediction task
    task = LinkPredictionTaskConstructor(
        params.load_texts,
        encoder,
        task_config,
        data_config,
        batch_size=params.batch_size,
        neg_samples=params.neg_samples,
        split_ratio=params.split_ratio,
    )
    
    # Construct task data
    train_edges, val_edges, test_edges = task.construct_edge_splits()

    # Remove llm model to save memory
    if encoder is not None:
        encoder.flush_model()

    """
    2. Initialize model for link prediction fine-tuning
    """
    out_dim = params.emb_dim + (params.rwpe if params.rwpe is not None else 0)

    gnn = PyGRGCNEdge(
        params.num_layers,
        5,
        out_dim, 
        out_dim,
        drop_ratio=params.dropout,
        JK=params.JK,
    )

    # 使用专门的链路预测模型
    bin_model = BinGraphAttModel if params.JK == "none" else BinGraphModel
    model = bin_model(
        model=gnn,
        llm_name=params.llm_name,
        outdim=out_dim,
        task_dim=1,
        add_rwpe=params.rwpe,
        dropout=params.dropout
    )

    # Load pretrained weights
    if pretrained_path:
        model.load_state_dict(pretrained_state, strict=False)
        
        if params.freeze_pretrained:
            model.freeze_gnn_parameters()

    """
    3. Prepare link prediction datasets
    """
    train_data = task.make_edge_data(train_edges, is_train=True)
    val_data = task.make_edge_data(val_edges, is_train=False)
    test_data = task.make_edge_data(test_edges, is_train=False)
    
    text_dataset = {
        "train": [train_data],
        "val": [val_data],
        "test": [test_data]
    }
    
    params.datamodule = DataModule(
        text_dataset,
        gpu_size=gpu_size,
        num_workers=params.num_workers
    )

    """
    4. Setup link prediction specific evaluation
    """
    eval_data = text_dataset["val"] + text_dataset["test"]
    val_state = ["link_pred_val"]
    test_state = ["link_pred_test"]
    eval_state = val_state + test_state
    
    # 使用AUC作为链路预测的评估指标
    metrics = EvalKit(
        ["auc"] * len(eval_data),
        [AUROC(task="binary") for _ in eval_data],
        torch.nn.BCEWithLogitsLoss(),
        [lambda x: x for _ in eval_data],
        flat_binary_func,
        eval_mode="max",
        exp_prefix="lp",
        eval_state=eval_state,
        val_monitor_state=val_state[0],
        test_monitor_state=test_state[0],
    )

    """
    5. Setup fine-tuning specific optimizer and scheduler
    """
    optimizer = AdamW(
        model.parameters(),
        lr=params.ft_lr,
        weight_decay=params.l2
    )
    
    lr_scheduler = {
        "scheduler": CosineAnnealingLR(
            optimizer,
            T_max=params.num_epochs,
            eta_min=params.ft_lr * 0.01
        ),
        "interval": "epoch",
        "frequency": 1,
    }

    exp_config = ExpConfig(
        "",
        optimizer,
        dataset_callback=train_data.update if hasattr(train_data, 'update') else None,
        lr_scheduler=lr_scheduler,
    )
    exp_config.val_state_name = val_state
    exp_config.test_state_name = test_state

    pred_model = GraphPredLightning(exp_config, model, metrics)

    """
    6. Start fine-tuning
    """
    strategy = "deepspeed_stage_2" if gpu_size > 1 else "auto"
    val_res, test_res = lightning_fit(
        None,
        pred_model,
        params.datamodule,
        metrics,
        params.num_epochs,
        strategy=strategy,
        save_model=True,
        save_path=f"checkpoints/link_pred_finetuned.pt",
        load_best=params.load_best,
        reload_freq=1,
        test_rep=params.test_rep,
        val_interval=params.val_interval
    )

    return val_res, test_res

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Link Prediction Fine-tuning")
    
    # Link prediction specific arguments
    parser.add_argument("--neg_samples", type=int, default=1,
                      help="Number of negative samples per positive edge")
    parser.add_argument("--split_ratio", type=str, default="0.8,0.1,0.1",
                      help="Train,val,test split ratio")
    
    # Fine-tuning arguments
    parser.add_argument("--pretrained_path", type=str, default=None,
                      help="Path to pretrained model checkpoint")
    parser.add_argument("--freeze_pretrained", action="store_true",
                      help="Whether to freeze pretrained parameters")
    parser.add_argument("--ft_lr", type=float, default=1e-4,
                      help="Learning rate for fine-tuning")
    
    # General arguments
    parser.add_argument("--override", type=str)
    parser.add_argument(
        "opts",
        default=[],
        nargs=argparse.REMAINDER,
        help="Modify config options using the command-line",
    )

    params = parser.parse_args()
    
    # Load configurations
    configs = []
    configs.append(
        load_yaml(
            os.path.join(
                os.path.dirname(__file__), "configs", "default_config.yaml"
            )
        )
    )

    if params.override is not None:
        override_config = load_yaml(params.override)
        configs.append(override_config)

    mod_params = combine_dict(*configs)
    mod_params = merge_mod(mod_params, params.opts)
    setup_exp(mod_params)

    params = SimpleNamespace(**mod_params)
    set_random_seed(params.seed)

    torch.set_float32_matmul_precision("high")
    params.exp_name += "_link_pred_finetuned"

    print(params)
    main(params)