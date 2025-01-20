import os

import torch
import argparse
import numpy as np
from ggfm.data.arxiv_text_pair_datasets import ArxivTextPairDataset
from texttable import Texttable
from collections import defaultdict
from ggfm.models import HGT, RNNModel, Matcher, GPT_GNN, get_optimizer
from ggfm.data import renamed_load, sample_subgraph, ndcg_at_k, feature_extractor
from warnings import filterwarnings
from ggfm.models.translator_model.translator_qformer_arxiv import TranslatorQformerArxiv
from ggfm.models.translator_model.translator_chatglm_arxiv import TranslatorCHATGLMArxiv
import time
import logging
import datetime
import os
from torch.utils.data import DataLoader
import math

filterwarnings("ignore")


# lr_scheduler

class LinearWarmupStepLRScheduler:
    def __init__(
        self,
        optimizer,
        max_epoch,
        min_lr,
        init_lr,
        decay_rate=1,
        warmup_start_lr=-1,
        warmup_steps=0,
        **kwargs
    ):
        self.optimizer = optimizer

        self.max_epoch = max_epoch
        self.min_lr = min_lr

        self.decay_rate = decay_rate

        self.init_lr = init_lr
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr if warmup_start_lr >= 0 else init_lr

    def step(self, cur_epoch, cur_step):
        if cur_epoch == 0:
            warmup_lr_schedule(
                step=cur_step,
                optimizer=self.optimizer,
                max_step=self.warmup_steps,
                init_lr=self.warmup_start_lr,
                max_lr=self.init_lr,
            )
        else:
            step_lr_schedule(
                epoch=cur_epoch,
                optimizer=self.optimizer,
                init_lr=self.init_lr,
                min_lr=self.min_lr,
                decay_rate=self.decay_rate,
            )


class LinearWarmupCosineLRScheduler:
    def __init__(
        self,
        optimizer,
        max_epoch,
        min_lr,
        init_lr,
        warmup_steps=0,
        warmup_start_lr=-1,
        **kwargs
    ):
        self.optimizer = optimizer

        self.max_epoch = max_epoch
        self.min_lr = min_lr

        self.init_lr = init_lr
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr if warmup_start_lr >= 0 else init_lr

    def step(self, cur_epoch, cur_step):
        # assuming the warmup iters less than one epoch
        if cur_epoch == 0:
            warmup_lr_schedule(
                step=cur_step,
                optimizer=self.optimizer,
                max_step=self.warmup_steps,
                init_lr=self.warmup_start_lr,
                max_lr=self.init_lr,
            )
        else:
            cosine_lr_schedule(
                epoch=cur_epoch,
                optimizer=self.optimizer,
                max_epoch=self.max_epoch,
                init_lr=self.init_lr,
                min_lr=self.min_lr,
            )


def cosine_lr_schedule(optimizer, epoch, max_epoch, init_lr, min_lr):
    """Decay the learning rate"""
    lr = (init_lr - min_lr) * 0.5 * (
        1.0 + math.cos(math.pi * epoch / max_epoch)
    ) + min_lr
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def warmup_lr_schedule(optimizer, step, max_step, init_lr, max_lr):
    """Warmup the learning rate"""
    lr = min(max_lr, init_lr + (max_lr - init_lr) * step / max(max_step, 1))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def step_lr_schedule(optimizer, epoch, init_lr, min_lr, decay_rate):
    """Decay the learning rate"""
    lr = max(min_lr, init_lr * (decay_rate**epoch))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def _load_checkpoint(self, filename):
        """
        Resume from a checkpoint.
        """

        if os.path.isfile(filename):
            checkpoint = torch.load(filename, map_location=self.device)
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]
        self.unwrap_dist_model(self.model).load_state_dict(state_dict)

        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scaler and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])

        self.start_epoch = checkpoint["epoch"] + 1
        logging.info("Resume checkpoint from {}".format(filename))


def evaluation(self, model, data_loader, cuda_enabled=True):
    header = "Evaluation"
    print_freq = 10

    results = []

    for samples in data_loader:
        samples = prepare_sample(samples, cuda_enabled=cuda_enabled)

        eval_output = model(samples)
        results.extend(eval_output)


    return results
    

@torch.no_grad()
def eval_epoch(data_loader, split_name, cur_epoch, skip_reload=False):
    """
    Evaluate the model on a given split.

    Args:
        split_name (str): name of the split to evaluate on.
        cur_epoch (int): current epoch.
        skip_reload_best (bool): whether to skip reloading the best checkpoint.
            During training, we will reload the best checkpoint for validation.
            During testing, we will use provided weights and skip reloading the best checkpoint .
    """
    assert data_loader, "data_loader for split {} is None.".format(split_name)

    if not skip_reload and cur_epoch == "best":
        model = _reload_best_model(model)
    model.eval()

    model.before_evaluation(dataset=datasets[split_name], task_type=type())
    results = evaluation(model, data_loader)

    if results is not None:
        return after_evaluation(
            val_result=results,
            split_name=split_name,
            epoch=cur_epoch,
        )


def evaluate(test_splits, cur_epoch="best", skip_reload=False):
    test_logs = dict()

    if len(test_splits) > 0:
        for split_name in test_splits:
            test_logs[split_name] = eval_epoch(
                split_name=split_name, cur_epoch=cur_epoch, skip_reload=skip_reload
            )

        return test_logs


def _train_inner_loop(
        epoch,
        iters_per_epoch,
        model,
        data_loader,
        optimizer,
        scaler,
        lr_scheduler,
        start_iters=None,
        accum_grad_iters=1,
    ):
        """


        An inner training loop compatible with both epoch-based and iter-based training.

        When using epoch-based, training stops after one epoch; when using iter-based,
        training stops after #iters_per_epoch iterations.
        """
        use_amp = scaler is not None

        if not hasattr(data_loader, "__next__"):
            # convert to iterator if not already
            data_loader = iter(data_loader)

        # if iter-based runner, schedule lr based on inner epoch.
        logging.info(
            "Start training epoch {}, {} iters per inner epoch.".format(
                epoch, iters_per_epoch
            )
        )
        if start_iters is None:
            # epoch-based runner
            inner_epoch = epoch
        else:
            # In iter-based runner, we schedule the learning rate based on iterations.
            inner_epoch = start_iters // iters_per_epoch

        current_step = 0
        for network_input in data_loader:
            lr_scheduler.step(cur_epoch=inner_epoch, cur_step=current_step) # TODO：换一个torch自带的lr_scheduler

            with torch.cuda.amp.autocast():
                loss = model(network_input)["loss"]

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # update gradients every accum_grad_iters iterations
            if (current_step + 1) % accum_grad_iters == 0:
                optimizer.step()
                optimizer.zero_grad()

            current_step += 1


        return loss.item()


def train_epoch(model, epoch, iters_per_epoch, data_loader, optimizer, scaler, lr_scheduler, accum_grad_iters):
    _train_inner_loop(
        epoch=epoch,
        iters_per_epoch=iters_per_epoch,
        model=model,
        data_loader=data_loader,
        optimizer=optimizer,
        scaler=scaler,
        lr_scheduler=lr_scheduler,
        accum_grad_iters=accum_grad_iters,
    )


def _save_checkpoint(model, optimizer, scaler, output_dir, cur_epoch, is_best=False):
    """
    Save the checkpoint at the current epoch.
    """
    model_no_ddp = model
    param_grad_dic = {
        k: v.requires_grad for (k, v) in model_no_ddp.named_parameters()
    }
    state_dict = model_no_ddp.state_dict()
    for k in list(state_dict.keys()):
        if k in param_grad_dic.keys() and not param_grad_dic[k]:
            # delete parameters that do not require gradient
            del state_dict[k]
    save_obj = {
        "model": state_dict,
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler else None,
        "epoch": cur_epoch,
    }
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_to = os.path.join(
        output_dir,
        "checkpoint_{}.pth".format("best" if is_best else cur_epoch),
    )
    logging.info("Saving checkpoint at epoch {} to {}.".format(cur_epoch, save_to))
    torch.save(save_obj, save_to)


def train_stage_1(args):
    task = args.task
    lr_sched = args.lr_sched
    datasets_dir = args.datasets_dir
    init_lr = args.init_lr
    min_lr = args.min_lr
    warmup_lr = args.warmup_lr
    weight_decay = args.weight_decay
    max_epoch = args.max_epoch
    batch_size_train = args.batch_size_train
    batch_size_eval = args.batch_size_eval
    warmup_steps = args.warmup_steps
    accum_grad_iters = args.accum_grad_iters
    log_freq = args.log_freq
    max_length = args.max_length
    vocab_size = args.vocab_size
    output_dir = 'output'
    mode = 'train'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # create dataset
    train_set = ArxivTextPairDataset(args.datasets_dir, max_length, vocab_size, mode)

    # create datalaoder
    dataloader = DataLoader(train_set,batch_size=batch_size_train,pin_memory=True,sampler=None,drop_last=True)

    # create model
    model = TranslatorQformerArxiv(num_features=768,
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
        max_txt_len=32,)
    model = model.to(device)
    
    # optimizer
    p_wd, p_non_wd = [], []
    num_parameters = 0 

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue  # frozen weights
        if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
            p_non_wd.append(p)
        else:
            p_wd.append(p)
    num_parameters += p.data.nelement()
    optim_params = [
            {
                "params": p_wd,
                "weight_decay": float(weight_decay),
            },
            {"params": p_non_wd, "weight_decay": 0},
        ]
    optimizer = torch.optim.AdamW(
        optim_params,
        lr = float(init_lr),
        weight_decay=float(weight_decay),
        betas=(0.9, 0.999),
    )

    # scaler
    scaler = torch.cuda.amp.GradScaler()

    # lr_scheduler
    if lr_sched == "linear_warmup_cosine_lr":
        lr_scheduler = LinearWarmupCosineLRScheduler(
            optimizer=optimizer,
            max_epoch=max_epoch,
            min_lr=min_lr,
            init_lr=init_lr,
            warmup_steps=warmup_steps,
            warmup_start_lr=warmup_lr,
        )
    elif lr_sched == "linear_warmup_step_lr":
        lr_scheduler = LinearWarmupStepLRScheduler(
            optimizer=optimizer,
            max_epoch=max_epoch,
            min_lr=min_lr,
            init_lr=init_lr,
            decay_rate=0.1,
            warmup_steps=warmup_steps,
            warmup_start_lr=warmup_lr,
        )
    else:
        raise ValueError("Unknown lr scheduler {}".format(lr_sched))

    # train_model
    start_time = time.time()
    best_agg_metric = 0
    best_epoch = 0
    mode = 'train'  # Mode for dataset, replace with actual mode if needed
    evaluate_only = False
    resume_ckpt_path = None
    start_epoch = 0
    valid_splits = []  # List of validation splits, replace with actual splits if needed
    test_splits = []
    iters_per_epoch = train_set.row_count // batch_size_train


    # resume from checkpoint if specified
    if not evaluate_only and resume_ckpt_path is not None:
        _load_checkpoint(resume_ckpt_path)

    for cur_epoch in range(start_epoch, max_epoch):
        # training phase
        if not evaluate_only:
            logging.info("Start training")
            train_loss = train_epoch(model, cur_epoch, iters_per_epoch, dataloader, optimizer, scaler, lr_scheduler, accum_grad_iters)

        # evaluation phase
        if len(valid_splits) > 0:
            for split_name in valid_splits:
                logging.info("Evaluating on {}.".format(split_name))

                val_log = eval_epoch(
                    split_name=split_name, cur_epoch=cur_epoch
                )
                if val_log is not None:

                    agg_metrics = val_log["agg_metrics"]
                    if agg_metrics > best_agg_metric and split_name == "val":
                        best_epoch, best_agg_metric = cur_epoch, agg_metrics

                        _save_checkpoint(model, optimizer, scaler, output_dir, cur_epoch, is_best=True)

                    val_log.update({"best_epoch": best_epoch})

        else:
            # if no validation split is provided, we just save the checkpoint at the end of each epoch.
            if not evaluate_only:
                _save_checkpoint(model, optimizer, scaler, output_dir, cur_epoch, is_best=False)# TODO:改为保存模型

        if evaluate_only:
            break

    # testing phase
    test_epoch = "best" if len(valid_splits) > 0 else cur_epoch
    evaluate(test_splits, cur_epoch=test_epoch, skip_reload=evaluate_only)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info("Training time {}".format(total_time_str))


def train_stage_2(args):
    task = args.task
    lr_sched = args.lr_sched
    datasets_dir = args.datasets_dir
    init_lr = args.init_lr
    min_lr = args.min_lr
    warmup_lr = args.warmup_lr
    weight_decay = args.weight_decay
    max_epoch = args.max_epoch
    batch_size_train = args.batch_size_train
    batch_size_eval = args.batch_size_eval
    warmup_steps = args.warmup_steps
    accum_grad_iters = args.accum_grad_iters
    log_freq = args.log_freq
    max_length = args.max_length
    vocab_size = args.vocab_size
    output_dir = 'output'
    mode = 'train'
    bert_dir = '/home/ubuntu/workspace/graphtranslator/ggfm/ggfm/models/bert-base-uncased'
    llm_dir = '/home/ubuntu/workspace/graphtranslator/GraphTranslator/Translator/models/chatglm2-6b'
    pretrained = '../model_output/pretrain_arxiv_stage1/checkpoint_0.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    # create dataset
    train_set = ArxivTextPairDataset(datasets_dir, max_length, vocab_size, mode)

    # create datalaoder
    dataloader = DataLoader(train_set,batch_size=batch_size_train,pin_memory=True,sampler=None,drop_last=True)

    # create model
    model = TranslatorCHATGLMArxiv(
        llm_dir=llm_dir,
        bert_dir=bert_dir,
        num_features=768,
        num_query_token=32,
        chatglm2_model="",
        max_txt_len=2048,)
    model = model.to(device)
    
    # optimizer
    p_wd, p_non_wd = [], []
    num_parameters = 0 

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue  # frozen weights
        if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
            p_non_wd.append(p)
        else:
            p_wd.append(p)
    num_parameters += p.data.nelement()
    optim_params = [
            {
                "params": p_wd,
                "weight_decay": float(weight_decay),
            },
            {"params": p_non_wd, "weight_decay": 0},
        ]
    optimizer = torch.optim.AdamW(
        optim_params,
        lr = float(init_lr),
        weight_decay=float(weight_decay),
        betas=(0.9, 0.999),
    )

    # scaler
    scaler = torch.cuda.amp.GradScaler()

    # lr_scheduler
    if lr_sched == "linear_warmup_cosine_lr":
        lr_scheduler = LinearWarmupCosineLRScheduler(
            optimizer=optimizer,
            max_epoch=max_epoch,
            min_lr=min_lr,
            init_lr=init_lr,
            warmup_steps=warmup_steps,
            warmup_start_lr=warmup_lr,
        )
    elif lr_sched == "linear_warmup_step_lr":
        lr_scheduler = LinearWarmupStepLRScheduler(
            optimizer=optimizer,
            max_epoch=max_epoch,
            min_lr=min_lr,
            init_lr=init_lr,
            decay_rate=0.1,
            warmup_steps=warmup_steps,
            warmup_start_lr=warmup_lr,
        )
    else:
        raise ValueError("Unknown lr scheduler {}".format(lr_sched))

    # train_model
    start_time = time.time()
    best_agg_metric = 0
    best_epoch = 0
    mode = 'train'  # Mode for dataset, replace with actual mode if needed
    evaluate_only = False
    resume_ckpt_path = None
    start_epoch = 0
    valid_splits = []  # List of validation splits, replace with actual splits if needed
    test_splits = []
    iters_per_epoch = train_set.row_count // batch_size_train


    # resume from checkpoint if specified
    if not evaluate_only and resume_ckpt_path is not None:
        _load_checkpoint(resume_ckpt_path)

    for cur_epoch in range(start_epoch, max_epoch):
        # training phase
        if not evaluate_only:
            logging.info("Start training")
            train_loss = train_epoch(model, cur_epoch, iters_per_epoch, dataloader, optimizer, scaler, lr_scheduler, accum_grad_iters)

        # evaluation phase
        if len(valid_splits) > 0:
            for split_name in valid_splits:
                logging.info("Evaluating on {}.".format(split_name))

                val_log = eval_epoch(
                    split_name=split_name, cur_epoch=cur_epoch
                )
                if val_log is not None:

                    agg_metrics = val_log["agg_metrics"]
                    if agg_metrics > best_agg_metric and split_name == "val":
                        best_epoch, best_agg_metric = cur_epoch, agg_metrics

                        _save_checkpoint(model, optimizer, scaler, output_dir, cur_epoch, is_best=True)

                    val_log.update({"best_epoch": best_epoch})

        else:
            # if no validation split is provided, we just save the checkpoint at the end of each epoch.
            if not evaluate_only:
                _save_checkpoint(model, optimizer, scaler, output_dir, cur_epoch, is_best=False)# TODO:改为保存模型

        if evaluate_only:
            break

    # testing phase
    test_epoch = "best" if len(valid_splits) > 0 else cur_epoch
    evaluate(test_splits, cur_epoch=test_epoch, skip_reload=evaluate_only)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info("Training time {}".format(total_time_str))


if __name__ == '__main__':
    argparse.ArgumentParser(description="Training")
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--task", type=str, default="arxiv_text_pretrain")
    parser.add_argument("--lr_sched", type=str, default="linear_warmup_cosine_lr")
    parser.add_argument("--datasets_dir", type=str, default="/home/ubuntu/workspace/graphtranslator/GraphTranslator/data/arxiv/summary_embeddings.csv")
    parser.add_argument("--init_lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--warmup_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--max_epoch", type=int, default=1)
    parser.add_argument("--batch_size_train", type=int, default=1)
    parser.add_argument("--batch_size_eval", type=int, default=4)
    parser.add_argument("--warmup_steps", type=int, default=5000)
    parser.add_argument("--accum_grad_iters", type=int, default=32)
    parser.add_argument("--log_freq", type=int, default=10)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--vocab_size", type=int, default=100000)
    args = parser.parse_args()
    train_stage_1(args)
    train_stage_2(args)
    