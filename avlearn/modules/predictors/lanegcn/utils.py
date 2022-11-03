# Copyright (c) 2020 Uber Technologies, Inc. All rights reserved.

import numpy as np
from numpy import ndarray
import sys

import torch
from torch import optim, nn, Tensor
from typing import Any, Dict, Optional, List


def gpu(data: Any):
    """
    Transfer tensor in `data` to gpu recursively
    `data` can be dict, list or tuple
    :param data: input dict/list/tuple.
    :returns: the transfered data.
    """
    if isinstance(data, list) or isinstance(data, tuple):
        data = [gpu(x) for x in data]
    elif isinstance(data, dict):
        data = {key: gpu(_data) for key, _data in data.items()}
    elif isinstance(data, torch.Tensor):
        data = data.contiguous().cuda(non_blocking=True)
    return data


def to_long(data: Any):
    """
    Converts a list/dict/Tensor into a list/dict/Tensor of long.
    :param data: input data.
    :returns: the transformed data.
    """
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = to_long(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [to_long(x) for x in data]
    if torch.is_tensor(data) and data.dtype == torch.int16:
        data = data.long()
    return data


def load_pretrain(net, pretrain_dict):
    """
    Loads a pretrained LaneGCN network from a checkpoint.
    :param net: a LaneGCN class object.
    :param pretrain_dict: the checkpoint dictionary.
    """
    state_dict = net.state_dict()
    for key in pretrain_dict.keys():
        if key in state_dict and (
                pretrain_dict[key].size() == state_dict[key].size()):
            value = pretrain_dict[key]
            if not isinstance(value, torch.Tensor):
                value = value.data
            state_dict[key] = value
    net.load_state_dict(state_dict)


class Logger(object):
    "Class for a Logger"

    def __init__(self, log):
        """
        Initializes the logger.
        """
        self.terminal = sys.stdout
        self.log = open(log, "a")

    def write(self, message: str):
        """
        writes a message to the logger.
        :param message: the written message.
        """
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()


class Optimizer(object):
    """
    Optimizer class for training LaneGCN.
    """

    def __init__(self, params: Any, config: Dict, coef: List = None):
        """
        Initializer an optimizer object.
        :param params: list of the LaneGCN network's parameters
        :param config: the configuration dictionary for the network
        :param coef: list of coefficents for each network parameter 
                    (Default: None)
        """
        if not (isinstance(params, list) or isinstance(params, tuple)):
            params = [params]

        if coef is None:
            coef = [1.0] * len(params)
        else:
            if isinstance(coef, list) or isinstance(coef, tuple):
                assert len(coef) == len(params)
            else:
                coef = [coef] * len(params)
        self.coef = coef

        param_groups = []
        for param in params:
            param_groups.append({"params": param, "lr": 0})

        opt = config["opt"]
        assert opt == "sgd" or opt == "adam"
        if opt == "sgd":
            self.opt = optim.SGD(
                param_groups, momentum=config["momentum"],
                weight_decay=config["wd"])
        elif opt == "adam":
            self.opt = optim.Adam(param_groups, weight_decay=0)

        self.lr_func = config["lr_func"]

        if "clip_grads" in config:
            self.clip_grads = config["clip_grads"]
            self.clip_low = config["clip_low"]
            self.clip_high = config["clip_high"]
        else:
            self.clip_grads = False

    def zero_grad(self):
        "zeroes out the gradients"
        self.opt.zero_grad()

    def step(self, epoch):
        "performs one step of the optimizer"
        if self.clip_grads:
            self.clip()

        lr = self.lr_func(epoch)
        for i, param_group in enumerate(self.opt.param_groups):
            param_group["lr"] = lr * self.coef[i]
        self.opt.step()
        return lr

    def clip(self):
        "performs gradient clipping"
        low, high = self.clip_low, self.clip_high
        params = []
        for param_group in self.opt.param_groups:
            params += list(filter(lambda p: p.grad is not None,
                           param_group["params"]))
        for p in params:
            mask = p.grad.data < low
            p.grad.data[mask] = low
            mask = p.grad.data > high
            p.grad.data[mask] = high

    def load_state_dict(self, opt_state):
        "loads a state dictionary of learnable parameters"
        self.opt.load_state_dict(opt_state)


class StepLR:
    """
    Class for implementing a learning rate step decay.
    """

    def __init__(self, lr: List, lr_epochs: List):
        """
        Initializes a step learning rate object.
        :param lr: the different values of learning rates.
        :param lr_epochs: the epochs, for which the learning rate changes, i.e.
                         a learning rate decay step occurs
        """
        assert len(lr) - len(lr_epochs) == 1
        self.lr = lr
        self.lr_epochs = lr_epochs

    def __call__(self, epoch):
        idx = 0
        for lr_epoch in self.lr_epochs:
            if epoch < lr_epoch:
                break
            idx += 1
        return self.lr[idx]


def pred_metrics(preds: List, gt_preds: List, has_preds: List):
    """
    Computes different prediction evaluation metrics for the given predicted 
    trajectories.
    :param preds: list of the predicted trajectories
    :param gt_preds: list of the ground-truth trajectories
    :param has_preds: list of whether a trajectory has ground-truth annotations
    """
    assert has_preds.all()
    preds = np.asarray(preds, np.float32)
    gt_preds = np.asarray(gt_preds, np.float32)

    """batch_size x num_mods x num_preds"""
    # compute error rate
    err = np.sqrt(((preds - np.expand_dims(gt_preds, 1)) ** 2).sum(3))

    ade1 = err[:, 0].mean()
    fde1 = err[:, 0, -1].mean()

    min_idcs = err[:, :, -1].argmin(1)
    row_idcs = np.arange(len(min_idcs)).astype(np.int64)
    err = err[row_idcs, min_idcs]
    ade = err.mean()
    fde = err[:, -1].mean()
    return ade1, fde1, ade, fde, min_idcs


class PostProcess(nn.Module):
    """
    Class for postprocessing the output of a LaneGCN network.
    """

    def __init__(self, config: Dict):
        """
        Initializes a PostProcess object.
        :param config: the configuration dictionary.
        """
        super(PostProcess, self).__init__()
        self.config = config

    def forward(self, out: Dict[str, List[Tensor]], data: Dict):
        """
        Calculates a dictionary containing lists of the predicted and 
        ground-truth trajectories.
        :param out: the predicted trajectories.
        :param data: information about ground truth trajectories.
        :returns: a dictionary containing: (i) a list of the predicted 
        trajectories, (ii) a list of the ground-truth trajectories, (iii) a 
        list of whether a trajectory has ground-truth annotations
        """
        post_out = dict()
        post_out["preds"] = [x[0:1].detach().cpu().numpy() for x in out["reg"]]
        post_out["gt_preds"] = [x[0:1].numpy() for x in data["gt_preds"]]
        post_out["has_preds"] = [x[0:1].numpy() for x in data["has_preds"]]
        return post_out
    
    def append(self, metrics: Dict, loss_out: Dict, post_out: Optional[Dict[str, List[ndarray]]]=None) -> Dict:
        if len(metrics.keys()) == 0:
            for key in loss_out:
                if key != "loss":
                    metrics[key] = 0.0

            for key in post_out:
                metrics[key] = []

        for key in loss_out:
            if key == "loss":
                continue
            if isinstance(loss_out[key], torch.Tensor):
                metrics[key] += loss_out[key].item()
            else:
                metrics[key] += loss_out[key]

        for key in post_out:
            metrics[key] += post_out[key]
        return metrics

    def display(self, metrics: Dict, dt: float, epoch: float, lr: float = None):
        """
        Displays training/val information.
        :param metrics: dictionary containing the different metrics 
                    to be displayed.
        :param dt: the current timestamp.
        :param epoch: the current epoch.
        :param lr: the current learning rate.
        """
        if lr is not None:
            print("Epoch %3.3f, lr %.5f, time %3.2f" % (epoch, lr, dt))
        else:
            print(
                "** Validation, time %3.2f **"
                % dt
            )

        cls = metrics["cls_loss"] / (metrics["num_cls"] + 1e-10)
        reg = metrics["reg_loss"] / (metrics["num_reg"] + 1e-10)
        loss = cls + reg

        preds = np.concatenate(metrics["preds"], 0)
        gt_preds = np.concatenate(metrics["gt_preds"], 0)
        has_preds = np.concatenate(metrics["has_preds"], 0)
        ade1, fde1, ade, fde, min_idcs = pred_metrics(
            preds, gt_preds, has_preds)

        print(
            "loss %2.4f %2.4f %2.4f, ade1 %2.4f, fde1 %2.4f, ade %2.4f, fde %2.4f"
            % (loss, cls, reg, ade1, fde1, ade, fde)
        )
        print()


def collate_fn(batch):
    batch = from_numpy(batch)
    return_batch = dict()
    # Batching by use a list for non-fixed size
    for key in batch[0].keys():
        return_batch[key] = [x[key] for x in batch]
    return return_batch


def from_numpy(data):
    """Recursively transform numpy.ndarray to torch.Tensor.
    """
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = from_numpy(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [from_numpy(x) for x in data]
    if isinstance(data, np.ndarray):
        """Pytorch now has bool type."""
        data = torch.from_numpy(data)
    return data
