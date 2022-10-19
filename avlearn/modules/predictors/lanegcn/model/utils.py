# Copyright (c) 2020 Uber Technologies, Inc. All rights reserved.

import numpy as np
from numpy import ndarray
import sys

import torch
from torch import optim, nn
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
        if key in state_dict and (pretrain_dict[key].size() == state_dict[key].size()):
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
