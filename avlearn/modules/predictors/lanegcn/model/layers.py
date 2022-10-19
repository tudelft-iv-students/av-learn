# Copyright (c) 2020 Uber Technologies, Inc. All rights reserved.

from fractions import gcd
import torch
from torch import nn, Tensor
from typing import List


class Conv(nn.Module):

    """
    Class for 2D convolutional layer with normalization (BN or GN) and relu 
    activation function for the LaneGCN network.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1,
                 norm: str = 'GN', ng: int = 32, act: bool = True):
        """
        Loads initial parameters for 2D convolution.
        :param in_channels: input channel dimensions
        :param out_channels: output channel dimensions
        :param kernel_size: kernel size for the convolution (Default: 3)
        :param stride: stride for the convolution (Default: 1)
        :param norm: the normalization method following the 2D convolution 
                    (Default: 'GN')
        :param ng: the number of groups, used in the group normalization 
                    (Default: 32)
        :param act: defines if the relu activation function will be used 
                    (Default: True)
        """
        super(Conv, self).__init__()
        assert(norm in ['GN', 'BN'])

        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              padding=(int(kernel_size) - 1) // 2,
                              stride=stride, bias=False)

        if norm == 'GN':
            self.norm = nn.GroupNorm(gcd(ng, out_channels), out_channels)
        else:
            self.norm = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.act = act

    def forward(self, x):
        """
        Performs a forward pass of the 2D convolutional layer.
        :param x: input tensor of shape [N, C_in, H_in, W_in]
        :return: output tensor of shape [N, C_out, H_out, W_out]
        """
        out = self.conv(x)
        out = self.norm(out)
        if self.act:
            out = self.relu(out)
        return out


class Conv1d(nn.Module):
    """
    Class for 1D convolutional layer with normalization (BN or GN) and relu 
    activation function for the LaneGCN network.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1, norm: str = 'GN',
                 ng: int = 32, act: bool = True):
        """
        Loads initial parameters for 1D convolution.
        :param in_channels: input channel dimensions
        :param out_channels: output channel dimensions
        :param kernel_size: kernel size for the convolution (Default: 3)
        :param stride: stride for the convolution (Default: 1)
        :param norm: the normalization method following the 2D convolution 
                    (Default: 'GN')
        :param ng: the number of groups, used in the group normalization 
                    (Default: 32)
        :param act: defines if the relu activation function will be used 
                    (Default: True)
        """
        super(Conv1d, self).__init__()
        assert(norm in ['GN', 'BN'])

        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              padding=(int(kernel_size) - 1) // 2,
                              stride=stride, bias=False)

        if norm == 'GN':
            self.norm = nn.GroupNorm(gcd(ng, out_channels), out_channels)
        else:
            self.norm = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.act = act

    def forward(self, x):
        """
        Performs a forward pass of the 1D convolutional layer.
        :param x: input tensor of shape [N, C_in, L_in]
        :return: output tensor of shape [N, C_out, L_out]
        """
        out = self.conv(x)
        out = self.norm(out)
        if self.act:
            out = self.relu(out)
        return out


class Linear(nn.Module):
    """
    Class for linear layer with normalization (BN or GN) and relu 
    activation function for the LaneGCN network.
    """

    def __init__(self, in_channels: int, out_channels: int, norm: str = 'GN',
                 ng: int = 32, act: bool = True):
        """
        Loads initial parameters for linear layer.
        :param in_channels: input channel dimensions
        :param out_channels: output channel dimensions
        :param norm: the normalization method following the 2D convolution 
                    (Default: 'GN')
        :param ng: the number of groups, used in the group normalization 
                    (Default: 32)
        :param act: defines if the relu activation function will be used 
                    (Default: True)
        """
        super(Linear, self).__init__()
        assert(norm in ['GN', 'BN'])

        self.linear = nn.Linear(in_channels, out_channels, bias=False)

        if norm == 'GN':
            self.norm = nn.GroupNorm(gcd(ng, out_channels), out_channels)
        else:
            self.norm = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.act = act

    def forward(self, x):
        """
        Performs a forward pass of the linear layer.
        :param x: input tensor of shape [*, in_channels]
        :return: output tensor of shape [*, out_channels]
        """
        out = self.linear(x)
        out = self.norm(out)
        if self.act:
            out = self.relu(out)
        return out


class PostRes(nn.Module):
    """
    Class for post residual layer with normalization (BN or GN) and relu 
    activation function for the LaneGCN network, as defined in Liang et al. 
    (2020).
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 norm: str = 'GN', ng: int = 32, act: bool = True):
        """
        Loads initial parameters for post residual layer.
        :param in_channels: input channel dimensions
        :param out_channels: output channel dimensions
        :param stride: stride for the convolutions (Default: 1)
        :param norm: the normalization method following the 2D convolution 
                    (Default: 'GN')
        :param ng: the number of groups, used in the group normalization 
                    (Default: 32)
        :param act: defines if the relu activation function will be used 
                    (Default: True)
        """
        super(PostRes, self).__init__()
        assert(norm in ['GN', 'BN'])

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if norm == 'GN':
            self.bn1 = nn.GroupNorm(gcd(ng, out_channels), out_channels)
            self.bn2 = nn.GroupNorm(gcd(ng, out_channels), out_channels)
        else:
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or out_channels != in_channels:
            if norm == 'GN':
                self.downsample = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1,
                              stride=stride, bias=False),
                    nn.GroupNorm(gcd(ng, out_channels), out_channels))
            else:
                self.downsample = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1,
                              stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels))
        else:
            self.downsample = None

        self.act = act

    def forward(self, x):
        """
        Performs a forward pass of the post residual layer.
        :param x: input tensor.
        :return: output tensor (addition of normal and downsampled paths).
        """
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x
        if self.act:
            out = self.relu(out)
        return out


class Res1d(nn.Module):
    """
    Class for residual 1D layer with normalization (BN or GN) and relu 
    activation function for the LaneGCN network, as defined in Liang et al. 
    (2020).
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1, norm: str = 'GN',
                 ng: int = 32, act: bool = True):
        """
        Loads initial parameters for post residual layer.
        :param in_channels: input channel dimensions
        :param out_channels: output channel dimensions
        :param kernel_size: kernel size for the convolution (Default: 3)
        :param stride: stride for the convolutions (Default: 1)
        :param norm: the normalization method following the 2D convolution 
                    (Default: 'GN')
        :param ng: the number of groups, used in the group normalization 
                    (Default: 32)
        :param act: defines if the relu activation function will be used 
                    (Default: True)
        """
        super(Res1d, self).__init__()
        assert(norm in ['GN', 'BN'])
        padding = (int(kernel_size) - 1) // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels,
                               kernel_size=kernel_size, stride=stride,
                               padding=padding, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels,
                               kernel_size=kernel_size, padding=padding,
                               bias=False)
        self.relu = nn.ReLU(inplace=True)

        if norm == 'GN':
            self.bn1 = nn.GroupNorm(gcd(ng, out_channels), out_channels)
            self.bn2 = nn.GroupNorm(gcd(ng, out_channels), out_channels)
        else:
            self.bn1 = nn.BatchNorm1d(out_channels)
            self.bn2 = nn.BatchNorm1d(out_channels)

        if stride != 1 or out_channels != in_channels:
            if norm == 'GN':
                self.downsample = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=1,
                              stride=stride, bias=False),
                    nn.GroupNorm(gcd(ng, out_channels), out_channels))
            else:
                self.downsample = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=1,
                              stride=stride, bias=False),
                    nn.BatchNorm1d(out_channels))
        else:
            self.downsample = None

        self.act = act

    def forward(self, x):
        """
        Performs a forward pass of the residual layer.
        :param x: input tensor.
        :return: output tensor (addition of normal and downsampled paths).
        """
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x
        if self.act:
            out = self.relu(out)
        return out


class LinearRes(nn.Module):
    """
    Class for residual linear layer with normalization (BN or GN) and relu 
    activation function for the LaneGCN network, as defined in Liang et al. 
    (2020).
    """

    def __init__(self, in_channels: int, out_channels: int, norm: str = 'GN',
                 ng: int = 32):
        """
        Loads initial parameters for linear layer.
        :param in_channels: input channel dimensions
        :param out_channels: output channel dimensions
        :param norm: the normalization method following the 2D convolution 
                    (Default: 'GN')
        :param ng: the number of groups, used in the group normalization 
                    (Default: 32)
        """
        super(LinearRes, self).__init__()
        assert(norm in ['GN', 'BN'])

        self.linear1 = nn.Linear(in_channels, out_channels, bias=False)
        self.linear2 = nn.Linear(out_channels, out_channels, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if norm == 'GN':
            self.norm1 = nn.GroupNorm(gcd(ng, out_channels), out_channels)
            self.norm2 = nn.GroupNorm(gcd(ng, out_channels), out_channels)
        else:
            self.norm1 = nn.BatchNorm1d(out_channels)
            self.norm2 = nn.BatchNorm1d(out_channels)

        if in_channels != out_channels:
            if norm == 'GN':
                self.transform = nn.Sequential(
                    nn.Linear(in_channels, out_channels, bias=False),
                    nn.GroupNorm(gcd(ng, out_channels), out_channels))
            else:
                self.transform = nn.Sequential(
                    nn.Linear(in_channels, out_channels, bias=False),
                    nn.BatchNorm1d(out_channels))
        else:
            self.transform = None

    def forward(self, x):
        """
        Performs a forward pass of the residual linear layer.
        :param x: input tensor.
        :return: output tensor (addition of normal and downsampled paths).
        """
        out = self.linear1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.norm2(out)

        if self.transform is not None:
            out += self.transform(x)
        else:
            out += x

        out = self.relu(out)
        return out


class Null(nn.Module):
    """
    Class for Null layer.
    """

    def __init__(self):
        super(Null, self).__init__()

    def forward(self, x):
        return x


class Att(nn.Module):
    """
    Class for Attention block to pass context nodes information to target nodes 
    for the LaneGCN network. The context nodes are defined to be the lane nodes 
    whose l2 distance from the reference node i is smaller than a threshold (7m, 
    6m, 100m for A2L, L2A, A2A respectively). This is used in Actor2Map, 
    Actor2Actor, Map2Actor and Map2Map.
    """

    def __init__(self, n_agt: int, n_ctx: int) -> None:
        """
        Loads initial parameters for attention block.
        :param n_agt: number of actor/lane nodes
        :param n_ctx: number of actor/lane nodes 
        """
        super(Att, self).__init__()
        norm = "GN"
        ng = 1

        self.dist = nn.Sequential(
            nn.Linear(2, n_ctx),
            nn.ReLU(inplace=True),
            Linear(n_ctx, n_ctx, norm=norm, ng=ng),
        )

        self.query = Linear(n_agt, n_ctx, norm=norm, ng=ng)

        self.ctx = nn.Sequential(
            Linear(3 * n_ctx, n_agt, norm=norm, ng=ng),
            nn.Linear(n_agt, n_agt, bias=False),
        )

        self.agt = nn.Linear(n_agt, n_agt, bias=False)
        self.norm = nn.GroupNorm(gcd(ng, n_agt), n_agt)
        self.linear = Linear(n_agt, n_agt, norm=norm, ng=ng, act=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,
                agts: Tensor,
                agt_idcs: List[Tensor],
                agt_ctrs: List[Tensor],
                ctx: Tensor,
                ctx_idcs: List[Tensor],
                ctx_ctrs: List[Tensor],
                dist_th: float) -> Tensor:
        """
        Performs a forward pass of the attention block.
        """
        res = agts
        if len(ctx) == 0:
            agts = self.agt(agts)
            agts = self.relu(agts)
            agts = self.linear(agts)
            agts += res
            agts = self.relu(agts)
            return agts

        batch_size = len(agt_idcs)
        hi, wi = [], []
        hi_count, wi_count = 0, 0
        for i in range(batch_size):
            dist = agt_ctrs[i].view(-1, 1, 2) - ctx_ctrs[i].view(1, -1, 2)
            dist = torch.sqrt((dist ** 2).sum(2))
            mask = dist <= dist_th

            idcs = torch.nonzero(mask, as_tuple=False)
            if len(idcs) == 0:
                continue

            hi.append(idcs[:, 0] + hi_count)
            wi.append(idcs[:, 1] + wi_count)
            hi_count += len(agt_idcs[i])
            wi_count += len(ctx_idcs[i])
        hi = torch.cat(hi, 0)
        wi = torch.cat(wi, 0)

        agt_ctrs = torch.cat(agt_ctrs, 0)
        ctx_ctrs = torch.cat(ctx_ctrs, 0)
        dist = agt_ctrs[hi] - ctx_ctrs[wi]
        dist = self.dist(dist)

        query = self.query(agts[hi])

        ctx = ctx[wi]
        ctx = torch.cat((dist, query, ctx), 1)
        ctx = self.ctx(ctx)

        agts = self.agt(agts)
        agts.index_add_(0, hi, ctx)
        agts = self.norm(agts)
        agts = self.relu(agts)

        agts = self.linear(agts)
        agts += res
        agts = self.relu(agts)
        return agts


class AttDest(nn.Module):
    """
    Class for AttDest layer.
    """

    def __init__(self, n_agt: int):
        """
        Loads initial parameters for attention block.
        :param n_agt: number of actor nodes
        """
        super(AttDest, self).__init__()
        norm = "GN"
        ng = 1

        self.dist = nn.Sequential(
            nn.Linear(2, n_agt),
            nn.ReLU(inplace=True),
            Linear(n_agt, n_agt, norm=norm, ng=ng),
        )

        self.agt = Linear(2 * n_agt, n_agt, norm=norm, ng=ng)

    def forward(self, agts: Tensor, agt_ctrs: Tensor, dest_ctrs: Tensor) -> Tensor:
        """
        Performs a forward pass of the attention block.
        """
        n_agt = agts.size(1)
        num_mods = dest_ctrs.size(1)

        dist = (agt_ctrs.unsqueeze(1) - dest_ctrs).view(-1, 2)
        dist = self.dist(dist)
        agts = agts.unsqueeze(1).repeat(1, num_mods, 1).view(-1, n_agt)

        agts = torch.cat((dist, agts), 1)
        agts = self.agt(agts)
        return agts
