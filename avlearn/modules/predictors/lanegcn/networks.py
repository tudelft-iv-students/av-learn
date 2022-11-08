# Copyright (c) 2020 Uber Technologies, Inc. All rights reserved.

import json
import os
import sys
from fractions import gcd
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
from mmcv import Config
import torch
from layers import Att, AttDest, Conv1d, Linear, LinearRes, Res1d
from loss import Loss
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import (Optimizer, PostProcess, collate_fn, cpu, gpu, load_pretrain,
                   to_long)

from avlearn.apis.evaluate import Evaluator
from avlearn.datasets.prediction.nuscenes_ds import InferenceNSDataset
from avlearn.datasets.prediction.nuscenes_ds import NuScenesDataset as Dataset
from avlearn.modules.__base__ import BasePredictor


class LaneGCN(BasePredictor):
    """
    LaneGCN wrapper class for av-learn.
    """

    def __init__(self,
                 config: str,
                 resume: bool = False,
                 checkpoint_pth: str = None) -> None:
        """
        Loads the initial LaneGCN parameters, needed for training or inference.
        :param config: Path to a LaneGCN configuration file

        :param resume: whether to resume from a pre-trained checkpoint.
        :param checkpoint_pth: the path to a ckeckpoint.
        """
        self.config = Config.fromfile(config)
        self.net = Net(self.config)
        self.net = self.net.cuda()

        self.loss = Loss(self.config).cuda()
        self.post_process = PostProcess(self.config).cuda()

        self.params = self.net.parameters()
        self.opt = Optimizer(self.params, self.config)
        self.seed = config["seed"]
        torch.cuda.manual_seed(self.seed)

        if checkpoint_pth is not None:
            if not os.path.isabs(checkpoint_pth):
                checkpoint_pth = Path(
                    config["save_dir"]) / Path(checkpoint_pth)
            ckpt = torch.load(
                checkpoint_pth, map_location=lambda storage, loc: storage)
            load_pretrain(self.net, ckpt["state_dict"])
            if resume:
                self.config["epoch"] = ckpt["epoch"]
                self.opt.load_state_dict(ckpt["opt_state"])

    def forward(
            self,
            dataroot: Union[str, Path],
            map_dataroot: Union[str, Path],
            tracking_results: dict,
            **kwargs) -> List[Dict]:
        """
        Given a dictionary of tracking results, performs inference on the 
        LaneGCN network.

        :param dataroot: Path to data.
        :param map_dataroot: the path to the root folder of the map data.
        :param tracking_results: a dictionary containing the tracking results 
                        in the nuScenes format.

        :returns: a list containing for each instance-sample token a dictionary 
        of its predicted trajectories:
        [{
            "instance" : the instance token id.
            "sample" : the sample token id.
            "prediction" : a [K x num_pred x 2] array with the K different 
                           predicted trajectories, num_pred frames into the 
                           future.
            "probablities" : a list of K probabilities, corresponding to the 
                             confidence score for each predicted 
                             trajectory.
        }
        {
            ...
        }
        ...
        ]
        """
        # load inference dataset
        inferrence_dataset = InferenceNSDataset(
            dataroot, self.config, tracking_results=tracking_results,
            maps_dir=map_dataroot, split="val")
        self.inferrence_loader = DataLoader(
            inferrence_dataset,
            batch_size=self.config["batch_size"],
            num_workers=self.config["workers"],
            collate_fn=collate_fn,
            pin_memory=True,
            shuffle=True,
        )

        results = []
        for data in tqdm(self.inferrence_loader):
            with torch.no_grad():
                output = self.net(data)
                for j, out in enumerate(output['reg']):
                    result = dict()
                    result['instance'], result["sample"] = data["ins_sam"][j]
                    result['prediction'] = out[0][:, :12, :]
                    result['probabilities'] = output['cls'][j][0]
                    results.append(result)

        return results

    def evaluate(self,
                 dataroot: Union[str, Path],
                 map_dataroot: Union[str, Path],
                 work_dir: Union[str, Path] = None,
                 batch_size: int = 8,
                 split: str = "val",
                 version: str = "v1.0-trainval",
                 **kwargs) -> None:
        """
        Performs inference on the validation set of nuScenes for the prediction 
        task on the LaneGCN network and saves the results in a json file, 
        containing a list containing for each instance-sample token a 
        dictionary of its predicted trajectories:
        [{
            "instance" : the instance token id.
            "sample" : the sample token id.
            "prediction" : a [K x num_pred x 2] array with the K different 
                           predicted trajectories, num_pred frames into the 
                           future.
            "probablities" : a list of K probabilities, corresponding to the 
                             confidence score for each predicted 
                             trajectory.
        }
        {
            ...
        }
        ...
        ]
        The inference results are then evaluated using the official nuScenes 
        prediction evaluation, with the evaluation results also being saved.
        :param dataroot: Path to data.
        :param map_dataroot: the path to the root folder of the map data.
        :param work_dir: Directory to save output.
        :param batch_size: Number of samples per batch.
        :param split: the nuScenes split (Default: "val")
        :param version: the nuScenes version (Default: "trainval")
        """
        # Create save directory
        self.save_dir = Path(self.config.get("save_dir", work_dir))
        if not self.save_dir.exists():
            self.save_dir.mkdir(parents=True, exist_ok=True)

        # Data loader for evaluation
        self.net.eval()
        val_dataset = Dataset(dataroot, self.config,
                              maps_dir=map_dataroot,
                              split=split, train=False)
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.get("val_batch_size", batch_size),
            num_workers=self.config["val_workers"],
            collate_fn=collate_fn,
            pin_memory=True,
        )

        results = []
        for data in tqdm(self.val_loader):
            with torch.no_grad():
                output = self.net(data)
                for j, out in enumerate(output['reg']):
                    result = dict()
                    result['instance'], result["sample"] = data["ins_sam"][j]
                    result['prediction'] = out[0][:, :12, :]
                    result['probabilities'] = output['cls'][j][0]
                    results.append(result)

        # convert to json serializable
        infer_results = cpu(results)
        save_path = Path(self.save_dir)
        infer_save_path = save_path / Path("inference_results")
        eval_save_path = save_path / Path("evaluation_results")
        # make save directories for both inference and evaluation results
        infer_save_path.mkdir(parents=True, exist_ok=True)
        eval_save_path.mkdir(parents=True, exist_ok=True)

        # save inference results
        with open(infer_save_path / Path("inference_result.json"), "w") as f:
            json.dump(infer_results, f)

        evaluator = Evaluator(
            task="prediction",
            dataset="nuscenes",
            results=infer_save_path / Path("inference_result.json"),
            output=eval_save_path,
            dataroot=dataroot,
            split=split,
            version=version,
            config_path=None,
            verbose=True,
            render_classes=None,
            render_curves=False,
            plot_examples=0)

        evaluator.evaluate()

    def train(
        self,
        dataroot: Union[str, Path],
        map_dataroot: Union[str, Path],
        work_dir: Union[str, Path] = None,
        epochs: int = 36,
        batch_size: int = 8,
        **kwargs
    ) -> None:
        """
        Performs the training of the LaneGCN network.

        :param dataroot: Path to data.
        :param map_dataroot: the path to the root folder of the map data.
        :param work_dir: Directory to save output.
        :param epochs: Number of epochs to train the network.
        :param batch_size: Number of samples per batch.
        """
        # Data loader for training
        train_dataset = Dataset(dataroot, self.config,
                                maps_dir=map_dataroot,
                                split="train", train=True)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.get("batch_size", batch_size),
            num_workers=self.config["workers"],
            collate_fn=collate_fn,
            pin_memory=True,
            shuffle=True,
        )

        # Create save directory
        self.save_dir = Path(self.config.get("save_dir", work_dir))
        if not self.save_dir.exists():
            self.save_dir.mkdir(parents=True, exist_ok=True)

        epoch = self.config.get("epoch", 0)
        # find remaining epochs
        self.remaining_epochs = int(
            np.ceil(self.config.get("num_epochs", epochs) - epoch))
        for i in tqdm(range(self.remaining_epochs)):
            self.__train_epoch(epoch + i)

    def __train_epoch(self, epoch: int) -> None:
        """
        Performs the training of a single epoch of the LaneGCN network.
        :param epoch: the current epoch to be trained.
        """
        self.net.train()

        num_batches = len(self.train_loader)
        epoch_per_batch = 1.0 / num_batches

        metrics = dict()
        for data in tqdm(self.train_loader):
            epoch += epoch_per_batch
            data = dict(data)

            output = self.net(data)
            loss_out = self.loss(output, data)
            post_out = self.post_process(output, data)
            self.post_process.append(metrics, loss_out, post_out)

            self.opt.zero_grad()
            loss_out["loss"].backward()

            if epoch >= self.config["num_epochs"]:
                break
        # save checkpoint for each epoch
        self.__save_ckpt(epoch)
        return

    def __save_ckpt(self, epoch: int) -> None:
        """
        Saves a checkpoint for the LaneGCN network on a specific epoch.
        :param epoch: the current train epoch of the model.
        """
        if not self.save_dir.exists():
            self.save_dir.mkdir(parents=True, exist_ok=True)

        state_dict = self.net.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()

        save_name = "%3.3f.ckpt" % epoch
        torch.save(
            {"epoch": epoch, "state_dict": state_dict,
                "opt_state": self.opt.opt.state_dict()},
            Path(self.save_dir) / Path(save_name),
        )


class Net(nn.Module):
    """
    Class for the LaneGCN network as defined in n Liang et al. (2020). 
    Contains following components:
        1. ActorNet: a 1D CNN to process the trajectory input
        2. MapNet: LaneGraphCNN to learn structured map representations 
           from vectorized map data
        3. Actor-Map Fusion Cycle: fuse the information between actor nodes 
           and lane nodes:
            a. A2M: introduces real-time traffic information to 
                lane nodes, such as blockage or usage of the lanes
            b. M2M:  updates lane node features by propagating the 
                traffic information over lane graphs
            c. M2A: fuses updated map features with real-time traffic 
                information back to actors
            d. A2A: handles the interaction between actors and produces
                the output actor features
        4. PredNet: prediction head for motion forecasting using 
           feature from A2A
    """

    def __init__(self, config: str) -> None:
        """
        Initializes the LaneGCN network given a configuration file.
        :param config: the path to the network's configuration file
        """
        super(Net, self).__init__()
        self.config = config

        self.actor_net = ActorNet(n_actor=config["n_actor"])
        self.map_net = MapNet(n_map=config["n_map"],
                              num_scales=config["num_scales"])

        self.a2m = A2M(n_map=config["n_map"],
                       n_actor=config["n_actor"], config=config)
        self.m2m = M2M(n_map=config["n_map"], num_scales=config["num_scales"])
        self.m2a = M2A(n_map=config["n_map"],
                       n_actor=config["n_actor"], config=config)
        self.a2a = A2A(n_actor=config["n_actor"], config=config)

        self.pred_net = PredNet(num_mods=config["num_mods"],
                                n_actor=config["n_actor"],
                                num_preds=config["num_preds"], config=config)

    def forward(self, data: Dict) -> Dict[str, List[Tensor]]:
        """
        Performs a forward pass of the LaneGCN network.
        :param data: dictionary containing the input data.
        :returns: :returns: a dictionary containing (i) a regression head with 
        K possible future trajectories for each actor [M x K x num_pred x 2], 
        transformed to world coordinates and (ii) their respective confidence 
        scores [M x K]
        """
        actors, actor_idcs = actor_gather(gpu(data["feats"]))
        actor_ctrs = gpu(data["ctrs"])
        # extract actor features
        actors = self.actor_net(actors)

        graph = graph_gather(to_long(gpu(data["graph"])))
        # extract map features
        nodes, node_idcs, node_ctrs = self.map_net(graph)

        # actor-map fusion cycle
        nodes = self.a2m(nodes, graph, actors, actor_idcs, actor_ctrs)
        nodes = self.m2m(nodes, graph)
        actors = self.m2a(actors, actor_idcs, actor_ctrs,
                          nodes, node_idcs, node_ctrs)
        actors = self.a2a(actors, actor_idcs, actor_ctrs)

        # prediction
        out = self.pred_net(actors, actor_idcs, actor_ctrs)
        rot, orig = gpu(data["rot"]), gpu(data["orig"])
        # transform prediction to world coordinates
        for i in range(len(out["reg"])):
            out["reg"][i] = torch.matmul(out["reg"][i], rot[i]) + orig[i].view(
                1, 1, 1, -1
            )
        return out


def actor_gather(actors: List[Tensor]) -> Tuple[Tensor, List[Tensor]]:
    """
    Concatenates a list of actor tensors and creates indices to differentiate 
    between the different timesteps, that each actor appears in.
    :param actors: list of the actor tensors, containing the 2D posititions of 
                the actors in the BeV plane for different timesteps
    :returns: a tuple containing (i) the concatenated tensor of all actor 
    tensors and (ii) a list of tensors, containing the indices to distinguish 
    between different timesteps for each actor.
    """
    batch_size = len(actors)
    num_actors = [len(x) for x in actors]

    actors = [x.transpose(1, 2) for x in actors]
    actors = torch.cat(actors, 0)

    actor_idcs = []
    count = 0
    for i in range(batch_size):
        idcs = torch.arange(count, count + num_actors[i]).to(actors.device)
        actor_idcs.append(idcs)
        count += num_actors[i]
    return actors, actor_idcs


def graph_gather(graphs: List[Dict]) -> Dict:
    """
    Constructs a lane graph, given the map data as input. For any two lanes 
    which are directly reachable, 4 types of connections are defined: 
    predecessor, successor. 
    :param graph: list of input lanes and their connectivity.
    :returns: a dictionary corresponding to the graph in the following format:

    graph:{
        "idcs":[
            0-th: 0-th seq [0, 1, 2, ..., num of node of 0-th seq]
        ]
        "ctrs":[
            0-th: 0-th seq ctrs, ndarray, (num of node, 2)
        ]
        "feats": torch.tensor, (all node in batch, 2)
        "pre":[
            0-th:{ # 0-th means 0-th dilated
                "u": torch.tensor, (all batch node num, )
                "v": torch.tensor, (all batch node num, ) # v is the pre of u
            }
        ]
        "suc": [
            0-th:{ # 0-th means 0-th dilated
                "u": torch.tensor, (all batch node num, )
                "v": torch.tensor, (all batch node num, ) # v is the suc of u
            }
        ]
    }
    """
    # number of lanes in the map
    batch_size = len(graphs)
    node_idcs = []
    count = 0
    counts = []
    for i in range(batch_size):
        counts.append(count)
        idcs = torch.arange(count, count + graphs[i]["num_nodes"]).to(
            graphs[i]["feats"].device
        )
        # mode_idcs: indices to distinguish between different nodes in one lane
        node_idcs.append(idcs)
        count = count + graphs[i]["num_nodes"]

    graph = dict()
    graph["idcs"] = node_idcs
    graph["ctrs"] = [x["ctrs"] for x in graphs]

    for key in ["feats"]:
        graph[key] = torch.cat([x[key] for x in graphs], 0)

    for k1 in ["pre", "suc"]:
        graph[k1] = []
        for i in range(len(graphs[0]["pre"])):
            graph[k1].append(dict())
            for k2 in ["u", "v"]:
                graph[k1][i][k2] = torch.cat(
                    [graphs[j][k1][i][k2] + counts[j]
                        for j in range(batch_size)], 0
                )

    return graph


class ActorNet(nn.Module):
    """
    Class for the actor feature extractor with Conv1D.
    """

    def __init__(self, n_actor: int):
        """
        Intilializes the ActorNet network. The network has 3 groups/scales of 
        1D convolutions. Each group consists of 2 residual blocks, with the 
        stride of the first block as 2. A Feature Pyramid Network is used to 
        fuse the multi-scale features, and apply another residual block to 
        obtain the output tensor. 
        :param n_actor: the number of actors, corresponds to the output channel 
                dimensions.
        """
        super(ActorNet, self).__init__()
        self.n_actor = n_actor
        # group normalization
        norm = "GN"
        ng = 1

        # number of input chanells
        n_in = 3
        n_out = [32, 64, 128]
        blocks = [Res1d, Res1d, Res1d]
        num_blocks = [2, 2, 2]

        groups = []
        for i in range(len(num_blocks)):
            group = []
            if i == 0:
                group.append(blocks[i](n_in, n_out[i], norm=norm, ng=ng))
            else:
                group.append(blocks[i](n_in, n_out[i],
                             stride=2, norm=norm, ng=ng))

            for j in range(1, num_blocks[i]):
                group.append(blocks[i](n_out[i], n_out[i], norm=norm, ng=ng))
            groups.append(nn.Sequential(*group))
            n_in = n_out[i]
        self.groups = nn.ModuleList(groups)

        lateral = []
        for i in range(len(n_out)):
            lateral.append(
                Conv1d(n_out[i], self.n_actor, norm=norm, ng=ng, act=False))
        self.lateral = nn.ModuleList(lateral)

        self.output = Res1d(self.n_actor, self.n_actor, norm=norm, ng=ng)

    def forward(self, actors: Tensor) -> Tensor:
        """
        Performs a forward pass of the ActorNet.
        :param actors: concatenated actor tensor of observed past trajectories 
                    for all actors in the scene of shape [M, 3, T], where T 
                    the trajectory size.
        :return: temporal actor feature map: [M, n_actor].
        """
        out = actors

        outputs = []
        for i in range(len(self.groups)):
            out = self.groups[i](out)
            outputs.append(out)

        out = self.lateral[-1](outputs[-1])
        for i in range(len(outputs) - 2, -1, -1):
            out = F.interpolate(out, scale_factor=2,
                                mode="linear", align_corners=False)
            out += self.lateral[i](outputs[i])

        out = self.output(out)[:, :, -1]
        return out


class MapNet(nn.Module):
    """
    Class for the Map Graph feature extractor with LaneGraphCNN.
    """

    def __init__(self, num_scales: int, n_map: int):
        """
        Intilializes the MapNet network.  The network is a stack of 4 
        multi-scale LaneConv residual blocks, each of which consists of a 
        LaneConv(1,2,4,8,16,32) and a linear layer with a residual connection. 
        All layers have n_map feature channels.
        :param num_scales: the number of steps to look for preceding or 
                succeeding nodes.
        :param n_map: the number of lane nodes in the map.
        """
        super(MapNet, self).__init__()
        # group normalization
        norm = "GN"
        ng = 1

        self.input = nn.Sequential(
            nn.Linear(2, n_map),
            nn.ReLU(inplace=True),
            Linear(n_map, n_map, norm=norm, ng=ng, act=False),
        )
        self.seg = nn.Sequential(
            nn.Linear(2, n_map),
            nn.ReLU(inplace=True),
            Linear(n_map, n_map, norm=norm, ng=ng, act=False),
        )

        keys = ["ctr", "norm", "ctr2"]
        for i in range(num_scales):
            keys.append("pre" + str(i))
            keys.append("suc" + str(i))

        fuse = dict()
        for key in keys:
            fuse[key] = []

        for i in range(4):
            for key in fuse:
                if key in ["norm"]:
                    fuse[key].append(nn.GroupNorm(gcd(ng, n_map), n_map))
                elif key in ["ctr2"]:
                    fuse[key].append(
                        Linear(n_map, n_map, norm=norm, ng=ng, act=False))
                else:
                    fuse[key].append(nn.Linear(n_map, n_map, bias=False))

        for key in fuse:
            fuse[key] = nn.ModuleList(fuse[key])
        self.fuse = nn.ModuleDict(fuse)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, graph: dict) -> Tuple[Tensor, List[Tensor], Tensor]:
        """
        Performs a forward pass of the MapNet.
        :param graph: the constructed lane graph.
        :returns: a tuple containing: (i) the concatenated lane nodes Tensor 
        [N x n_map], (ii) a list of lane node indices, (iii) a list of lane node 
        ctrs.
        """
        if (
            len(graph["feats"]) == 0
            or len(graph["pre"][-1]["u"]) == 0
            or len(graph["suc"][-1]["u"]) == 0
        ):
            temp = graph["feats"]
            return (
                temp.new().resize_(0),
                [temp.new().long().resize_(0) for x in graph["node_idcs"]],
                temp.new().resize_(0),
            )

        ctrs = torch.cat(graph["ctrs"], 0)
        feat = self.input(ctrs)
        feat += self.seg(graph["feats"])
        feat = self.relu(feat)

        """fuse map"""
        res = feat
        for i in range(len(self.fuse["ctr"])):
            temp = self.fuse["ctr"][i](feat)
            for key in self.fuse:
                if key.startswith("pre") or key.startswith("suc"):
                    k1 = key[:3]
                    k2 = int(key[3:])
                    temp.index_add_(
                        0,
                        graph[k1][k2]["u"],
                        self.fuse[key][i](feat[graph[k1][k2]["v"]]),
                    )

            feat = self.fuse["norm"][i](temp)
            feat = self.relu(feat)

            feat = self.fuse["ctr2"][i](feat)
            feat += res
            feat = self.relu(feat)
            res = feat
        return feat, graph["idcs"], graph["ctrs"]


class A2M(nn.Module):
    """
    Class for the Actor to Map Fusion: fuses real-time traffic information from
    actor nodes to lane nodes.
    """

    def __init__(self, n_map: int, n_actor: int, config: str):
        """
        Intilializes the A2M network. A2M introduces real-time traffic 
        information to lane nodes, such as blockage or usage of the lanes. Given 
        an actor node, the features from its context lane nodes are aggregated 
        using a spatial attention layer.
        :param n_map: the number of lane nodes in the map.
        :param n_actor: the number of actor nodes.
        :param config: the path to the network's configuration file
        """
        super(A2M, self).__init__()
        self.n_map = n_map
        self.n_actor = n_actor
        self.config = config
        # group normalization
        norm = "GN"
        ng = 1

        # fuse meta, static, dyn
        self.meta = Linear(self.n_map + 4, self.n_map, norm=norm, ng=ng)
        att = []
        for i in range(2):
            # spatial attention module
            att.append(Att(self.n_map, self.n_actor))
        self.att = nn.ModuleList(att)

    # self.a2m(nodes, graph, actors, actor_idcs, actor_ctrs)
    def forward(
            self, feat: Tensor,
            graph:
            Dict
            [str,
             Union
             [List[Tensor],
              Tensor, List[Dict[str, Tensor]],
              Dict[str, Tensor]]],
            actors: Tensor, actor_idcs: List[Tensor],
            actor_ctrs: List[Tensor]) -> Tensor:
        """
        Performs a forward pass of the A2M network.
        :param feat: the lane nodes feature map [N x n_map] (output of MapNet)
        :param graph: the constructed lane graph.
        :param actors: the actor nodes feature map [M x n_actor] (output of 
                        ActorNet).
        :param actor_idcs: list of actor indices.
        :param actor_ctrs: list of actor ctrs.
        :returns: the updated lane nodes features [N x n_map] (input of M2M).
        """
        for i in range(len(self.att)):
            feat = self.att[i](
                feat,
                graph["idcs"],
                graph["ctrs"],
                actors,
                actor_idcs,
                actor_ctrs,
                self.config["actor2map_dist"],
            )
        return feat


class M2M(nn.Module):
    """
    Class for the lane to lane block: propagates information over lane
            graphs and updates the features of lane nodes.
    """

    def __init__(self, n_map: int, num_scales: int):
        """
        Intilializes the M2M network. M2M updates lane node features by 
        propagating the traffic information over the lane graph. Implemented 
        using another LaneGCN, which has the same architecture as the one used 
        in MapNet.
        :param n_map: the number of lane nodes in the map.
        :param num_scales: the number of steps to look for preceding or 
                succeeding nodes.
        """
        super(M2M, self).__init__()
        self.n_map = n_map
        self.num_scales = num_scales
        norm = "GN"
        ng = 1

        keys = ["ctr", "norm", "ctr2"]
        for i in range(self.num_scales):
            keys.append("pre" + str(i))
            keys.append("suc" + str(i))

        fuse = dict()
        for key in keys:
            fuse[key] = []

        for i in range(4):
            for key in fuse:
                if key in ["norm"]:
                    fuse[key].append(nn.GroupNorm(
                        gcd(ng, self.n_map), self.n_map))
                elif key in ["ctr2"]:
                    fuse[key].append(
                        Linear(self.n_map, self.n_map, norm=norm, ng=ng,
                               act=False))
                else:
                    fuse[key].append(
                        nn.Linear(self.n_map, self.n_map, bias=False))

        for key in fuse:
            fuse[key] = nn.ModuleList(fuse[key])
        self.fuse = nn.ModuleDict(fuse)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feat: Tensor, graph: Dict) -> Tensor:
        """
        Performs a forward pass of the M2M network.
        :param feat: the lane nodes feature map [N x n_map] (output of A2M).
        :param graph: the constructed lane graph.
        :returns: the updated lane nodes features [N x n_map] (input of M2A).
        """
        # fuse-map
        res = feat
        for i in range(len(self.fuse["ctr"])):
            temp = self.fuse["ctr"][i](feat)
            for key in self.fuse:
                if key.startswith("pre") or key.startswith("suc"):
                    k1 = key[:3]
                    k2 = int(key[3:])
                    temp.index_add_(
                        0,
                        graph[k1][k2]["u"],
                        self.fuse[key][i](feat[graph[k1][k2]["v"]]),
                    )

            feat = self.fuse["norm"][i](temp)
            feat = self.relu(feat)

            feat = self.fuse["ctr2"][i](feat)
            feat += res
            feat = self.relu(feat)
            res = feat
        return feat


class M2A(nn.Module):
    """
    Class for the lane to actor block: fuses updated
        map information from lane nodes to actor nodes.
    """

    def __init__(self, n_actor: int, n_map: int, config: str):
        """
        Intilializes the M2A network. M2A fuses updated map features with 
        real-time traffic information back to the actors. Given a lane node, the
        features from its context actor nodes are aggregated using a spatial 
        attention layer.
        :param n_map: the number of lane nodes in the map.
        :param n_actor: the number of actor nodes.
        :param config: the path to the network's configuration file.
        """
        super(M2A, self).__init__()
        norm = "GN"
        ng = 1
        self.config = config

        self.n_actor = n_actor
        self.n_map = n_map

        att = []
        for i in range(2):
            # attention module
            att.append(Att(self.n_actor, self.n_map))
        self.att = nn.ModuleList(att)

    def forward(self,
                actors: Tensor,
                actor_idcs: List[Tensor],
                actor_ctrs: List[Tensor],
                nodes: Tensor,
                node_idcs: List[Tensor],
                node_ctrs: List[Tensor]) -> Tensor:
        """
        Performs a forward pass of the M2A network.
        :param actors: the actor nodes feature map [M x n_actor] (output of 
                        ActorNet).
        :param actor_idcs: list of actor indices.
        :param actor_ctrs: list of actor ctrs.
        :param nodes: the lane nodes feature map [N x n_map] (output of M2M). 
        :param node_idcs: list of lane node indices.
        :param node_ctrs: list of lane node ctrs.
        :returns: the updated actor nodes features [M x n_actor] (input of A2A).
        """
        for i in range(len(self.att)):
            actors = self.att[i](
                actors,
                actor_idcs,
                actor_ctrs,
                nodes,
                node_idcs,
                node_ctrs,
                self.config["map2actor_dist"],
            )
        return actors


class A2A(nn.Module):
    """
    Class for the actor to actor block: performs interactions among actors.
    """

    def __init__(self, n_actor: int, config: str):
        """
        Intilializes the A2A network. A2A handles the interactions between 
        actors and produces the output actor features, which are then used by 
        the prediction header for motion forecasting. Given an actor node, the
        features from its context actor nodes are aggregated using a spatial 
        attention layer.
        :param n_actor: the number of actor nodes.
        :param config: the path to the network's configuration file.
        """
        super(A2A, self).__init__()
        norm = "GN"
        ng = 1
        self.config = config

        self.n_actor = n_actor

        att = []
        for i in range(2):
            att.append(Att(self.n_actor, self.n_actor))
        self.att = nn.ModuleList(att)

    def forward(self,
                actors: Tensor,
                actor_idcs: List[Tensor],
                actor_ctrs: List[Tensor]) -> Tensor:
        """
        Performs a forward pass of the M2A network.
        :param actors: the actor nodes feature map [M x n_actor] (output of 
                        M2A).
        :param actor_idcs: list of actor indices.
        :param actor_ctrs: list of actor ctrs.
        :returns: the updated actor nodes features [M x n_actor] (input of 
                        PredNet).
        """
        for i in range(len(self.att)):
            actors = self.att[i](
                actors,
                actor_idcs,
                actor_ctrs,
                actors,
                actor_idcs,
                actor_ctrs,
                self.config["actor2actor_dist"],
            )
        return actors


class PredNet(nn.Module):
    """
    Class for the final motion forecasting with Linear Residual block.
    """

    def __init__(self, num_mods: int, n_actor: int, num_preds: int,
                 config: str):
        """
        Intilializes the prediction network. 
        :param n_mods: the number of predicted trajectories.
        :param n_actor: the number of actor nodes
        :param num_preds: the number of predicted frames.
        :param config: the path to the network's configuration file.
        """
        super(PredNet, self).__init__()
        norm = "GN"
        ng = 1
        self.config = config

        self.num_mods = num_mods
        self.n_actor = n_actor
        self.num_preds = num_preds

        pred = []
        for i in range(self.num_mods):
            pred.append(
                nn.Sequential(
                    LinearRes(self.n_actor, self.n_actor, norm=norm, ng=ng),
                    nn.Linear(self.n_actor, 2 * self.num_preds),
                )
            )
        self.pred = nn.ModuleList(pred)

        self.att_dest = AttDest(self.n_actor)
        self.cls = nn.Sequential(
            LinearRes(self.n_actor, self.n_actor, norm=norm,
                      ng=ng), nn.Linear(self.n_actor, 1)
        )

    def forward(self,
                actors: Tensor,
                actor_idcs: List[Tensor],
                actor_ctrs: List[Tensor]) -> Dict[str, List[Tensor]]:
        """
        Takes the after-fusion actor features as input and for each actor 
        predicts K possible future trajectories and their confidence scores.
        Performs a forward pass of the prediction network.
        :param actors: the actor nodes feature map [M x n_actor] (output of 
                        A2A)
        :param actor_idcs: list of actor indices.
        :param actor_ctrs: list of actor ctrs.
        :returns: a dictionary containing (i) a regression head with K possible 
        future trajectories for each actor [M x K x num_pred x 2] and (ii) 
        their respective confidence scores [M x K].
        """
        preds = []
        for i in range(len(self.pred)):
            preds.append(self.pred[i](actors))
        reg = torch.cat([x.unsqueeze(1) for x in preds], 1)
        reg = reg.view(reg.size(0), reg.size(1), -1, 2)

        for i in range(len(actor_idcs)):
            idcs = actor_idcs[i]
            ctrs = actor_ctrs[i].view(-1, 1, 1, 2)
            reg[idcs] = reg[idcs] + ctrs

        dest_ctrs = reg[:, :, -1].detach()
        feats = self.att_dest(actors, torch.cat(actor_ctrs, 0), dest_ctrs)
        cls = self.cls(feats).view(-1, self.config["num_mods"])

        cls, sort_idcs = cls.sort(1, descending=True)
        row_idcs = torch.arange(len(sort_idcs)).long().to(sort_idcs.device)
        row_idcs = row_idcs.view(-1, 1).repeat(1, sort_idcs.size(1)).view(-1)
        sort_idcs = sort_idcs.view(-1)
        reg = reg[row_idcs, sort_idcs].view(cls.size(0), cls.size(1), -1, 2)

        out = dict()
        out["cls"], out["reg"] = [], []
        for i in range(len(actor_idcs)):
            idcs = actor_idcs[i]
            ctrs = actor_ctrs[i].view(-1, 1, 1, 2)
            out["cls"].append(cls[idcs])
            out["reg"].append(reg[idcs])
        return out
