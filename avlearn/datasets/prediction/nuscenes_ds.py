"""
This module implements a nuScenes prediction dataset, used for the training 
of LaneGCN. It is built on top of
    https://github.com/bdokim/LaPred/blob/master/data_process.py 
and 
    https://github.com/uber-research/LaneGCN/blob/master/data.py
"""
import copy
from typing import Tuple

import numpy as np
from nuscenes import NuScenes
from nuscenes.eval.prediction.splits import get_prediction_challenge_split
from nuscenes.prediction import PredictHelper
from nuscenes.prediction.input_representation.static_layers import \
    load_all_maps
from torch.utils.data import Dataset

from map_representation.nuscenes_map import NuScenesMap
from utils.data_utils import dilated_nbrs


class NuScenesDataset(Dataset):
    """This class implements a nuScenes prediction dataset.

    :param root_dir: Path to nuScenes data.
    :param config: Dictionary with configurations for the dataset.
    :param map: Whether or not to return map information of the centerlines.
    :param maps_dir: Path to data files created for the map representation of
        the centerlines via the `create_nuscenes_graph.py` script. Required
        if `map` is set to True.
    :param split: Which split of nuScenes to use (e.g. 'train', 'mini_train')
    :pram train: Whether this dataset will be used for training.


    Returns
        A dictionary containing data for the following keys:
        [
            'ctrs', 'feats', 'orig', 'theta', 'rot', 'ori_trajs', 'gt_preds', 
            'has_preds', 'ins_sam', 'city', 'idx', 'graph'
        ]
    """

    def __init__(
            self,
            root_dir: str,
            config: dict,
            use_map: bool = True,
            maps_dir: str = None,
            split: str = 'train',
            train: bool = True
    ) -> None:
        self.config = config
        self.train = train
        self.root_dir = root_dir
        self.use_map = use_map

        self.ns = NuScenes("v1.0-trainval", dataroot=root_dir)
        self.helper = PredictHelper(self.ns)
        self.maps = load_all_maps(self.helper)
        self.token_list = get_prediction_challenge_split(
            split, dataroot=root_dir)

        if self.use_map and maps_dir is not None:
            # Initialize nuScenes map of centerlines
            self.nsmap = NuScenesMap(root_dir=maps_dir)
        elif self.use_map:
            raise FileNotFoundError(
                """Please provide path to map representation files generated 
                   by the create_nuscenes_graph.py script.""")

    def __getitem__(self, idx: int) -> dict:
        """Return a sample."""
        instance_token, sample_token = self.token_list[idx].split("_")
        map_name = self.helper.get_map_name_from_sample_token(sample_token)
        self.map_api = self.maps[map_name]

        data = self.__get_agent_feats(instance_token, sample_token)

        data['city'] = self.helper.get_map_name_from_sample_token(sample_token)
        data['idx'] = idx

        if self.use_map:
            data['graph'] = self.__get_lane_graph(data)

        return data

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.token_list)

    def __get_agent_feats(
            self, instance_token: str, sample_token: str) -> dict:
        """This method returns prediction features about the agent of a
        specific `instance_token` and `sample_token`.

        :param instance_token: nuScenes instance token.
        :param sample_token: nuScenes sample token.
        """
        past_traj = self.helper.get_past_for_agent(
            instance_token, sample_token, seconds=self.config['train_size'],
            in_agent_frame=False)
        past_traj = np.asarray(past_traj, dtype=np.float32)

        cur_traj = self.helper.get_sample_annotation(
            instance_token, sample_token)["translation"][:2]
        orig = np.asarray(cur_traj, dtype=np.float32)

        prev = past_traj[0] - orig

        if self.train and self.config['rot_aug']:
            theta = np.random.rand() * np.pi * 2.0
        else:
            theta = np.pi - np.arctan2(prev[1], prev[0])

        rot = np.asarray([[np.cos(theta), -np.sin(theta)],
                          [np.sin(theta), np.cos(theta)]], np.float32)

        ori_trajs, trajs, gt_preds, has_preds = [], [], [], []
        ori_traj, agt_traj, agt_gt_pred, agt_has_pred = \
            self.get_trajs(instance_token, sample_token, orig, rot)

        trajs.append(agt_traj)
        ori_trajs.append(ori_traj)
        gt_preds.append(agt_gt_pred)
        has_preds.append(agt_has_pred)

        present_history = self.helper.get_annotations_for_sample(sample_token)
        for pre_h in range(len(present_history)):
            if present_history[pre_h]['category_name'][:7] == 'vehicle':
                nei_ins, nei_sam = (
                    present_history[pre_h]["instance_token"],
                    present_history[pre_h]["sample_token"])
                ori_traj, nei_traj, nei_gt_pred, nei_has_pred = self.get_trajs(
                    nei_ins,
                    nei_sam,
                    orig,
                    rot)

                if len(nei_traj) == 1:
                    continue
                if np.sum(trajs[0]-nei_traj) == 0.:
                    continue

                x_min, x_max, y_min, y_max = self.config['pred_range']
                if (nei_traj[-1, 0] < x_min or
                    nei_traj[-1, 0] > x_max or
                    nei_traj[-1, 1] < y_min or
                        nei_traj[-1, 1] > y_max):
                    continue

                trajs.append(nei_traj)
                ori_trajs.append(ori_traj)
                gt_preds.append(nei_gt_pred)
                has_preds.append(nei_has_pred)

        ori_trajs = np.asarray(ori_trajs, np.float32)
        trajs = np.asarray(trajs, np.float32)
        ctrs = np.asarray(trajs[:, -1, :2], np.float32)

        gt_preds = np.asarray(gt_preds, np.float32)
        has_preds = np.asarray(has_preds, np.bool)

        data = dict()

        data['ctrs'] = ctrs
        data['feats'] = trajs
        data['orig'] = orig
        data['theta'] = theta
        data['rot'] = rot
        data['ori_trajs'] = ori_trajs
        data['gt_preds'] = gt_preds
        data['has_preds'] = has_preds
        data['ins_sam'] = [instance_token, sample_token]
        return data

    def get_trajs(self,
                  instance_token: str,
                  sample_token: str,
                  orig: np.ndarray,
                  rot: np.ndarray
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute trajectory data for the corresponding `instance_token` and
        `sample_token` of nuScenes.

        :param instance_token: nuScenes instance token.
        :param sample_token: nuScenes sample token.
        :param orig: Current coordinates of the object.
        :param rot: Rotation matrix for the object.
        """

        past_traj = self.helper.get_past_for_agent(
            instance_token, sample_token, seconds=self.config['train_size'],
            in_agent_frame=False)

        past_traj = np.asarray(past_traj, dtype=np.float32)

        if past_traj.shape[0] == 0:
            return [0], [0], [0], [0]

        traj_records = self.helper.get_past_for_agent(
            instance_token, sample_token, seconds=self.config['train_size'],
            in_agent_frame=False, just_xy=False)

        if past_traj.shape[0] > self.config['train_size'] * 2:
            past_traj = past_traj[0:int(self.config['train_size']) * 2]
            traj_records = traj_records[0:int(self.config['train_size']) * 2]

        cur_traj = self.helper.get_sample_annotation(
            instance_token, sample_token)["translation"][:2]
        cur_traj = np.asarray(cur_traj, dtype=np.float32)

        all_trajs = np.zeros((past_traj.shape[0] + 1, 2), np.float32)
        all_trajs[0, :] = cur_traj
        all_trajs[1:, :] = past_traj
        ori_traj = copy.deepcopy(all_trajs)

        trajs = np.zeros((all_trajs.shape[0], 6), dtype=np.float32)
        trajs[:, 0:2] = np.matmul(rot, (all_trajs - orig.reshape(-1, 2)).T).T

        sam_tokens = [traj_records[i]['sample_token']
                      for i in range(len(traj_records))]
        sam_tokens.insert(0, sample_token)

        i_t = instance_token
        for k in range(len(sam_tokens)):
            s_t = sam_tokens[k]
            v = self.helper.get_velocity_for_agent(i_t, s_t)
            a = self.helper.get_acceleration_for_agent(i_t, s_t)
            theta = self.helper.get_heading_change_rate_for_agent(i_t, s_t)

            if np.isnan(v):
                v = 0
            if np.isnan(a):
                a = 0
            if np.isnan(theta):
                theta = 0
            trajs[k, 2] = v
            trajs[k, 3] = a
            trajs[k, 4] = theta
            trajs[k, 5] = 1.0

        traj_zeropadded = np.zeros(
            (int(self.config['train_size']) * 2 + 1, 6),
            dtype=np.float32)
        ori_zeropadded = np.zeros(
            (int(self.config['train_size']) * 2 + 1, 2),
            dtype=np.float32)

        trajs = np.flip(trajs, 0)
        traj_zeropadded[-trajs.shape[0]:] = trajs

        ori_traj = np.flip(ori_traj, 0)
        ori_zeropadded[-ori_traj.shape[0]:] = ori_traj

        agt_pred = np.zeros((self.config["pred_size"] * 2, 2), np.float32)
        agt_has_pred = np.zeros(self.config["pred_size"] * 2, np.bool)

        agt_gt_trajs = self.helper.get_future_for_agent(
            instance_token, sample_token, seconds=self.config["pred_size"],
            in_agent_frame=False)

        agt_gt_trajs = np.asarray(agt_gt_trajs, dtype=np.float32)

        if agt_gt_trajs.shape[0] == 0:
            return [0], [0], [0], [0]

        agt_pred[:agt_gt_trajs.shape[0], :] = agt_gt_trajs
        agt_has_pred[:agt_gt_trajs.shape[0]] = 1

        return np.asarray(ori_zeropadded, np.float32), \
            np.asarray(traj_zeropadded, np.float32), agt_pred, agt_has_pred

    def __get_lane_graph(self, data: dict) -> dict:
        """Get data from the centerline graph."""
        x_min, x_max, y_min, y_max = self.config['pred_range']
        radius = max(abs(x_min), abs(x_max)) + max(abs(y_min), abs(y_max))
        lane_tokens = self.nsmap.get_lane_tokens_in_xy_bbox(
            data['orig'][0], data['orig'][1], data['city'], radius)
        lane_tokens = copy.deepcopy(lane_tokens)

        lanes = dict()
        for lane_token in lane_tokens:
            lane = self.nsmap.city_lane_centerlines_dict[data['city']][
                lane_token]
            lane = copy.deepcopy(lane)
            centerline = np.matmul(
                data['rot'],
                (lane.centerline - data['orig'].reshape(-1, 2)).T).T
            x, y = centerline[:, 0], centerline[:, 1]
            if (x.max() < x_min or
                x.min() > x_max or
                y.max() < y_min or
                    y.min() > y_max):
                continue
            else:
                """Getting polygons requires original centerline"""
                polygon = self.nsmap.get_lane_segment_polygon(
                    lane_token, data['city'])
                polygon = copy.deepcopy(polygon)
                lane.centerline = centerline
                lane.polygon = np.matmul(
                    data['rot'],
                    (polygon[:, : 2] - data['orig'].reshape(-1, 2)).T).T
                lanes[lane_token] = lane

        lane_tokens = list(lanes.keys())
        ctrs, feats = [], []
        for lane_token in lane_tokens:
            lane = lanes[lane_token]
            ctrln = lane.centerline

            ctrs.append(np.asarray((ctrln[:-1] + ctrln[1:]) / 2.0, np.float32))
            feats.append(np.asarray(ctrln[1:] - ctrln[:-1], np.float32))

        node_idcs = []
        count = 0
        for i, ctr in enumerate(ctrs):
            node_idcs.append(range(count, count + len(ctr)))
            count += len(ctr)
        num_nodes = count

        pre, suc = dict(), dict()
        for key in ['u', 'v']:
            pre[key], suc[key] = [], []
        for i, lane_token in enumerate(lane_tokens):
            lane = lanes[lane_token]
            idcs = node_idcs[i]

            pre['u'] += idcs[1:]
            pre['v'] += idcs[:-1]
            if lane.predecessors is not None:
                for nbr_token in lane.predecessors:
                    if nbr_token in lane_tokens:
                        j = lane_tokens.index(nbr_token)
                        pre['u'].append(idcs[0])
                        pre['v'].append(node_idcs[j][-1])

            suc['u'] += idcs[:-1]
            suc['v'] += idcs[1:]
            if lane.successors is not None:
                for nbr_token in lane.successors:
                    if nbr_token in lane_tokens:
                        j = lane_tokens.index(nbr_token)
                        suc['u'].append(idcs[-1])
                        suc['v'].append(node_idcs[j][0])

        lane_idcs = []
        for i, idcs in enumerate(node_idcs):
            lane_idcs.append(i * np.ones(len(idcs), np.int64))
        lane_idcs = np.concatenate(lane_idcs, 0)

        pre_pairs, suc_pairs = [], []
        for i, lane_token in enumerate(lane_tokens):
            lane = lanes[lane_token]

            nbr_tokens = lane.predecessors
            if nbr_tokens is not None:
                for nbr_token in nbr_tokens:
                    if nbr_token in lane_tokens:
                        j = lane_tokens.index(nbr_token)
                        pre_pairs.append([i, j])

            nbr_tokens = lane.successors
            if nbr_tokens is not None:
                for nbr_token in nbr_tokens:
                    if nbr_token in lane_tokens:
                        j = lane_tokens.index(nbr_token)
                        suc_pairs.append([i, j])

        pre_pairs = np.asarray(pre_pairs, np.int64)
        suc_pairs = np.asarray(suc_pairs, np.int64)

        graph = dict()
        graph['ctrs'] = np.concatenate(ctrs, 0)
        graph['num_nodes'] = num_nodes
        graph['feats'] = np.concatenate(feats, 0)
        graph['pre'] = [pre]
        graph['suc'] = [suc]
        graph['lane_idcs'] = lane_idcs
        graph['pre_pairs'] = pre_pairs
        graph['suc_pairs'] = suc_pairs

        for k1 in ['pre', 'suc']:
            for k2 in ['u', 'v']:
                graph[k1][0][k2] = np.asarray(graph[k1][0][k2], np.int64)

        for key in ['pre', 'suc']:
            graph[key] += dilated_nbrs(graph[key][0],
                                       graph['num_nodes'],
                                       self.config['num_scales'])
        return graph
