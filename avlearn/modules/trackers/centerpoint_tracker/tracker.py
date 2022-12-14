# This module is implemented on top of https://github.com/tianweiy/CenterPoint
# open source code base.

import copy
import json
import time
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
from nuscenes import NuScenes
from nuscenes.utils import splits
from tqdm import tqdm

from avlearn.apis.evaluate import Evaluator
from avlearn.modules.__base__ import BaseTracker

from .configs.nuscenes import (NUSCENE_CLS_VELOCITY_ERROR,
                               NUSCENES_TRACKING_CLASSES)
from .utils import greedy_assignment, linear_assignment


class CenterPointTracker(BaseTracker):
    """
    CenterPoint tracker class for av-learn.
    """

    def __init__(self,
                 match_algorithm: str = "greedy",
                 dataset: str = "nuscenes",
                 max_age: int = 3):
        """
        Loads the initial CenterPoint tracker parameters.
        :param match_algorithm: defines the matching algorithm used
                        (Default: "greedy").
        :param dataset: the used dataset (Default: "nuscenes")
        :param max_age: the maximum number frames allowed for a tracker to have
                        no matches before being deactivated (Default: 3).
        """
        if (match_algorithm not in {"hungarian", "greedy"}):
            raise ValueError("match_algorithm takes only values {'hungarian',"
                             "'greedy'}.")
        self.match_algorithm = match_algorithm
        self.dataset = dataset
        self.max_age = max_age

    def forward(self,
                dataroot: Union[str, Path],
                det_path: Union[str, Path],
                work_dir: Union[str, Path] = None,
                data_version: str = "v1.0-trainval",
                **kwargs):
        """
        Executes the tracking process for each dataset

        :param dataroot: The path to the dataset.
        :param det_path: Path to the json file with the detections.
        :param work_dir: The path to which the results will be saved .
        :param data_version: The version of the dataset used.
        """
        # TODO: add support for multiple datasets
        if self.dataset == "nuscenes":
            self.track_nuscenes(dataroot, det_path, work_dir, data_version)

    def track_nuscenes(
            self,
            dataroot: Union[str, Path],
            det_path: Union[str, Path],
            work_dir: Union[str, Path],
            data_version: str = "v1.0-trainval",
            save: bool = True) -> Dict[str, Any]:
        """
        :param dataroot: The path to the dataset.
        :param det_path: Path to the json file with the detections.
        :param work_dir: The path to which the results will be saved.
        :param data_version: The version of the dataset used.

        Outputs the CenterPoint filter tracklets in json format, as specified by
        the nuscenes dataset:
        submission {
            "results": {
                sample_token <str>: List[sample_result] -- Maps each
                                    sample_token to a list of sample_results.
            },
            "meta": {
                "use_camera":   <bool>  -- Whether this submission uses camera
                                           data as an input.
                "use_lidar":    <bool>  -- Whether this submission uses lidar
                                           data as an input.
                "use_radar":    <bool>  -- Whether this submission uses radar
                                           data as an input.
                "use_map":      <bool>  -- Whether this submission uses map data
                                           as an input.
                "use_external": <bool>  -- Whether this submission uses external
                                           data as an input.
            }
        }
        """
        # save first item
        # create a Database object for nuScenes
        nusc = NuScenes(version=data_version, dataroot=dataroot,
                        verbose=True)

        # select appropriate split, based on version
        if data_version == 'v1.0-trainval':
            scenes = splits.val
        elif data_version == 'v1.0-test':
            scenes = splits.test
        elif data_version == 'v1.0-mini':
            scenes = splits.mini_val
        else:
            raise ValueError("unknown")

        frames = []
        for sample in nusc.sample:
            scene_name = nusc.get("scene", sample['scene_token'])['name']
            # check if scenes is present in the NuScenes database
            if scene_name not in scenes:
                continue

            timestamp = sample["timestamp"] * 1e-6
            token = sample["token"]
            frame = {}
            frame['token'] = token
            frame['timestamp'] = timestamp

            # find starts of a sequence, i.e. first frames
            if sample['prev'] == '':
                frame['first'] = True
            else:
                frame['first'] = False
            frames.append(frame)

        # delete database
        del nusc

        # create result folder
        if work_dir is None:
            res_dir = Path("results/trackings/centerpoint/")
        else:
            res_dir = Path(work_dir) / "trackings/centerpoint/"
        res_dir.mkdir(parents=True, exist_ok=True)

        # save frame metadata
        with open(res_dir / 'frames_meta.json', "w") as f:
            json.dump({'frames': frames}, f)

        # initialize a tracker
        tracker = CenterTracker(max_age=self.max_age,
                                match_algorithm=self.match_algorithm)

        # read json file with object detections
        with open(det_path, 'rb') as f:
            detections = json.load(f)['results']

        # read json file with frame metadata
        with open(res_dir / 'frames_meta.json', 'rb') as f:
            frames = json.load(f)['frames']

        # create dictionary to save the results
        nusc_annos = {
            "results": {},
            "meta": None,
        }
        size = len(frames)

        print("Begin Tracking \n")
        start = time.time()
        # for scene token in the detection results file
        for i in tqdm(range(size)):
            token = frames[i]['token']

            # reset tracking after one video sequence
            if frames[i]['first']:
                # use this for sanity check to ensure your token order is
                # correct
                tracker.reset()
                last_time_stamp = frames[i]['timestamp']

            time_lag = (frames[i]['timestamp'] - last_time_stamp)
            last_time_stamp = frames[i]['timestamp']

            # get token detections
            dets = detections[token]

            # perfrom a tracking step for the current frame
            outputs = tracker.step_centertrack(dets, time_lag)
            annos = []

            # append the results for each box in the scene
            for item in outputs:
                if item['active'] == 0:
                    continue
                nusc_anno = {
                    "sample_token": token,
                    "translation": item['translation'],
                    "size": item['size'],
                    "rotation": item['rotation'],
                    "velocity": item['velocity'],
                    "tracking_id": str(item['tracking_id']),
                    "tracking_name": item['detection_name'],
                    "tracking_score": item['detection_score'],
                }
                annos.append(nusc_anno)
            nusc_annos["results"].update({token: annos})

        end = time.time()
        total_time = (end-start)
        speed = size / total_time

        nusc_annos["meta"] = {
            "use_camera": False,
            "use_lidar": True,
            "use_radar": False,
            "use_map": False,
            "use_external": False,
        }

        # save tracking results
        with open(res_dir / 'tracking_result.json', "w") \
                as f:
            json.dump(nusc_annos, f)

        print("Total Tracking took: {} or {} FPS"
              .format(total_time, speed))

        return nusc_annos

    def evaluate(
            self,
            dataroot: Union[str, Path],
            work_dir: Union[str, Path],
            track_path: Union[str, Path] = None,
            det_path: Union[str, Path] = None,
            data_version: str = "v1.0-trainval",
            split: str = "val",
            **kwargs) -> None:
        """
        :param dataroot: The path to the dataset.
        :param work_dir: The path to which the results will be saved.
        :param track_path: Path to the json file with the trackings.
        :param det_path: Path to the json file with the detections.
        :param data_version: The version of the dataset used.
        :param split: Which dataset split to use.

        Executes the CenterPointTracker evaluation process for each dataset.
        """
        print("Evaluating tracking module...")
        if self.dataset == "nuscenes":
            self.__evaluate_nuscenes(
                dataroot, work_dir, track_path, det_path,
                data_version, split)

    def __evaluate_nuscenes(self,
                            dataroot: Union[str, Path],
                            work_dir: Union[str, Path],
                            track_path: Union[str, Path] = None,
                            det_path: Union[str, Path] = None,
                            data_version: str = "v1.0-trainval",
                            split: str = "val") -> None:
        """
        :param dataroot: The path to the dataset.
        :param work_dir: The path to which the results will be saved.
        :param track_path: Path to the json file with the trackings.
        :param det_path: Path to the json file with the detections.
        :param data_version: The version of the dataset used.
        :param split: Which dataset split to use.

        Executes the CenterPointTracker evaluation process for nuScenes.
        """
        if track_path is None:
            if det_path is None:
                print(
                    "Please specify a path to the detection results "
                    "('det_path') or to the tracking results ('track_path') "
                    "to be evaluated when calling the evaluate method."
                )
                exit()
            else:
                self.track_nuscenes(
                    dataroot=dataroot,
                    data_version=data_version,
                    det_path=det_path,
                    work_dir=work_dir)

        track_path = track_path if track_path is not None else Path(
            work_dir) / "trackings/centerpoint/tracking_result.json"

        if work_dir is None:
            work_dir = "results"
        save_path = Path(work_dir) / "evaluations/tracking/centerpoint/"

        evaluator = Evaluator(
            task="tracking",
            dataset="nuscenes",
            results=track_path,
            output=save_path,
            dataroot=dataroot,
            split=split,
            version=data_version,
            config_path=None,
            verbose=True,
            render_classes=None,
            render_curves=False,
            plot_examples=0)

        evaluator.evaluate()


class CenterTracker(object):
    """
    Center tracker class for av-learn, used to track individual video sequences. 
    Specifically, the object centers in the current frame are projected back to 
    the previous frame by applying the negative velocity estimate and then 
    matched to the tracked objects by closest distance matching.
    """

    def __init__(self,
                 match_algorithm: str = "greedy",
                 max_age: int = 3):
        """
        Loads the initial Center tracker parameters.
        :param match_algorithm: defines the matching algorithm used
                        (Default: "greedy").
        :param max_age: the maximum number frames allowed for a tracker to have 
                        no matches before being deactivated (Default: 3).
        """
        self.match_algorithm = match_algorithm
        self.max_age = max_age

        print("Use hungarian: {}".format(self.match_algorithm == "hungarian"))

        self.NUSCENE_CLS_VELOCITY_ERROR = NUSCENE_CLS_VELOCITY_ERROR

        self.reset()

    def reset(self):
        """
        Resets all the tracks in the CenterTracker.
        """
        self.id_count = 0
        self.tracks = []

    def step_centertrack(self, dets: list, time_lag: float) -> list:
        """
        Perfroms a tracking step for one frame token of a video sequence.
        :param dets: the object detections for this frame token.
        :param time_lag: the time lag of this frame from the first frame of the
                        video sequence.
        :returns: list containing matches, unmatched_detections and 
                    unmatched_trackers
        """
        if len(dets) == 0:
            self.tracks = []
            return []
        else:
            temp = []
            for det in dets:
                # filter out classes not evaluated for tracking
                if det['detection_name'] not in NUSCENES_TRACKING_CLASSES:
                    continue

                det['ct'] = np.array(det['translation'][:2])
                det['tracking'] = np.array(det['velocity'][:2]) * -1 * time_lag
                det['label_preds'] = NUSCENES_TRACKING_CLASSES.index(
                    det['detection_name'])
                temp.append(det)

            detections = temp

        # number of detections in this frame
        N = len(detections)
        # number of currenlty active tracks
        M = len(self.tracks)

        if 'tracking' in detections[0]:
            dets = np.array(
                [det['ct'] + det['tracking'].astype(np.float32)
                 for det in detections], np.float32)
        else:
            dets = np.array(
                [det['ct'] for det in detections], np.float32)

        # detection classes in current frame
        item_cls = np.array([item['label_preds']
                            for item in detections], np.int32)  # N
        # tracking classes in currently active tracks
        track_cls = np.array([track['label_preds']
                             for track in self.tracks], np.int32)  # M

        # max velocity error for each box, based on its class
        max_diff = np.array(
            [self.NUSCENE_CLS_VELOCITY_ERROR[box['detection_name']]
             for box in detections], np.float32)

        tracks = np.array(
            [pre_det['ct'] for pre_det in self.tracks], np.float32)  # M x 2

        # if not the first frame
        if len(tracks) > 0:
            dist = (((tracks.reshape(1, -1, 2) -
                      dets.reshape(-1, 1, 2)) ** 2).sum(axis=2))  # N x M
            # create absolute distance (in meter) matrix between detection and
            # tracking boxes
            dist = np.sqrt(dist)

            # sanity check of distances
            invalid = ((dist > max_diff.reshape(N, 1)) +
                       (item_cls.reshape(N, 1) != track_cls.reshape(1, M))) > 0

            dist = dist + invalid * 1e18
            if self.match_algorithm == "hungarian":
                # set 1e18 as maximum allowed distance
                dist[dist > 1e18] = 1e18
                # houngarian algorithm to solve linear matching problem
                matched_indices = linear_assignment(copy.deepcopy(dist))
            else:
                # greedy algorithm to solve linear matching problem
                matched_indices = greedy_assignment(copy.deepcopy(dist))
        # if first frame for this video sequence
        else:
            assert M == 0
            matched_indices = np.array([], np.int32).reshape(-1, 2)

        unmatched_dets = [d for d in range(dets.shape[0])
                          if not (d in matched_indices[:, 0])]

        unmatched_tracks = [d for d in range(tracks.shape[0])
                            if not (d in matched_indices[:, 1])]

        # in case of hungarian discard matched indices with a distance greater
        # than 1e16 as unmatched
        if self.match_algorithm == "hungarian":
            matches = []
            for m in matched_indices:
                if dist[m[0], m[1]] > 1e16:
                    unmatched_dets.append(m[0])
                else:
                    matches.append(m)
            matches = np.array(matches).reshape(-1, 2)
        else:
            matches = matched_indices

        ret = []
        # append matches to results
        for m in matches:
            track = detections[m[0]]
            track['tracking_id'] = self.tracks[m[1]]['tracking_id']
            track['age'] = 1
            track['active'] = self.tracks[m[1]]['active'] + 1
            ret.append(track)

        # append unmatched detections to results
        for i in unmatched_dets:
            track = detections[i]
            self.id_count += 1
            track['tracking_id'] = self.id_count
            track['age'] = 1
            track['active'] = 1
            ret.append(track)

        # append unmatched detections to results
        for i in unmatched_tracks:
            track = self.tracks[i]
        # still store unmatched tracks if its age doesn't exceed max_age,
        # however, we shouldn't output the object in current frame
            if track['age'] < self.max_age:
                track['age'] += 1
                track['active'] = 0
                ct = track['ct']

                # movement in the last second
                if 'tracking' in track:
                    offset = track['tracking'] * -1  # move forward
                    track['ct'] = ct + offset
                ret.append(track)

        self.tracks = ret
        return ret
