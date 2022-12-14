# This module is implemented on top of
# https://github.com/eddyhkchiu/mahalanobis_3d_multi_object_tracking
# and https://github.com/xinshuoweng/AB3DMOT tracking open source code bases.

from __future__ import print_function

import json
import time
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
from .ab3d_mot_tracker import AB3DMOT
from .configs.nuscenes import NUSCENES_TRACKING_CLASSES
from nuscenes import NuScenes
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.detection.data_classes import DetectionBox
from pyquaternion import Quaternion
from tqdm import tqdm

from avlearn.apis.evaluate import Evaluator
from avlearn.modules.__base__.tracker import BaseTracker

DEFAULT_CFG = "configs/covariance_base.py"


class KalmanTracker(BaseTracker):
    """
    Kalman tracker class for av-learn.
    """

    def __init__(self,
                 match_distance: str = "iou",
                 match_threshold: float = 0.1,
                 match_algorithm: str = "hungarian",
                 dataset: str = "nuscenes",
                 cfg_path: Union[str, Path] = None,
                 max_age: int = 2,
                 min_hits: int = 3):
        """
        Loads the initial Kalman tracker parameters.
        :param match_distance: defines the mathcing distance used, either iou 
                    or mahalanobis (Default: "iou")
        :param match_threshold: defines the matching threshold used in the
                    hungarian algorithm (Default: 0.1)
        :param match_algorithm: defines the matching algorithm used
                    (Default: "hungarian")
        :param dataset: the used dataset (Default: "nuscenes")
        :param cfg_path: path to configuration file for Kalman filter 
                    covariance matrices
        :param max_age: the maximum number frames allowed for a tracker to have
                        no matches before being deactivated
        :param min_hits: the minimum number matches allowed for a tracker 
                    before being activated
        """
        if (match_distance not in {"iou", "mahalanobis"}):
            raise ValueError("match_distance takes only values {'iou',"
                             "'mahalanobis'}.")
        if (match_algorithm not in {"hungarian", "greedy", "hungarian_thres"}):
            raise ValueError("match_algorithm takes only values {'hungarian',"
                             "'greedy', 'hungarian_thres'}.")

        if cfg_path is not None:
            self.cfg_path = cfg_path
        else:
            self.cfg_path = str(Path(
                __file__).parent / DEFAULT_CFG)

        self.match_distance = match_distance
        self.match_threshold = match_threshold
        self.match_algorithm = match_algorithm
        self.dataset = dataset
        self.max_age = max_age
        self.min_hits = min_hits

    def forward(
            self,
            dataroot: Union[str, Path],
            work_dir: Union[str, Path],
            det_path: Union[str, Path],
            data_version: str = "v1.0-trainval",
            **kwargs):
        """
        :param dataroot: The path to the dataset.
        :param work_dir: The path to which the results will be saved.
        :param det_path: Path to the json file with the detections.
        :param data_version: The version of the dataset used.

        Executes the Kalman tracking process for each dataset
        """
        # TODO: add support for multiple datasets
        if self.dataset == "nuscenes":
            self.track_nuscenes(dataroot, work_dir,
                                det_path, data_version, **kwargs)

    def track_nuscenes(
            self,
            dataroot: Union[str, Path],
            work_dir: Union[str, Path],
            det_path: Union[str, Path],
            data_version: str = "v1.0-trainval",
            save: bool = True) -> Dict[str, Any]:
        """
        :param dataroot: The path to the dataset.
        :param work_dir: The path to which the results will be saved.
        :param det_path: Path to the json file with the detections.
        :param data_version: The version of the dataset used.
        :param save: Whether to save the tracking results.

        Outputs the Kalman filter tracklets in json format, as specified by the
        nuscenes dataset:
        submission {
            "meta": {
                "use_camera":   <bool>  -- Whether this submission uses camera
                                           data as an input.
                "use_lidar":    <bool>  -- Whether this submission uses lidar
                                           data as an input.
                "use_radar":    <bool>  -- Whether this submission uses radar
                                           data as an input.
                "use_map":      <bool>  -- Whether this submission uses map 
                                           data as an input.
                "use_external": <bool>  -- Whether this submission uses 
                                           external data as an input.
            },
            "results": {
                sample_token <str>: List[sample_result] -- Maps each
                                    sample_token to a list of sample_results.
            }
        }
        """
        detection_file = det_path

        # create directory folder for the results
        if work_dir is not None:
            save_dir = Path(work_dir)
        else:
            save_dir = Path(f"results")

        save_dir.mkdir(parents=True, exist_ok=True)
        print("Results saved in:", save_dir)
        output_path = save_dir / "trackings/kalman/tracking_result.json"

        # create a Database object for nuScenes
        nusc = NuScenes(version=data_version, dataroot=dataroot, verbose=True)

        results = {}
        total_time = 0.0
        total_frames = 0
        total_scenes = 0
        total_bb_boxes = 0

        # read json file with object detections
        with open(detection_file) as f:
            det_data = json.load(f)
        assert 'results' in det_data, 'Error: No field `results` in result ' \
            'file. Please note that the result format changed. See'\
            'https://www.nuscenes.org/object-detection for more information.'

        # group EvalBox instances by sample.
        det_results = EvalBoxes.deserialize(det_data['results'], DetectionBox)
        # save metadata, related to the dataset
        meta = det_data['meta']
        print('meta: ', meta)
        print("Loaded results from {}. Found detections for {} samples."
              .format(detection_file, len(det_results.sample_tokens)))

        processed_scenes = set()
        # for scene token in the detection results file
        for sample_token_idx in tqdm(range(len(det_results.sample_tokens))):
            sample_token = det_results.sample_tokens[sample_token_idx]
            # if the scene is present in the NuScenes database
            if sample_token in nusc._token2ind["sample"].keys():
                scene_token = nusc.get('sample', sample_token)['scene_token']
                if scene_token in processed_scenes:
                    continue
                total_scenes += 1
                # extract first sample/keyframe token for this scene
                first_sample_token = nusc.get('scene', scene_token)[
                    'first_sample_token']
                current_sample = first_sample_token
                # initialize an AB3DMOT tracker for each of the tracked classes
                mot_trackers = {tracking_name: AB3DMOT(
                                tracking_name=tracking_name,
                                cfg_path=self.cfg_path,
                                tracking_nuscenes=True,
                                max_age=self.max_age,
                                min_hits=self.min_hits)
                                for tracking_name in NUSCENES_TRACKING_CLASSES}

                # for all sample/keyframe tokens in the scene
                while current_sample != '':
                    results[current_sample] = []
                    dets = {tracking_name: []
                            for tracking_name in NUSCENES_TRACKING_CLASSES}
                    score = {tracking_name: []
                             for tracking_name in NUSCENES_TRACKING_CLASSES}
                    # for each bounding box in the current sample/keyframe token
                    for box in det_results.boxes[current_sample]:
                        # only consider bounding boxes for {'bicycle','bus',
                        # 'car','motorcycle','pedestrian','trailer','truck'}
                        if box.detection_name not in NUSCENES_TRACKING_CLASSES:
                            continue
                        # define Quaternion object to represent rotation in 3D
                        # space.
                        q = Quaternion(box.rotation)
                        angle = q.angle if q.axis[2] > 0 else -q.angle

                        # store detection array containing the translation
                        # (x,y,z), size (s_x,s_y,s_z), and rotation (quaternion
                        # 3D angle)
                        detection = np.array([
                            box.size[2], box.size[0], box.size[1],
                            box.translation[0],  box.translation[1],
                            box.translation[2],
                            angle])

                        # store box detection and detection score
                        det_score = np.array([box.detection_score])
                        dets[box.detection_name].append(detection)
                        score[box.detection_name].append(det_score)
                        total_bb_boxes += 1

                    # store for all boxes
                    dets_all = {tracking_name: {'dets': np.array(
                                                dets[tracking_name]),
                                                'scores': np.array(
                                                score[tracking_name])}
                                for tracking_name in NUSCENES_TRACKING_CLASSES}

                    total_frames += 1
                    start_time = time.time()
                    for tracking_name in NUSCENES_TRACKING_CLASSES:
                        # for detected classes, update tracks using detections
                        # of curremt frame
                        if dets_all[tracking_name]['dets'].shape[0] > 0:
                            trackers = mot_trackers[tracking_name].update(
                                dets_all[tracking_name], self.match_distance,
                                self.match_threshold, self.match_algorithm)

                            for i in range(trackers.shape[0]):
                                # create output format for each sample token
                                # result
                                sample_result = format_sample_result(
                                    current_sample, tracking_name,
                                    trackers[i])
                                results[current_sample].append(
                                    sample_result)
                    cycle_time = time.time() - start_time
                    total_time += cycle_time

                    # get next frame and continue the while loop
                    current_sample = nusc.get(
                        'sample', current_sample)['next']

                # left while loop and mark this scene as processed
                processed_scenes.add(scene_token)

        # finished tracking all scenes, write output data
        output_data = {'meta': meta, 'results': results}

        if save:
            with open(output_path, 'w') as outfile:
                json.dump(output_data, outfile)

        print("Total Number of frames processed from detection file: {}"
              .format(total_frames))
        print("Total Number of scene tokens processed from detection file: {}"
              .format(total_scenes))
        print("Total Number of bounding boxes processed from detection file: {}"
              .format(total_bb_boxes))
        print("Total Tracking took: {} or {} FPS"
              .format(total_time, total_frames/total_time))

        return output_data

    def evaluate(self,
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

        Executes the Kalman evaluation process for each dataset.
        """
        print("Evaluating tracking module")
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

        Executes the Kalman evaluation process for nuScenes.
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
            work_dir) / "trackings/kalman/tracking_result.json"

        if work_dir is None:
            work_dir = "results"
        save_path = Path(work_dir) / "evaluations/tracking/kalman/"

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


def format_sample_result(sample_token: str,
                         tracking_name: str,
                         tracker: np.ndarray) -> dict:
    """
    Takes as input a tracking bounding box and produces a sample result
    dictionary in the nuscenes format.
    :param sample_token: the sample/keyframe associated with this bounding box
    :param tracking_name: predicted class for this sample_result, e.g. car,
                         pedestrian
    :param tracker: an [h, w, l, x, y, z, rot_y] array of the 3D bounding box
    :return: dictionary in format:

    sample_result {
        "sample_token":   <str>       -- Foreign key. Identifies the sample/
                                         keyframe for which objects are 
                                         detected.
        "translation":    <float> [3] -- Estimated bounding box location in
                                         meters in the global frame: center_x,
                                         center_y, center_z.
        "size":           <float> [3] -- Estimated bounding box size in meters:
                                         width, length, height.
        "rotation":       <float> [4] -- Estimated bounding box orientation as
                                         quaternion in the global frame: w, x,
                                         y, z.
        "velocity":       <float> [2] -- Estimated bounding box velocity in m/s
                                         in the global frame: vx, vy.
        "tracking_id":    <str>       -- Unique object id that is used to
                                         identify an object track across 
                                         samples.
        "tracking_name":  <str>       -- The predicted class for this
                                         sample_result, e.g. car, pedestrian.
                                         Note that the tracking_name cannot
                                         change throughout a track.
        "tracking_score": <float>     -- Object prediction score between 0 and 
                                         1 for the class identified by
                                         tracking_name. We average over frame
                                         level scores to compute the track 
                                         level score. The score is used to 
                                         determine positive and negative tracks
                                         via thresholding.
}
    """
    rotation = Quaternion(axis=[0, 0, 1], angle=tracker[6]).elements
    sample_result = {
        'sample_token': sample_token,
        'translation': [tracker[3], tracker[4], tracker[5]],
        'size': [tracker[1], tracker[2], tracker[0]],
        'rotation': [rotation[0], rotation[1], rotation[2], rotation[3]],
        'velocity': [0, 0],
        'tracking_id': str(int(tracker[7])),
        'tracking_name': tracking_name,
        'tracking_score': tracker[8]
    }

    return sample_result
