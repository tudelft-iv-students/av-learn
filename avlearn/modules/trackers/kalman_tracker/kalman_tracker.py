# This method is implemented on top of
# https://github.com/eddyhkchiu/mahalanobis_3d_multi_object_tracking
# and https://github.com/xinshuoweng/AB3DMOT tracking open source code bases.

from __future__ import print_function
import os.path
import numpy as np
import time
import json
from nuscenes import NuScenes
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.detection.data_classes import DetectionBox
from pyquaternion import Quaternion
from tqdm import tqdm

from ab3d_mot_tracker import AB3DMOT
from utils import mkdir_if_missing, NUSCENES_TRACKING_CLASSES


class KalmanTracker(object):
    """
    Kalman tracker class for av-learn. 
    """

    def __init__(self,
                 data_path: str,
                 cfg_path: str,
                 det_path: str,
                 match_distance: str = "iou",
                 match_threshold: float = 0.1,
                 match_algorithm: str = "hungarian",
                 dataset: str = "nuscenes",
                 save_root: str = "results/default"):

        # python main.py val 0 iou 0.1 h false nuscenes results/000001
        """
        Loads the initial Kalman tracker parameters.
        :param data_path: the path to the dataset
        :param cfg_path: path to configuration file for Kalman filter covariance
                     matrices
        :param det_path: path to the json file with the detections
        :param match_distance: defines the mathcing distance used, either iou or
                        mahalanobis (Default: "iou")
        :param match_threshold: defines the matching threshold used in the 
                        hungarian algorithm (Default: 0.1)
        :param match_algorithm: defines the matching algorithm used
                        (Default: "hungarian")
        :param dataset: the used dataset (Default: "nuscenes")
        :param save_root: the path to which the results will be saved 
                        (Default: "results/default")
        """
        if (match_distance not in {"iou", "mahalanobis"}):
            raise ValueError("match_distance takes only values {'iou',"
                             "'mahalanobis'}.")
        if (match_algorithm not in {"hungarian", "greedy", "hungarian_thres"}):
            raise ValueError("match_algorithm takes only values {'hungarian',"
                             "'greedy', 'hungarian_thres'}.")
        self.data_path = data_path
        self.cfg_path = cfg_path
        self.det_path = det_path
        self.match_distance = match_distance
        self.match_threshold = match_threshold
        self.match_algorithm = match_algorithm
        self.dataset = dataset
        self.save_root = os.path.join('./' + save_root)

    def track(self):
        """
        Executes the Kalman tracking process for each dataset
        """
        # TODO: add support for multiple datasets
        if self.dataset == "nuscenes":
            self.track_nuscenes()

    def track_nuscenes(self):
        """
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
                "use_map":      <bool>  -- Whether this submission uses map data 
                                           as an input.
                "use_external": <bool>  -- Whether this submission uses external 
                                           data as an input.
            },
            "results": {
                sample_token <str>: List[sample_result] -- Maps each 
                                    sample_token to a list of sample_results.
            }
        }
        """
        # create directory folder for the results
        version = self.data_path.split("/")[-1]
        data_root = self.data_path[:-len(version)-1]

        save_dir = os.path.join(self.save_root, self.dataset + version)
        output_path = os.path.join(
            save_dir, 'results_tracking.json')
        mkdir_if_missing(save_dir)

        detection_file = self.det_path

        # nusc_dataset = NuScenesDataset("PATH TO NUSCENES CONFIG")
        # version = nusc_dataset.version
        # data_root = nusc_dataset.data_root
        # save_dir = os.path.join(self.save_root, self.dataset + version)
        # output_path = os.path.join(save_dir, 'results_tracking.json')
        # mkdir_if_missing(save_dir)
        # nusc = nusc_dataset.get_nuscenes_db()

        # create a Database object for nuScenes
        nusc = NuScenes(version=version, dataroot=data_root, verbose=True)

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
                                tracking_nuscenes=True)
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
                                         keyframe for which objects are detected.
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
                                         identify an object track across samples.
        "tracking_name":  <str>       -- The predicted class for this 
                                         sample_result, e.g. car, pedestrian. 
                                         Note that the tracking_name cannot 
                                         change throughout a track.
        "tracking_score": <float>     -- Object prediction score between 0 and 1
                                         for the class identified by 
                                         tracking_name. We average over frame 
                                         level scores to compute the track level 
                                         score. The score is used to determine 
                                         positive and negative tracks via 
                                         thresholding.
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
