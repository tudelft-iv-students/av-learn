# This module is implemented on top of
# https://github.com/eddyhkchiu/mahalanobis_3d_multi_object_tracking
# and https://github.com/xinshuoweng/AB3DMOT tracking open source code bases.

from __future__ import print_function

import copy

import numpy as np
from mmcv import Config

from kalman_filter import KalmanFilter
from utils import (diff_orientation_correction, greedy_match, iou3d,
                   linear_assignment, roty)


class AB3DMOT(object):
    """
    This class represents the internel state of an overall AB3DMOT tracker,
    that manages all box trackers for one of the predicted tracking classes.
    """

    def __init__(self,
                 cfg_path: str,
                 max_age: int = 2,
                 min_hits: int = 3,
                 tracking_name: str = 'car',
                 tracking_nuscenes: bool = False):
        """      
        Initializes an AB3DMOT tracker for the given class to be tracked. 
        :param cfg_path: path to configuration file for Kalman filter covariance
                     matrices
        :param max_age: the maximum number frames allowed for a tracker to have 
                        no matches before being deactivated
        :param min_hits: the minimum number matches allowed for a tracker before 
                    being deactivated
        :param tracking_name: predicted class for this sample_result, e.g. car,
                        pedestrian
        :param tracking nuscenes: determines whether the nuscenes dataset is 
                        used
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []      # list of active trackers
        self.frame_count = 0
        # before reorder: [h, w, l, x, y, z, rot_y]
        # after reorder:  [x, y, z, rot_y, l, w, h]
        self.reorder = [3, 4, 5, 6, 2, 1, 0]
        self.reorder_back = [6, 5, 4, 0, 1, 2, 3]
        self.cfg_path = cfg_path
        self.tracking_name = tracking_name
        self.tracking_nuscenes = tracking_nuscenes

    def update(self, dets_all: dict, match_distance: str,
               match_threshold: float, match_algorithm: str) -> np.ndarray:
        """
        Updates the tracklets given an array of detections in the current frame. 
        This method is called once for each frame, even for frames without any 
        associated detections
        :param dets_all: dictionary containing a numpy array of detections in 
                    the format [[x,y,z,theta,l,w,h],[x,y,z,theta,l,w,h],...] 
                    along with the scores information for each detection
        :param match_distance: defines the mathcing distance used
        :param match_threshold: defines the matching threshold used in the 
                    hungarian algorithm
        :param match_algorithm: defines the matching algorithm used
        :returns: a similar array, where the last column is the tracked object 
                ID.
        """
        dets, scores = dets_all['dets'], dets_all['scores']
        # dets: N x 7, float numpy array, N: # of detections
        dets = dets[:, self.reorder]

        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 7))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            # Advances the state vector for each active KalmanBoxTracker
            # and returns the predicted bounding box estimate.
            pos = self.trackers[t].predict().reshape((-1, 1))
            trk[:] = [pos[0], pos[1], pos[2], pos[3], pos[4], pos[5], pos[6]]
            # if any of the predicted values is NaN append the tracker to
            # deletion list
            if (np.any(np.isnan(pos))):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        # remove trackers that are no longer active
        for t in reversed(to_del):
            self.trackers.pop(t)

        # project 3d bounding boxes to image plane for all detections in a frame
        dets_8corner = [convert_3dbox_to_8corner(
            det_tmp, match_distance == 'iou' and self.tracking_nuscenes)
            for det_tmp in dets]

        if len(dets_8corner) > 0:
            dets_8corner = np.stack(dets_8corner, axis=0)
        else:
            dets_8corner = []

         # project 3d bounding boxes of trackers to image plane
        trks_8corner = [convert_3dbox_to_8corner(
            trk_tmp, match_distance == 'iou' and self.tracking_nuscenes)
            for trk_tmp in trks]

        # update system uncertainty for trackers using Kalman filter update
        # equation
        trks_S = [np.matmul(np.matmul(tracker.kf.H, tracker.kf.P),
                            tracker.kf.H.T) + tracker.kf.R
                  for tracker in self.trackers]

        if len(trks_8corner) > 0:
            trks_8corner = np.stack(trks_8corner, axis=0)
            trks_S = np.stack(trks_S, axis=0)

        # use either iou or mahalanobis distance to associate bounding box
        # detections to existing trackers
        if match_distance == 'iou':
            matched, unmatched_dets, unmatched_trks \
                = associate_detections(dets_8corner, trks_8corner,
                                       iou_threshold=match_threshold,
                                       match_algorithm=match_algorithm)
        else:
            matched, unmatched_dets, unmatched_trks \
                = associate_detections(dets_8corner, trks_8corner,
                                       use_mahalanobis=True, dets=dets,
                                       trks=trks, trks_S=trks_S,
                                       mahalanobis_threshold=match_threshold,
                                       match_algorithm=match_algorithm)

        # update matched trackers with assigned detections using Kalman filter
        # updates
        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trks:
                d = matched[np.where(matched[:, 1] == t)[
                    0], 0]     # a list of index
                trk.update(dets[d, :][0], scores[d, :][0])
                detection_score = scores[d, :][0][-1]
                trk.track_score = detection_score

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            detection_score = scores[i][-1]
            track_score = detection_score
            trk = KalmanBoxTracker(self.cfg_path, dets[i, :], scores[i, :],
                                   track_score, self.tracking_name)
            # append new KalmanBoxTrackers to active trackers
            self.trackers.append(trk)

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()      # bbox location
            d = d[self.reorder_back]

            if ((trk.hits >= self.min_hits or self.frame_count <= self.min_hits)
               and (trk.time_since_update < self.max_age)):
                # +1 as MOT benchmark requires positive
                ret.append(np.concatenate(
                    (d, [trk.id+1], trk.scores[:-1],
                     [trk.track_score])).reshape(1, -1))

            i -= 1
            # remove inactive tracklet
            if (trk.time_since_update >= self.max_age):
                self.trackers.pop(i)
        if (len(ret) > 0):
            # x, y, z, theta, l, w, h, ID, score, confidence
            return np.concatenate(ret)
        return np.empty((0, 15 + 7))


class KalmanBoxTracker(object):
    """
    This class represents the internel state of individual tracked objects 
    observed as bbox.
    """
    count = 0

    def __init__(self,
                 cfg_path: str,
                 bbox3D: np.ndarray,
                 score: float,
                 track_score: float = None,
                 tracking_name: str = 'car'):
        """
        Initialises a tracker using initial bounding box.
        :param cfg_path: path to configuration file for Kalman filter covariance
                     matrices
        :param bbox3D: the observed 3D box 
                    ([x, y, z, rot_y, l, w, h, x_dot, y_dot, z_dot])
        :param score: the detection score for the box
        :param track_score: the tracking score for this detector
        :param tracking_name: predicted class for this sample_result, e.g. car,
                        pedestrian
        """
        # define constant velocity model
        cfg = Config.fromfile(cfg_path)
        covariance = cfg.covariance

        if "nuscenes" not in cfg_path:
            self.kf = KalmanFilter(F=covariance.F, H=covariance.H,
                                   P=covariance.P,
                                   Q=covariance.Q,
                                   R=covariance.R)
        else:
            self.kf = KalmanFilter(F=covariance.F, H=covariance.H,
                                   P=covariance.P[tracking_name],
                                   Q=covariance.Q[tracking_name],
                                   R=covariance.R[tracking_name])

        self.kf.x[:7] = bbox3D.reshape((7, 1))

        # time in frames since last update of the tracker
        self.time_since_update = 0
        # unique id for the tracker
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        # number of total hits including the first detection
        self.hits = 1
        # number of continuing hits considering the first detection
        self.hit_streak = 1
        self.first_continuing_hit = 1
        self.still_first = True
        # age of tracker in number of frames
        self.age = 0
        # detection score
        self.scores = score
        # tracking score
        self.track_score = track_score
        self.tracking_name = tracking_name

    def update(self, bbox3D: np.ndarray, score: float):
        """ 
        Updates the state vector with observed bbox, using Kalman update.
        :param bbox3D: the observed 3D box for this frame 
                    ([x, y, z, rot_y, l, w, h, x_dot, y_dot, z_dot])
        :param score: the detection score for the bounding box
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        # number of continuing hits
        self.hit_streak += 1
        # number of continuing hits in the fist time
        if self.still_first:
            self.first_continuing_hit += 1

        # orientation correction
        if self.kf.x[3] >= np.pi:
            self.kf.x[3] -= np.pi * 2
        if self.kf.x[3] < -np.pi:
            self.kf.x[3] += np.pi * 2

        new_theta = bbox3D[3]
        if new_theta >= np.pi:
            new_theta -= np.pi * 2
        if new_theta < -np.pi:
            new_theta += np.pi * 2
        bbox3D[3] = new_theta

        predicted_theta = self.kf.x[3]
        # if the angle of two theta is not acute angle
        if abs(new_theta - predicted_theta) > np.pi / 2.0 and \
                abs(new_theta - predicted_theta) < np.pi * 3 / 2.0:
            self.kf.x[3] += np.pi
            if self.kf.x[3] > np.pi:
                self.kf.x[3] -= np.pi * 2
            if self.kf.x[3] < -np.pi:
                self.kf.x[3] += np.pi * 2

        # now the angle is acute: < 90 or > 270, convert the case of
        # > 270 to < 90
        if abs(new_theta - self.kf.x[3]) >= np.pi * 3 / 2.0:
            if new_theta > 0:
                self.kf.x[3] += np.pi * 2
            else:
                self.kf.x[3] -= np.pi * 2

        # Kalman filter update
        self.kf.update(bbox3D)

        # orientation correction
        if self.kf.x[3] >= np.pi:
            self.kf.x[3] -= np.pi * 2
        if self.kf.x[3] < -np.pi:
            self.kf.x[3] += np.pi * 2
        self.score = score

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box 
        estimate.
        """
        self.kf.predict()       # use predict function of Kalman filter
        if self.kf.x[3] >= np.pi:
            self.kf.x[3] -= np.pi * 2
        if self.kf.x[3] < -np.pi:
            self.kf.x[3] += np.pi * 2

        # increase age by 1 frame
        self.age += 1
        if (self.time_since_update > 0):
            self.hit_streak = 0
            self.still_first = False
        self.time_since_update += 1
        self.history.append(self.kf.x)
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.kf.x[:7].reshape((7, ))


def associate_detections(detections: np.ndarray,
                         trackers: np.ndarray,
                         iou_threshold: float = 0.1,
                         use_mahalanobis: bool = False,
                         dets: np.ndarray = None,
                         trks: np.ndarray = None,
                         trks_S=None,
                         mahalanobis_threshold: float = 0.1,
                         match_algorithm: str = "hungarian") -> tuple:
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    :param detections: N x 8 x 3 array of the N detections associated with 
                    current frame, where each detection is represneted by 8 
                    corners in a plane.
    :param trackers:  M x 8 x 3 array of the M trackers associated with 
                    current frame
    :param iou_threshold: defines the matching threshold used in the 
                        hungarian algorithm (Default: 0.1)
    :param use_mahalanobis: defines whether mahalanobis distance will be used
    :param dets: N x 7 with the detections [[x,y,z,theta,l,w,h],...]
    :param trks: M x 7 with the trackers [[x,y,z,theta,l,w,h],...]
    :param trks_S: system uncertainty for trackers (N x 7 x 7)
    :param mahalanobis_threshold: defines the matching threshold used in the 
                        hungarian algorithm (Default: 0.1)
    :param match_algorithm: efines the matching algorithm used
                        (Default: "hungarian")
    :returns: tuple of 3 lists containing matches, unmatched_detections and u
            nmatched_trackers
    """
    if (len(trackers) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), \
            np.empty((0, 8, 3), dtype=int)
    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
    distance_matrix = np.zeros(
        (len(detections), len(trackers)), dtype=np.float32)

    if use_mahalanobis:
        assert (dets is not None)
        assert (trks is not None)
        assert (trks_S is not None)

    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            if use_mahalanobis:
                S_inv = np.linalg.inv(trks_S[t])  # 7 x 7
                diff = np.expand_dims(dets[d] - trks[t], axis=1)  # 7 x 1
                # manual reversed angle by 180 when diff > 90 or < -90 degree
                corrected_angle_diff = diff_orientation_correction(
                    dets[d][3], trks[t][3])
                diff[3] = corrected_angle_diff
                # create mahalanobis distance matrix
                distance_matrix[d, t] = np.sqrt(
                    np.matmul(np.matmul(diff.T, S_inv), diff)[0][0])
            else:
                # create iou distance matrix
                iou_matrix[d, t] = iou3d(det, trk)[0]
                distance_matrix = -iou_matrix

    if match_algorithm == 'greedy':
        # greedy algorithm to solve linear matching problem
        matched_indices = greedy_match(distance_matrix)
    elif match_algorithm == 'hungarian_thres':
        if use_mahalanobis:
            to_max_mask = distance_matrix > mahalanobis_threshold
            distance_matrix[to_max_mask] = mahalanobis_threshold + 1
        else:
            to_max_mask = iou_matrix < iou_threshold
            distance_matrix[to_max_mask] = 0
            iou_matrix[to_max_mask] = 0
        # houngarian algorithm to solve linear matching problem
        matched_indices = linear_assignment(distance_matrix)
    else:
        # houngarian algorithm to solve linear matching problem
        matched_indices = linear_assignment(distance_matrix)

    # use matched_indices to check for unmatched detections and trackers
    unmatched_detections = []
    for d, det in enumerate(detections):
        if (d not in matched_indices[:, 0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if len(matched_indices) == 0 or (t not in matched_indices[:, 1]):
            unmatched_trackers.append(t)

    # filter out matches with low IOU
    matches = []
    for m in matched_indices:
        match = True
        if use_mahalanobis:
            if distance_matrix[m[0], m[1]] > mahalanobis_threshold:
                match = False
        else:
            if (iou_matrix[m[0], m[1]] < iou_threshold):
                match = False
        if not match:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if (len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(
        unmatched_trackers)


def convert_3dbox_to_8corner(bbox3d_input: np.ndarray,
                             nuscenes_to_kitti: bool = False) -> np.ndarray:
    """
    Takes a bounding box object and projects the it into the image plane, 
    using its 8 corners.
    :param bbox3d_input: the input 3D bounding box: [x,y,z,theta,l,w,h]
    :param nuscenes_to_kitti: determines if transformation to kitti format is 
                              needed (Default: False)
    :returns: an (8,3) array in rect camera coord.
    """
    # compute rotational matrix around yaw axis
    bbox3d = copy.copy(bbox3d_input)

    if nuscenes_to_kitti:
        # transform to kitti format first
        bbox3d_nuscenes = copy.copy(bbox3d)
        # kitti:    [x,  y,  z,  a, l, w, h]
        # nuscenes: [y, -z, -x, -a, w, l, h]
        bbox3d[0] = bbox3d_nuscenes[1]
        bbox3d[1] = -bbox3d_nuscenes[2]
        bbox3d[2] = -bbox3d_nuscenes[0]
        bbox3d[3] = -bbox3d_nuscenes[3]
        bbox3d[4] = bbox3d_nuscenes[5]
        bbox3d[5] = bbox3d_nuscenes[4]

    R = roty(bbox3d[3])

    # 3d bounding box dimensions
    l = bbox3d[4]
    w = bbox3d[5]
    h = bbox3d[6]

    # 3d bounding box corners
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]

    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0, :] = corners_3d[0, :] + bbox3d[0]
    corners_3d[1, :] = corners_3d[1, :] + bbox3d[1]
    corners_3d[2, :] = corners_3d[2, :] + bbox3d[2]

    return np.transpose(corners_3d)
