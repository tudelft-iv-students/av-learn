# This module is implemented on top of
# https://github.com/eddyhkchiu/mahalanobis_3d_multi_object_tracking
# tracking open source code base.

import colorsys

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull


def angle_in_range(angle):
    """
    Checks whether an angle is > 90 or < -90 and rotates it if thats the case
    :param angle: the input angle
    :returns: an anble <= 90 and >= -90
    """
    if angle > np.pi:
        angle -= 2 * np.pi
    if angle < -np.pi:
        angle += 2 * np.pi
    return angle


def diff_orientation_correction(det, trk):
    """
    Calculates the angle difference between detecion and tracking boxes.
    :param det: the rotation of the detection box
    :param trk: the rotation of the detection box
    : returns the angle diff = det - trk
    """
    diff = det - trk
    # if angle diff > 90 or < -90, rotate trk and update the angle diff
    diff = angle_in_range(diff)
    if diff > np.pi / 2:
        diff -= np.pi
    if diff < -np.pi / 2:
        diff += np.pi
    diff = angle_in_range(diff)
    return diff


def greedy_match(distance_matrix: np.ndarray) -> np.ndarray:
    """
    Finds the one-to-one matchings, using greedy smallest distance criterion
    :param distance_matrix: N x M distance matrix between detections and 
                    trackers
    :returns: an array with all the matched indices between detections and 
            trackers
    """
    matched_indices = []

    det_num, track_num = distance_matrix.shape
    distance_1d = distance_matrix.reshape(-1)
    index_1d = np.argsort(distance_1d)
    index_2d = np.stack([index_1d // track_num, index_1d %
                        track_num], axis=1)
    det_to_track = [-1] * det_num
    track_to_det = [-1] * track_num
    for sort_i in range(index_2d.shape[0]):
        detection_id = int(index_2d[sort_i][0])
        tracking_id = int(index_2d[sort_i][1])
        if track_to_det[tracking_id] == -1 and \
                det_to_track[detection_id] == -1:
            track_to_det[tracking_id] = detection_id
            det_to_track[detection_id] = tracking_id
            matched_indices.append([detection_id, tracking_id])

    matched_indices = np.array(matched_indices)
    return matched_indices


def poly_area(x: np.array, y: np.array) -> float:
    """
    Calculates the area of a polygon given a set of (x, y) coordinates
    :param x: an array of the x coordinates
    :param y: an array of the y coordinates
    :returns: the area of the polygon defined by these points
    """
    return 0.5*np.abs(np.dot(x, np.roll(y, 1))-np.dot(y, np.roll(x, 1)))


def box3d_vol(corners: np.ndarray) -> float:
    """
    Calculates the volume of a 3D bounding box
    :param corners:  an (8,3) array of the corners of a 3D box
    :returns: the volume of the 3D box
    """
    a = np.sqrt(np.sum((corners[0, :] - corners[1, :])**2))
    b = np.sqrt(np.sum((corners[1, :] - corners[2, :])**2))
    c = np.sqrt(np.sum((corners[0, :] - corners[4, :])**2))
    return a*b*c


def convex_hull_intersection(p1: list, p2: list) -> list:
    """ 
    Computes area of two convex hull's intersection area.
    :param p1: list of (x,y) tuples of hull vertices.
    :param p2: list of (x,y) tuples of hull vertices.
    :returns: a list of (x,y) for the intersection and its volume
    """
    inter_p = polygon_clip(p1, p2)
    if inter_p is not None:
        hull_inter = ConvexHull(inter_p)
        return inter_p, hull_inter.volume
    else:
        return None, 0.0


def polygon_clip(subjectPolygon: list, clipPolygon: list) -> list:
    """ 
    Clips a polygon with another polygon.
    :param subjectPolygon: a list of (x,y) 2d points (any polygon)
    :param clipPolygon: a list of (x,y) 2d points (convex polygon)
    :returns: a list of (x,y) vertex point for the intersection polygon.
    """
    def inside(p):
        return (cp2[0]-cp1[0])*(p[1]-cp1[1]) > (cp2[1]-cp1[1])*(p[0]-cp1[0])

    def computeIntersection():
        dc = [cp1[0] - cp2[0], cp1[1] - cp2[1]]
        dp = [s[0] - e[0], s[1] - e[1]]
        n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
        n2 = s[0] * e[1] - s[1] * e[0]
        n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
        return [(n1*dp[0] - n2*dc[0]) * n3, (n1*dp[1] - n2*dc[1]) * n3]

    outputList = subjectPolygon
    cp1 = clipPolygon[-1]

    for clipVertex in clipPolygon:
        cp2 = clipVertex
        inputList = outputList
        outputList = []
        s = inputList[-1]

        for subjectVertex in inputList:
            e = subjectVertex
            if inside(e):
                if not inside(s):
                    outputList.append(computeIntersection())
                outputList.append(e)
            elif inside(s):
                outputList.append(computeIntersection())
            s = e
        cp1 = cp2
        if len(outputList) == 0:
            return None
    return (outputList)


def iou3d(corners1: np.ndarray, corners2: np.ndarray) -> tuple:
    """
    Computes the IoU between two 3D detection boxes.
    :param corners1: an (8,3) array of the corners of a 3D box
    :param corners2: an (8,3) array of the corners of a 3D box
    :returns: the 3D bounding box IoU and the BEV 2D bounding box IoU
    """
    # corner points are in counter clockwise order
    rect1 = [(corners1[i, 0], corners1[i, 2]) for i in range(3, -1, -1)]
    rect2 = [(corners2[i, 0], corners2[i, 2]) for i in range(3, -1, -1)]
    area1 = poly_area(np.array(rect1)[:, 0], np.array(rect1)[:, 1])
    area2 = poly_area(np.array(rect2)[:, 0], np.array(rect2)[:, 1])
    inter, inter_area = convex_hull_intersection(rect1, rect2)
    iou_2d = inter_area/(area1+area2-inter_area)
    ymax = min(corners1[0, 1], corners2[0, 1])
    ymin = max(corners1[4, 1], corners2[4, 1])
    inter_vol = inter_area * max(0.0, ymax-ymin)
    vol1 = box3d_vol(corners1)
    vol2 = box3d_vol(corners2)
    iou = inter_vol / (vol1 + vol2 - inter_vol)
    return iou, iou_2d


def roty(t):
    """
    Performs a rotation about the y-axis
    """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])


def rotz(t):
    """
    Performs a rotation about the y-axis
    """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])


def linear_assignment(X: np.ndarray) -> np.ndarray:
    """Solve the linear assignment problem.

    This method uses the scipy.linear_sum_assignment module, but converts its
    output format to match the format of the sklearn.utils.linear_assignment_ 
    module which was deprecated.

    :param X: the cost matrix of the bipartite graph
    """
    row_ind, col_ind = linear_sum_assignment(X)
    return np.array(list(zip(row_ind, col_ind)))


# visualization
def random_colors(N: int, bright: bool = True) -> list:
    """
    Generates random colors. To get visually distinct colors, generate them in 
    HSV space then convert to RGB.
    :param N: number of unique colours needed
    :param bright: controls brightness of produced colours
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / float(N), 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    # random.shuffle(colors)
    return colors
