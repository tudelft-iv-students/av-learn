# This method is implemented on top of
# https://github.com/eddyhkchiu/mahalanobis_3d_multi_object_tracking
# tracking open source code base.

import os
import copy
import glob
import glob2
import numpy as np
import colorsys
from scipy.spatial import ConvexHull
from numba import jit

# Nuscenes classes for which the tracklets are calculated
NUSCENES_TRACKING_CLASSES = [
    'bicycle',
    'bus',
    'car',
    'motorcycle',
    'pedestrian',
    'trailer',
    'truck'
]


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


@jit
def poly_area(x: np.array, y: np.array) -> float:
    """
    Calculates the area of a polygon given a set of (x, y) coordinates
    :param x: an array of the x coordinates
    :param y: an array of the y coordinates
    :returns: the area of the polygon defined by these points
    """
    return 0.5*np.abs(np.dot(x, np.roll(y, 1))-np.dot(y, np.roll(x, 1)))


@jit
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


@jit
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


@jit
def roty(t):
    """
    Performs a rotation about the y-axis
    """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])


@jit
def rotz(t):
    """
    Performs a rotation about the y-axis
    """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])


def mkdir_if_missing(input_path: str, warning: bool = True, debug: bool = True):
    """
    Creates a directory if not existing. If the input is a path of file, then 
    creates the parent directory of this file. If the he root directory does not 
    exists for the input, then creates all the root directories recursively 
    until the parent directory of input exists.
    :param input_path: the input path
    """
    good_path = safe_path(input_path, warning=warning, debug=debug)
    if debug:
        assert is_path_exists_or_creatable(
            good_path), 'input path is not valid or creatable: %s' % good_path
    dirname, _, _ = fileparts(good_path)
    if not is_path_exists(dirname):
        mkdir_if_missing(dirname)
    if isfolder(good_path) and not is_path_exists(good_path):
        os.mkdir(good_path)


def safe_path(input_path: str, warning: bool = True, debug: bool = True) -> str:
    """
    Converts a oath to a valid OS format
    :param input_path: the input path
    :returns: a valid OS path
    """
    if debug:
        assert isstring(input_path), 'path is not a string: %s' % input_path
    safe_data = copy.copy(input_path)
    safe_data = os.path.normpath(safe_data)
    return safe_data


def isstring(input: any) -> bool:
    """
    Checks if the given input is a string
    :param input: the input 
    :returns: boolean
    """
    return isinstance(input, str)


def islist(input: any) -> bool:
    """
    Checks if the given input is a list
    :param input: the input 
    :returns: boolean
    """
    return isinstance(input, list)


def islogical(input: any) -> bool:
    """
    Checks if the given input is a boolean
    :param input: the input 
    :returns: boolean
    """
    return isinstance(input, bool)


def isnparray(input: any) -> bool:
    """
    Checks if the given input is an np.array
    :param input: the input 
    :returns: boolean
    """
    return isinstance(input, np.ndarray)


def isinteger(input: any) -> bool:
    """
    Checks if the given input is an integer
    :param input: the input 
    :returns: boolean
    """
    if isnparray(input):
        return False
    try:
        return isinstance(input, int) or int(input) == input
    except (TypeError, ValueError):
        return False


def is_path_valid(pathname: str) -> bool:
    """
    Checks if a path is valid
    :param pathname: the input path
    :returns: boolean
    """
    try:
        if not isstring(pathname) or not pathname:
            return False
    except TypeError:
        return False
    else:
        return True


def is_path_creatable(pathname: str) -> bool:
    """
    Checks if any previous level of parent folder exists
    :param pathname: the input path
    :returns: boolean
    """
    if not is_path_valid(pathname):
        return False
    pathname = os.path.normpath(pathname)
    pathname = os.path.dirname(os.path.abspath(pathname))

    # recursively to find the previous level of parent folder existing
    while not is_path_exists(pathname):
        pathname_new = os.path.dirname(os.path.abspath(pathname))
        if pathname_new == pathname:
            return False
        pathname = pathname_new
    return os.access(pathname, os.W_OK)


def is_path_exists(pathname: str) -> bool:
    """
    Checks if a path exists
    :param pathname: the input path
    :returns: boolean
    """
    try:
        return is_path_valid(pathname) and os.path.exists(pathname)
    except OSError:
        return False


def is_path_exists_or_creatable(pathname: str) -> bool:
    """
    Checks if a path exists and is creatable
    :param pathname: the input path
    :returns: boolean
    """
    try:
        return is_path_exists(pathname) or is_path_creatable(pathname)
    except OSError:
        return False


def isfolder(pathname: str) -> bool:
    """
    Checks if a path is a folder and not a file
    :param pathname: the given path
    :returns: boolean
    """
    if is_path_valid(pathname):
        pathname = os.path.normpath(pathname)
        if pathname == './':
            return True
        name = os.path.splitext(os.path.basename(pathname))[0]
        ext = os.path.splitext(pathname)[1]
        return len(name) > 0 and len(ext) == 0
    else:
        return False


def fileparts(input_path: str, warning: bool = True, debug: bool = True) -> tuple:
    """
    Returns a tuple, containing the (directory, filename, extension) info
    :param input_path: the given path
    :returns: a (directory, filename, extension) tuple
    """
    good_path = safe_path(input_path, debug=debug)
    if len(good_path) == 0:
        return ('', '', '')
    if good_path[-1] == '/':
        if len(good_path) > 1:
            return (good_path[:-1], '', '')  # ignore the final '/'
        else:
            # ignore the final '/'
            return (good_path, '', '')

    directory = os.path.dirname(os.path.abspath(good_path))
    filename = os.path.splitext(os.path.basename(good_path))[0]
    ext = os.path.splitext(good_path)[1]
    return (directory, filename, ext)


def load_txt_file(file_path: str,  debug: bool = True) -> tuple:
    """
    Loads data or string from txt file
    :param file_path: the given path
    """
    file_path = safe_path(file_path)
    if debug:
        assert is_path_exists(
            file_path), 'text file is not existing at path: %s!' % file_path
    with open(file_path, 'r') as file:
        data = file.read().splitlines()
    num_lines = len(data)
    file.close()
    return data, num_lines


def load_list_from_folder(folder_path: str, ext_filter: str = None,
                          depth: int = 1, recursive: bool = False,
                          sort: bool = True, save_path: str = None,
                          debug: bool = True) -> tuple:
    """
    Given a system path loads a list of all files and folders
    :param folder_path: the root to search
    :param ext_filter: a string to represent the extension of files interested
    :param depth: maximum depth of folder to search, when it's None, all levels
                of folders will be searched
    :param recursive: controlls if only current or all levels will be returned
    :returns: a tuple containing the full list of elements and their count.
    """
    folder_path = safe_path(folder_path)
    if debug:
        assert isfolder(
            folder_path), 'input folder path is not correct: %s' % folder_path
    if not is_path_exists(folder_path):
        print('the input folder does not exist\n')
        return [], 0
    if debug:
        assert islogical(
            recursive), 'recursive should be a logical variable: {}' \
            .format(recursive)
        assert depth is None or (
            isinteger(depth) and depth >= 1), 'input depth is not correct {}' \
            .format(depth)
        assert ext_filter is None or (islist(ext_filter) and all(isstring(
            ext_tmp) for ext_tmp in ext_filter)) or isstring(ext_filter), \
            'extension filter is not correct'
    if isstring(ext_filter):
        # convert to a list
        ext_filter = [ext_filter]
    # zxc

    fulllist = list()
    if depth is None:        # find all files recursively
        recursive = True
        wildcard_prefix = '**'
        if ext_filter is not None:
            for ext_tmp in ext_filter:
                # wildcard = os.path.join(wildcard_prefix, '*' +
                # string2ext_filter(ext_tmp))
                wildcard = os.path.join(wildcard_prefix, '*' + ext_tmp)
                curlist = glob2.glob(os.path.join(folder_path, wildcard))
                if sort:
                    curlist = sorted(curlist)
                fulllist += curlist
        else:
            wildcard = wildcard_prefix
            curlist = glob2.glob(os.path.join(folder_path, wildcard))
            if sort:
                curlist = sorted(curlist)
            fulllist += curlist
    else:                    # find files based on depth and recursive flag
        wildcard_prefix = '*'
        for index in range(depth-1):
            wildcard_prefix = os.path.join(wildcard_prefix, '*')
        if ext_filter is not None:
            for ext_tmp in ext_filter:
                # wildcard = wildcard_prefix + string2ext_filter(ext_tmp)
                wildcard = wildcard_prefix + ext_tmp
                curlist = glob.glob(os.path.join(folder_path, wildcard))
                if sort:
                    curlist = sorted(curlist)
                fulllist += curlist
            # zxc
        else:
            wildcard = wildcard_prefix
            curlist = glob.glob(os.path.join(folder_path, wildcard))
            # print(curlist)
            if sort:
                curlist = sorted(curlist)
            fulllist += curlist
        if recursive and depth > 1:
            newlist, _ = load_list_from_folder(
                folder_path=folder_path, ext_filter=ext_filter, depth=depth-1,
                recursive=True)
            fulllist += newlist

    fulllist = [os.path.normpath(path_tmp) for path_tmp in fulllist]
    num_elem = len(fulllist)

    # save list to a path
    if save_path is not None:
        save_path = safe_path(save_path)
        if debug:
            assert is_path_exists_or_creatable(
                save_path), 'the file cannot be created'
        with open(save_path, 'w') as file:
            for item in fulllist:
                file.write('%s\n' % item)
        file.close()

    return fulllist, num_elem

# visualizatioz


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
