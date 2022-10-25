# This module is implemented on top of https://github.com/tianweiy/CenterPoint
# open source code base.

import numpy as np
from scipy.optimize import linear_sum_assignment


def linear_assignment(X: np.ndarray) -> np.ndarray:
    """Solve the linear assignment problem.

    This method uses the scipy.linear_sum_assignment module, but converts its
    output format to match the format of the sklearn.utils.linear_assignment_ 
    module which was deprecated.

    :param X: the cost matrix of the bipartite graph
    """
    row_ind, col_ind = linear_sum_assignment(X)
    return np.array(list(zip(row_ind, col_ind)))


def greedy_assignment(dist: np.array) -> np.ndarray:
    """
    Finds the one-to-one matchings, using greedy smallest distance criterion
    :param distance_matrix: N x M distance matrix between detections and 
                    trackers
    :returns: an array with all the matched indices between detections and 
            trackers
    """
    matched_indices = []
    if dist.shape[1] == 0:
        return np.array(matched_indices, np.int32).reshape(-1, 2)
    for i in range(dist.shape[0]):
        j = dist[i].argmin()
        if dist[i][j] < 1e16:
            dist[:, j] = 1e18
            matched_indices.append([i, j])
    return np.array(matched_indices, np.int32).reshape(-1, 2)
