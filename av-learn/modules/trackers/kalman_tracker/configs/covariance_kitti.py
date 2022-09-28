# This method is implemented on top of
# https://github.com/eddyhkchiu/mahalanobis_3d_multi_object_tracking
# tracking open source code base.


import numpy as np


class CovarianceKitti(object):
    """
    Defines Kalman Filter covariance matrices configurations for the Kitti 
    dataset.
    """

    def __init__(self):
        """
        Initializes a covariance object.
        """
        self.num_states = 10
        self.num_observations = 7
        # state transition matrix
        self.F = np.array([[1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                          [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                          [0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
                          [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
        # measurement function
        self.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]])
        self.P = np.eye(self.num_states)
        self.Q = np.eye(self.num_states)
        self.R = np.eye(self.num_observations)

        # from kitti stats
        self.P[0, 0] = 0.01969623
        self.P[1, 1] = 0.01179107
        self.P[2, 2] = 0.04189842
        self.P[3, 3] = 0.52534431
        self.P[4, 4] = 0.11816206
        self.P[5, 5] = 0.00983173
        self.P[6, 6] = 0.01602004
        self.P[7, 7] = 0.01334779
        self.P[8, 8] = 0.00389245
        self.P[9, 9] = 0.01837525

        self.Q[0, 0] = 2.94827444e-03
        self.Q[1, 1] = 2.18784125e-03
        self.Q[2, 2] = 6.85044585e-03
        self.Q[3, 3] = 1.10964054e-01
        self.Q[4, 4] = 0
        self.Q[5, 5] = 0
        self.Q[6, 6] = 0
        self.Q[7, 7] = 2.94827444e-03
        self.Q[8, 8] = 2.18784125e-03
        self.Q[9, 9] = 6.85044585e-03

        self.R[0, 0] = 0.01969623
        self.R[1, 1] = 0.01179107
        self.R[2, 2] = 0.04189842
        self.R[3, 3] = 0.52534431
        self.R[4, 4] = 0.11816206
        self.R[5, 5] = 0.00983173
        self.R[6, 6] = 0.01602004


# model settings
covariance = CovarianceKitti()
