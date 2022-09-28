# This method is implemented on top of
# https://github.com/eddyhkchiu/mahalanobis_3d_multi_object_tracking
# tracking open source code base.


import numpy as np


class CovarianceBase(object):
    """
    Defines Kalman Filter covariance matrices configurations for an AB3DMOT 
    baseline.
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

        # default from baseline code
        self.P[self.num_observations:, self.num_observations:] *= 1000.
        self.P *= 10.
        self.Q[self.num_observations:, self.num_observations:] *= 0.01


# model settings
covariance = CovarianceBase()
