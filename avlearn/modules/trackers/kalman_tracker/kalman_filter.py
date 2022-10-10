# This module is based on the Kalman filter implementation by filterpy:
# https://filterpy.readthedocs.io/en/latest/

from copy import deepcopy

import numpy as np
import numpy.linalg as linalg


class KalmanFilter(object):
    """
    Kalman filter class for av-learn. 
    """

    def __init__(self,
                 F: np.ndarray = None,
                 B: np.ndarray = None,
                 H: np.ndarray = None,
                 Q: np.ndarray = None,
                 R: np.ndarray = None,
                 P: np.ndarray = None,
                 x0: np.ndarray = None):
        """
        Loads initial Kalman filter parameters and gives default values for 
        unspecified parameters.
        :param F: (optional) state transition matrix
        :param B: (optional) control transition matrix
        :param H: (optional) measurement function
        :param Q: (optional) process uncertainty
        :param R: (optional) state uncertainty covariance
        :param P: (optional) state covariance
        :param x0: (optional) initial state
        """
        if (F is None):
            raise ValueError("Define state transition matrix F.")
        if (H is None):
            raise ValueError("Define easurement function H.")

        # number of variables in the state
        self.dim_x = F.shape[0]
        # number of measurements
        self.dim_z = H.shape[0]

        # initialize state
        self.x = np.zeros((self.dim_x, 1)) if x0 is None else x0
        self.P = np.eye(self.dim_x) if P is None else P
        self.Q = np.eye(self.dim_x) if Q is None else Q
        self.B = 0 if B is None else B
        self.F = np.eye(self.dim_x) if F is None else F
        self.H = np.zeros((self.dim_z, self.dim_x)) if H is None else H
        self.R = np.eye(self.dim_z) if R is None else R
        # fading memory control
        self._alpha_sq = 1.

        # for storing future measurements
        self.z = np.array([[None]*self.dim_z]).T

        # initialize Kalman gain, residual and system uncertainty
        self.K = np.zeros((self.dim_x, self.dim_z))  # kalman gain
        self.y = np.zeros((self.dim_z, 1))
        self.S = np.zeros((self.dim_z, self.dim_z))  # system uncertainty

        # identity matrix.
        self._I = np.eye(self.dim_x)

        # for storing x and P after predict() is called
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

        # for storing x and P after update() is called
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

    def predict(self,
                u: np.ndarray = None,
                B: np.ndarray = None,
                F: np.ndarray = None,
                Q: np.ndarray = None):
        """
        Uses the Kalman filter state propagation equations to predict next 
        state.
        :param u: (optional) control vector
        :param B: (optional) control transition matrix
        :param F: (optional) state transition matrix
        :param Q: (optional) process noise matrix
        """

        # for any undefined matrix use already set values
        if B is None:
            B = self.B
        if F is None:
            F = self.F
        if Q is None:
            Q = self.Q

        # x = Fx + Bu
        if B != None and u != None:
            self.x = np.dot(F, self.x) + np.dot(B, u)
        else:
            self.x = np.dot(F, self.x)
        # P = FPF' + Q
        self.P = self._alpha_sq * np.dot(np.dot(F, self.P), F.T) + Q

        # save prior state
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

    def update(self,
               z: np.ndarray,
               R: np.ndarray = None,
               H: np.ndarray = None):
        """
        Adds a new measurement to the Kalman filter and updates its state.
        :param z: measurement for this update
        :param R: (optional) state uncertainty covariance, used to overwite 
                measurement noise
        :param H: (optional) measurement function
        """

        # in case z is None, no calculation takes place, but the posterios are
        # updated from the priors and self.z is set to None
        if z is None:
            self.z = np.array([[None]*self.dim_z]).T
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            self.y = np.zeros((self.dim_z, 1))
            return
        # for undefined R or H use already set values
        if R is None:
            R = self.R
        if H is None:
            H = self.H

        # ensure that z is a (dim_z, 1) shaped vector
        z = reshape_z(z, self.dim_z, self.x.ndim)

        # y = z - Hx - error (residual) between measurement and prediction
        self.y = z - np.dot(H, self.x)

        # common subexpression
        PHT = np.dot(self.P, H.T)

        # S = HPH' + R - project system uncertainty into measurement space
        self.S = R + np.dot(H, PHT)
        # K = PH'inv(S) - map system uncertainty into Kalman gain
        self.K = np.dot(PHT, linalg.inv(self.S))

        # x = x + Ky - predict new x with residual scaled by the Kalman gain
        self.x = self.x + np.dot(self.K, self.y)

        # P = (I-KH)P(I-KH)' + KRK'
        I_KH = self._I - np.dot(self.K, H)
        self.P = np.dot(np.dot(I_KH, self.P), I_KH.T) + \
            np.dot(np.dot(self.K, R), self.K.T)

        # save measurement and posterior state
        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()


def reshape_z(z: np.ndarray, dim_z: int, ndim: int) -> np.ndarray:
    """
    Ensures that measurement z is of (dim_z, 1) dimensions
    :param z: measurement for this update
    :param dim_z: number of measurements
    :param ndim: dimensionality of x state
    """

    z = np.atleast_2d(z)
    if z.shape[1] == dim_z:
        z = z.T
    if z.shape != (dim_z, 1):
        raise ValueError(
            'z must be convertible to shape ({}, 1)'.format(dim_z))
    if ndim == 1:
        z = z[:, 0]
    if ndim == 0:
        z = z[0, 0]

    return z
