# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>
# https://github.com/argoai/argoverse-api/blob/master/argoverse/utils/centerline_utils.py

import datetime
from typing import Iterable, List, Sequence, Set, Tuple, Mapping, Dict

import matplotlib.pyplot as plt
import numpy as np


def swap_left_and_right(
    condition: np.ndarray, left_centerline: np.ndarray, right_centerline: np.ndarray
) -> Iterable[np.ndarray]:
    """
    Swap points in left and right centerline according to condition.

    Args:
       condition: Numpy array of shape (N,) of type boolean. Where true, swap the values in the left and
                   right centerlines.
       left_centerline: The left centerline, whose points should be swapped with the right centerline.
       right_centerline: The right centerline.

    Returns:
       left_centerline
       right_centerline
    """

    right_swap_indices = right_centerline[condition]
    left_swap_indices = left_centerline[condition]

    left_centerline[condition] = right_swap_indices
    right_centerline[condition] = left_swap_indices
    return left_centerline, right_centerline


def centerline_to_polygon(
        centerline: np.ndarray, width_scaling_factor: float = 1.0,
        visualize: bool = False) -> np.ndarray:
    """
    Convert a lane centerline polyline into a rough polygon of the lane's area.

    On average, a lane is 3.8 meters in width. Thus, we allow 1.9 m on each side.
    We use this as the length of the hypotenuse of a right triangle, and compute the
    other two legs to find the scaled x and y displacement.

    Args:
       centerline: Numpy array of shape (N,2).
       width_scaling_factor: Multiplier that scales 3.8 meters to get the lane width.
       visualize: Save a figure showing the the output polygon.

    Returns:
       polygon: Numpy array of shape (2N+1,2), with duplicate first and last vertices.
    """
    # eliminate duplicates
    _, inds = np.unique(centerline, axis=0, return_index=True)
    # does not return indices in sorted order
    inds = np.sort(inds)
    centerline = centerline[inds]

    dx = np.gradient(centerline[:, 0])
    dy = np.gradient(centerline[:, 1])

    # compute the normal at each point
    slopes = dy / dx
    inv_slopes = -1.0 / slopes

    thetas = np.arctan(inv_slopes)
    x_disp = 3.8 * width_scaling_factor / 2.0 * np.cos(thetas)
    y_disp = 3.8 * width_scaling_factor / 2.0 * np.sin(thetas)

    displacement = np.hstack([x_disp[:, np.newaxis], y_disp[:, np.newaxis]])
    right_centerline = centerline + displacement
    left_centerline = centerline - displacement

    # right centerline position depends on sign of dx and dy
    subtract_cond1 = np.logical_and(dx > 0, dy < 0)
    subtract_cond2 = np.logical_and(dx > 0, dy > 0)
    subtract_cond = np.logical_or(subtract_cond1, subtract_cond2)
    left_centerline, right_centerline = swap_left_and_right(
        subtract_cond, left_centerline, right_centerline)

    # right centerline also depended on if we added or subtracted y
    neg_disp_cond = displacement[:, 1] > 0
    left_centerline, right_centerline = swap_left_and_right(
        neg_disp_cond, left_centerline, right_centerline)

    if visualize:
        plt.scatter(
            centerline[:, 0],
            centerline[:, 1],
            20, marker=".", color="b")
        plt.scatter(
            right_centerline[:, 0],
            right_centerline[:, 1],
            20, marker=".", color="r")
        plt.scatter(
            left_centerline[:, 0],
            left_centerline[:, 1],
            20, marker=".", color="g")
        fname = datetime.datetime.utcnow().strftime("%Y_%m_%d_%H_%M_%S_%f")
        plt.savefig(f"polygon_unit_tests/{fname}.png")
        plt.close("all")

    # return the polygon
    return convert_lane_boundaries_to_polygon(
        right_centerline, left_centerline)


def convert_lane_boundaries_to_polygon(right_lane_bounds: np.ndarray, left_lane_bounds: np.ndarray) -> np.ndarray:
    """
    Take a left and right lane boundary and make a polygon of the lane segment, closing both ends of the segment.

    These polygons have the last vertex repeated (that is, first vertex == last vertex).

    Args:
       right_lane_bounds: Right lane boundary points. Shape is (N, 2).
       left_lane_bounds: Left lane boundary points.

    Returns:
       polygon: Numpy array of shape (2N+1,2)
    """
    assert right_lane_bounds.shape[0] == left_lane_bounds.shape[0]
    polygon = np.vstack([right_lane_bounds, left_lane_bounds[::-1]])
    polygon = np.vstack([polygon, right_lane_bounds[0]])
    return polygon
