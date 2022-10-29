"""This module implements Node, Lane, and Graph classes for the representation
of the centerlines map of nuScenes."""
import numpy as np
from typing import Dict, List
import json


class Node:
    """Represents a node in nuScenes."""

    def __init__(self, id: int, x: str, y: str) -> None:
        """Represents a node in nuScenes.

        :param id: Arbitrary node id.
        :param x: X coordinate.
        :param y: Y coordinate.
        """
        self.id = id
        self.x = x
        self.y = y

    def __str__(self):
        return str(vars(self))


class Lane:
    """Represents a lane in nuScenes."""

    def __init__(
        self,
        token: str,
        predecessors: List[str],
        successors: List[str],
        centerline: np.ndarray,
        **kwargs
    ) -> None:
        """Represents a lane in nuScenes.

        :param token: Lane token from nuScenes dataset.
        :param predecessors: The tokens of the lanes that come after this one.
        :param successors: The tokens of the lanes that come before this one.
        :param centerline: The coordinates of the lane's center line.
        """
        self.token = token
        self.predecessors = predecessors
        self.successors = successors
        self.centerline = centerline

    def __str__(self):
        return str(vars(self))


class Graph:
    """Represents a graph of nuScenes' centerlines."""

    def __init__(self, json_file: str) -> None:
        """Represents a graph of nuScenes' centerlines.

        :param json_file: Map file created by `create_nuscenes_graph.py`.

        Attributes
            :nodes: Dictionary of the graph's nodes
            :lanes: Dictionary of the graph's lanes
        """
        with open(json_file, "r") as f:
            self.graph = json.load(f)

        self.nodes = self.__get_nodes()
        self.lanes = self.__get_lanes()

    def __get_nodes(self) -> Dict[str, Node]:
        nodes = {}
        for node_id, node_info in self.graph["nodes"].items():
            nodes[node_id] = Node(**node_info)

        return nodes

    def __get_centerline_coords(
            self, centerline_nodes: List[str]) -> np.ndarray:
        centerline = []
        for node_id in centerline_nodes:
            node = self.nodes[node_id]
            centerline.append([float(node.x), float(node.y)])

        return np.array(centerline)

    def __get_lanes(self) -> Dict[str, Lane]:
        lanes = {}
        for lane_token, lane_info in self.graph["lanes"].items():
            centerline = self.__get_centerline_coords(
                lane_info["centerline_nodes"])
            lanes[lane_token] = Lane(**lane_info, centerline=centerline)

        return lanes
