"""This module implements a nuScenes map."""
import json
from pathlib import Path
from typing import Mapping, List

import numpy as np

from .graph import Graph, Lane
from .utils.manhattan_search import \
    find_all_polygon_bboxes_overlapping_query_bbox
from .utils.centerline_utils import centerline_to_polygon

NUSCENES_MAPS = [
    "boston-seaport",
    "singapore-hollandvillage",
    "singapore-onenorth",
    "singapore-queenstown"
]

class NuScenesMap:
    """Implements a nuScenes' map."""

    def __init__(self, root_dir: Path) -> None:
        """Implements a nuScenes' map.
        
        :param root_dir: Path to the json files created by 
            `create_nuscenes_graph.py`script.
        """
        self.root_dir = Path(root_dir)
        self.city_halluc_bbox = self.load_hallucinated_bbox()
        self.city_lane_centerlines_dict = self.build_centerline_index()

    
    def load_hallucinated_bbox(
        self
    ) -> Mapping[str, List[float]]:
        """
        Load the hallucinated bounding boxes from the corresponding files
        created by the `create_nuscenes_graph.py`script.
        """
        city_halluc_bbox = {}

        for city in NUSCENES_MAPS:
            json_path = self.root_dir / Path(f"{city}_laneid_to_halluc_bbox.json")
            with open(json_path, "rb") as f:
                city_halluc_bbox[city] = json.load(f)
        
        return city_halluc_bbox

    
    def build_centerline_index(self) -> Mapping[str, Mapping[str, Lane]]:
        """
        Load the map files created by the `create_nuscenes_graph.py`script, and
        return a nested dictionary of the following format:

        {
            "<city_name>: {
                "<lane_token>": Lane object,
                "<lane_token>": Lane object,
                ...
            },
            ...
        }
        """
        city_lane_centerlines_dict = {}
        for city in NUSCENES_MAPS:
            json_file = self.root_dir / f"{city}_map.json"
            city_lane_centerlines_dict[city] = Graph(json_file).lanes

        return city_lane_centerlines_dict


    def get_lane_tokens_in_xy_bbox(
        self, 
        query_x: float, 
        query_y: float, 
        city:str, 
        manhattan_range:float=5.0
    ) -> List[str]:
        """Finds all lanes inside a query bbox.
        
        :param query_x: X coordinate for the query bounding box.
        :param query_y: Y coordinate for the query bounding box.
        :param city: Name of the city.
        :param manhattan_range: Range for the manhattan search.
        """
        query_min_x = query_x - manhattan_range
        query_max_x = query_x + manhattan_range
        query_min_y = query_y - manhattan_range
        query_max_y = query_y + manhattan_range

        tokens = list(self.city_halluc_bbox[city].keys())
        overlap_indxs = find_all_polygon_bboxes_overlapping_query_bbox(
            np.array(list(self.city_halluc_bbox[city].values())),
            np.array([query_min_x, query_min_y, query_max_x, query_max_y]),
        )

        neighborhood_lane_tokens = []
        for overlap_idx in overlap_indxs:
            lane_token = tokens[overlap_idx]
            neighborhood_lane_tokens.append(lane_token)

        return neighborhood_lane_tokens

    
    def get_lane_segment_polygon(self, 
        lane_token: str, 
        city: str
    ) -> np.ndarray:
        """
        Generate 3D lane polygon around the centerline of a `lane_token`. The 
        ground height is assumed zero for all points.

        :param lane_token: Lane token of the nuScenes dataset.
        :param city: Name of the city.
        """
        lane_centerline = \
            self.city_lane_centerlines_dict[city][lane_token].centerline
        lane_polygon = centerline_to_polygon(lane_centerline[:, :2])

        # Append zero ground height
        ground_height = np.zeros(
            lane_polygon.shape[0], dtype=lane_polygon.dtype)

        return np.hstack([lane_polygon, ground_height[:, np.newaxis]])

    


    

    