"""This module generates a graph for the centerlines of a nuScenes
map and hallucinated bounding boxes for the corresponding lanes of the map, and
saves them in JSON files.

This implementation is inspired by bhayva01's nuscenes_to_argoverse repository
https://github.com/bhavya01/nuscenes_to_argoverse.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from nuscenes.map_expansion import arcline_path_utils
from nuscenes.map_expansion.map_api import NuScenesMap


def load_table(data: list) -> dict:
    """Load a nuScenes table.

    This method takes a list of nested dictionaries and returns a
    dictionary using the inner tokens as keys.

    :param data: A list of dictionaries.
    """
    table = {}
    for record in data:
        table[record["token"]] = record

    return table


def get_point_cloud_bbox(point_cloud: np.ndarray) -> list:
    """Calculate bounding box for `point_cloud`.

    This method returns the minimum size xy axis-aligned bounding box of a
    given set of 2D or 3D points.

    :param point_cloud: A point cloud of size (N,2) or (N,3).
    """
    x_min = np.amin(point_cloud[:, 0])
    x_max = np.amax(point_cloud[:, 0])

    y_min = np.amin(point_cloud[:, 1])
    y_max = np.amax(point_cloud[:, 1])

    return [x_min, y_min, x_max, y_max]


def create_halluc_bboxes(nodes: dict, lanes: list, polygons: dict) -> dict:
    """Create hallucinated bounding boxes for nuScenes `lanes`.

    This method creates bounding boxes for the lanes and lane connectors of the
    nuScenes dataset, using the exterior nodes of the polygons that form the
    lanes.

    :param nodes: The nodes in the nuScenes map.
    :param lanes: The lanes in the nuScenes map.
    :param polygons: The polygons that form the lanes of the nuScenes map.
    """
    polygons_to_nodes_xy = {}
    for polygon in polygons.values():
        polygons_to_nodes_xy[polygon["token"]] = []
        for exterior_node_token in polygon["exterior_node_tokens"]:
            node_x = nodes[exterior_node_token]["x"]
            node_y = nodes[exterior_node_token]["y"]

            polygons_to_nodes_xy[polygon["token"]].append((node_x, node_y))

    halluc_bboxes = {}
    for lane in lanes:
        lane_polygon = np.array(polygons_to_nodes_xy[lane["polygon_token"]])
        halluc_bboxes[lane["token"]] = get_point_cloud_bbox(lane_polygon)

    return halluc_bboxes


def create_json_graph(
        nusc_map: NuScenesMap,
        lanes: list, centerlines: dict,
        discretization_resolution: float = 0.5) -> dict:
    """Create a graph for nuScenes centerlines.

    This method creates a graph for the centerlines of the nuScenes
    dataset. The graph contains all of the lanes of the corresponding map,
    along with the nodes of the discretized centerlines. The JSON file may look
    like this:

        {
            "lanes":{
                "0073298b-b2f4-4f89-97cd-4241a1599831": {
                    "token":"0073298b-b2f4-4f89-97cd-4241a1599831",
                    "centerline_nodes":["1","2",...,"17"],
                    "predecessors": [
                        "09a84e1d-17a3-40d7-a734-f426df3e2089",
                        ...
                    ],
                    "successors":[
                        ...
                    ]
                },
                "01a3b994-8aa9-450d-865a-ceb0666c90fa": {
                    "token": "01a3b994-8aa9-450d-865a-ceb0666c90fa", 
                    "centerline_nodes": [...]
                    ...
                }
            },
            "nodes": {
                "1": {
                    "id": 1, 
                    "x": "486.91778944573264", 
                    "y": "812.8782745377198"
                }, 
                "2": {
                    ...
                },
                ...
            }
        }

    The ids of the nodes are arbitrary, since there are no official tokens.

    :param nusc_map: A NuScenesMap object.
    :param lanes: The lanes (lanes and lane connectors) of the nuScenes map.
    :param centerlines: The centerlines (arcline paths) for the lanes of the
        nuScenes map.
    :param discretization_resolution: The discretization resolution of the
        parametric curves of the lanes.
    """
    global_node_id = 1
    present_nodes = {}
    lanes_dict = {}
    nodes_dict = {}

    for lane in lanes:
        lanes_dict[lane["token"]] = {}
        lanes_dict[lane["token"]]["token"] = lane["token"]
        lanes_dict[lane["token"]]["centerline_nodes"] = []

        poses = arcline_path_utils.discretize_lane(
            centerlines[lane["token"]], discretization_resolution
        )

        for pose in poses:
            currNode = (pose[0], pose[1])
            if currNode not in present_nodes:
                nodes_dict[global_node_id] = {}
                nodes_dict[global_node_id]["id"] = global_node_id
                nodes_dict[global_node_id]["x"] = str(pose[0])
                nodes_dict[global_node_id]["y"] = str(pose[1])

                present_nodes[currNode] = str(global_node_id)
                global_node_id += 1

            lanes_dict[lane["token"]]["centerline_nodes"].append(
                present_nodes[currNode])

        lanes_dict[lane["token"]]["predecessors"] = nusc_map.get_incoming_lane_ids(
            lane["token"])
        lanes_dict[lane["token"]]["successors"] = nusc_map.get_outgoing_lane_ids(
            lane["token"])

        graph = {
            "nodes": nodes_dict,
            "lanes": lanes_dict
        }

    return graph


def main(nuscenes_dir: Path, output_dir: Path, maps_dir: Path) -> None:
    """The main method of this module.

    This method iterates through the maps of the nuScenes maps expansion
    package, and creates json files that contain the hallucinated bounding
    boxes for the lanes of the map and an the graph of the centerlines.

    :param nuscenes_dir: Path to the nuScenes data.
    :param output_dir: Path to save the results.
    :param maps_dir: Path to the maps expansion directory of nuScenes.
    """
    # Load json file. Loop over all the json files in directory
    for map_file in maps_dir.glob("*.json"):
        with open(map_file) as f:
            data = json.load(f)

        nusc_map = NuScenesMap(
            dataroot=f"{nuscenes_dir}", map_name=map_file.stem)

        # Load the data
        nodes = load_table(data["node"])
        polygons = load_table(data["polygon"])
        lanes = data["lane"] + data["lane_connector"]
        centerlines = data["arcline_path_3"]

        # Generate hallucinated bounding boxes
        lane_token_to_halluc_bbox = create_halluc_bboxes(
            nodes, lanes, polygons)

        # Save hallucinated bounding boxes
        with open(f"{output_dir}/{map_file.stem}"
                  "_laneid_to_halluc_bbox.json", "w") as outfile:
            json.dump(lane_token_to_halluc_bbox, outfile)

        # Generate graph for nuScenes centerlines to json file
        graph = create_json_graph(nusc_map, lanes, centerlines)

        with open(f"{output_dir}/{map_file.stem}_map.json", "w") as outfile:
            json.dump(graph, outfile)


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(
        description="Create an XML graph for nuScenes centerlines.")

    # Main arguments
    parser.add_argument(
        '--data_dir', type=str, help="Path to nuscenes data", required=True)
    parser.add_argument(
        '--output_dir', type=str, help="Path to write output", required=True)
    parser.add_argument(
        '--maps_dir', type=str, default=None,
        help="Path to maps dir (nuScenes maps expansion)")

    args = parser.parse_args()

    nuscenes_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    maps_dir = nuscenes_dir / Path(
        "maps/expansion/") if args.maps_dir is None else Path(args.maps_dir)

    if not nuscenes_dir.exists():
        raise FileNotFoundError(f"Directory '{nuscenes_dir}' does not exist.")

    if not maps_dir.exists():
        raise FileNotFoundError(f"Directory '{maps_dir}' does not exist.")

    output_dir.mkdir(parents=True, exist_ok=True)

    main(nuscenes_dir=nuscenes_dir, output_dir=output_dir, maps_dir=maps_dir)
