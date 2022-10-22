"""This module generated an XML graph for the centerlines of a nuScenes 
map and a json file containing the hallucinated bounding boxes for the 
corresponding lanes of the map.

This implementation is inspired by bhayva01's nuscenes_to_argoverse repository
https://github.com/bhavya01/nuscenes_to_argoverse. 
"""
import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from nuscenes.map_expansion import arcline_path_utils
from nuscenes.map_expansion.map_api import NuScenesMap

from utils import load_table


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


def create_xml_graph(
        nusc_map: NuScenesMap, root: ET.Element,
        lanes: list, centerlines: dict,
        discretization_resolution: float = 0.5) -> ET.ElementTree:
    """Create XML graph for nuScenes centerlines.

    This method creates an XML graph for the centerlines of the nuScenes
    dataset. The graph contains all of the lanes of the corresponding map, 
    along with the nodes of the discretized centerlines. The XML may look 
    like this:

        <lane token="0073298b-b2f4-4f89-97cd-4241a1599831">
            <node ref="1" />
            <node ref="2" />
            ...
            <node ref="17" />
            <tag k="predecessor" v="09a84e1d-17a3-40d7-a734-f426df3e2089" />
            ...
            <tag k="predecessor" v="5cf3ec09-477a-40d4-9c3a-03d610fad1b8" />
        </lane>
        <node id="1" x="486.91778944573264" y="812.8782745377198" />
        <node id="2" x="487.1413230286224" y="813.3026182765033" />
        ...

    The ids of the nodes are arbitrary, since there are no official tokens.

    :param nusc_map: A NuScenesMap object.
    :param root: An XML ETree to append nodes on.
    :param lanes: The lanes (lanes and lane connectors) of the nuScenes map.
    :param centerlines: The centerlines (arcline paths) for the lanes of the 
        nuScenes map.
    :param discretization_resolution: The discretization resolution of the
        parametric curves of the lanes.
    """
    global_node_id = 1
    present_nodes = {}

    for lane in lanes:
        lane_node = ET.SubElement(root, "lane")
        lane_node.set("token", lane["token"])

        poses = arcline_path_utils.discretize_lane(
            centerlines[lane["token"]], discretization_resolution
        )

        for pose in poses:
            currNode = (pose[0], pose[1])
            if currNode not in present_nodes:
                node = ET.SubElement(root, "node")
                node.set("id", str(global_node_id))
                node.set("x", str(pose[0]))
                node.set("y", str(pose[1]))

                present_nodes[currNode] = str(global_node_id)
                global_node_id += 1

            lane_subnode = ET.SubElement(lane_node, "node")
            lane_subnode.set("ref", present_nodes[currNode])

        predecessors = nusc_map.get_incoming_lane_ids(lane["token"])
        successors = nusc_map.get_outgoing_lane_ids(lane["token"])

        for pred_lane_token in predecessors:
            pre = ET.SubElement(lane_node, "tag")
            pre.set("k", "predecessor")
            pre.set("v", pred_lane_token)

        for succ_lane_token in successors:
            succ = ET.SubElement(node, "tag")
            succ.set("k", "successor")
            succ.set("v", succ_lane_token)

    tree = ET.ElementTree(root)
    return tree


def main(nuscenes_dir: Path, output_dir: Path, maps_dir: Path) -> None:
    """The main method of this module.

    This method iterates through the maps of the nuScenes maps expansion
    package, and creates a json file that contains the hallucinated bounding
    boxes for the lanes of the map and an XML graph for the centerlines.

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

        # Create new xml ETree
        root = ET.Element("nuScenesMap")

        # Generate hallucinated bounding boxes
        lane_token_to_halluc_bbox = create_halluc_bboxes(
            nodes, lanes, polygons)

        # Save hallucinated bounding boxes
        with open(f"{output_dir}/{map_file.stem}"
                  "_laneid_to_halluc_bbox.json", "w") as outfile:
            json.dump(lane_token_to_halluc_bbox, outfile)

        # Generate XML graph for nuScenes centerlines
        tree = create_xml_graph(nusc_map, root, lanes, centerlines)

        # Save the XML graph
        with open(
                f"{output_dir}/{map_file.stem}_map.xml", "wb") as files:
            tree.write(files)


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
