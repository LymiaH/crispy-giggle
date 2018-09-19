import json
from typing import Any, Dict, List, Set, Tuple

import cv2
import numpy as np

from common import eprint
from filters.waysimp import distance_squared

DISPLAY_WIDTH = 1600
DISPLAY_HEIGHT = 1200

# Selects the path with a node closest to the centre
def run(iteration: int, img: np.ndarray, data: Dict[str, Any], global_data: Dict[str, Any]) -> (np.ndarray, bool):
    R, C = img.shape[0:2]
    waysimp_data = global_data["waysimp"]
    points = waysimp_data["points"] # type: Set[int]
    positions = waysimp_data["positions"] # type: Dict[int, Tuple[int, int]]
    positions_reverse = waysimp_data["positions_reverse"] # type: Dict[Tuple[int, int], int]
    edges = waysimp_data["edges"] # type: Dict[int, Set[int]]
    dists = waysimp_data["dists"] # type: Dict[Tuple[int, int], float]

    # Background Image
    background_img = np.full([R, C], 255, np.uint8)
    background_img = cv2.cvtColor(background_img, cv2.COLOR_GRAY2BGR)

    valid_wids = set()
    for wid, connections in edges.items():
        r1 = positions[wid][0]
        c1 = positions[wid][1]
        if len(connections) == 0:
            continue
        valid_wids.add(wid)
        for cwid in connections:
            valid_wids.add(cwid)
            r2 = positions[cwid][0]
            c2 = positions[cwid][1]
            cv2.line(background_img, (c1, r1), (c2, r2), (0,255,0), thickness=3)

    for wid in valid_wids:
        r = int(positions[wid][0])
        c = int(positions[wid][1])
        cv2.circle(background_img, (c, r), 3, (0, 0, 255), -1)

    # Edges
    output_edges = dict()
    for wid, connections in edges.items():
        if len(connections) == 0:
            continue
        output_edges[int(wid)] = [int(cwid) for cwid in connections]

    # Nodes that are part of at least one edge
    output_nodes = dict()
    for wid in valid_wids:
        output_nodes[int(wid)] = [int(p) for p in positions[wid]]

    # Output
    table_data = {}
    table_data["nodes"] = output_nodes
    table_data["edges"] = output_edges
    print(json.dumps(table_data))

    data["img"] = background_img
    return img, True
