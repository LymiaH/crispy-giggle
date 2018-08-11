from common import hsl_to_rgb, eprint, jdump
from typing import Any, Callable, Dict, Iterator, List, Set, Tuple
from waysimp import distance_squared

import cv2
import math
import numpy as np
import json

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

    # Ideas
    # Select a random point and follow it along till it reaches the start again?
    # Maybe it should be the agent's job to determine what path to take at the intersections...
    # For now, will just take a random path and hope it leads back to the start...
    # Do a DFS then stop once you reach the start?? how to detect this...
    # Get exit direction of path then make it

    centre = (R // 2, C // 2)
    min_distsq = img.shape[0] ** 2 + img.shape[1] ** 2
    best_point = (-1, -1)
    for point in points:
        distsq = distance_squared(centre, positions[point])
        if distsq < min_distsq:
            min_distsq = distsq
            best_point = positions[point]

    stack = [] # type: List[int]
    visited = set() # type: Set[int]
    if best_point == (-1, -1):
        eprint("Oh come on...")
        return img, True
    pstart = positions_reverse[best_point]
    stack.append(pstart)
    waypoint_ids = [] # type: List[int]
    while len(stack) > 0:
        curr = stack.pop()
        if curr in visited:
            continue
        visited.add(curr)
        waypoint_ids.append(curr)
        for neighbour in edges[curr]:
            if neighbour in visited:
                continue
            stack.append(neighbour)
    # Format for tabletop-car-simulator
    MAP_WIDTH = 640
    MAP_HEIGHT = 480
    WAY_WIDTH = 640
    WAY_HEIGHT = 480
    waypoints = []
    for wid in waypoint_ids:
        pos = positions[wid]
        r = int(pos[0]) * WAY_HEIGHT // R
        c = int(pos[1]) * WAY_WIDTH // C
        waypoints.append([r, c])

    # Background Image
    background_img = np.full([MAP_HEIGHT, MAP_WIDTH], 255, np.uint8)
    background_img = cv2.cvtColor(background_img, cv2.COLOR_GRAY2BGR)

    for wid in waypoint_ids:
        pos = positions[wid]
        r = int(pos[0]) * MAP_HEIGHT // R
        c = int(pos[1]) * MAP_WIDTH // C
        cv2.circle(background_img, (c, r), 5, (0, 0, 0), -1)
    table_data = {}
    table_data["waypoints"] = waypoints
    # table_data["background"] = pickle.dumps(background_img)
    # print(pickle.dumps(table_data))
    print(json.dumps(table_data))
    data["img"] = background_img
    return img, True
