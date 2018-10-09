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
    waypoint_ids = []  # type: List[int]
    if best_point == (-1, -1):
        eprint("Oh come on...")
        #return img, True
    else:
        stack.append(positions_reverse[best_point])

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
    waypoints = []
    for wid in waypoint_ids:
        pos = positions[wid]
        r = int(pos[0])
        c = int(pos[1])
        waypoints.append([r, c])

    # Background Image
    background_img = np.full([R, C], 255, np.uint8)
    background_img = cv2.cvtColor(background_img, cv2.COLOR_GRAY2BGR)

    for wid in waypoint_ids:
        pos = positions[wid]
        r = int(pos[0])
        c = int(pos[1])
        cv2.circle(background_img, (c, r), 5, (0, 0, 0), -1)

    # they need way points to be output in x, y format
    for way in waypoints:
        temp = way[0]
        way[0] = way[1]
        way[1] = temp

    table_data = {}
    table_data["waypoints"] = waypoints
    # table_data["background"] = pickle.dumps(background_img)
    # print(pickle.dumps(table_data))
    print(json.dumps(table_data))
    data["img"] = background_img
    return img, True
