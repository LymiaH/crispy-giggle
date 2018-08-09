from common import hsl_to_rgb
from typing import Any, Callable, Dict, Iterator, List, Set, Tuple
from itertools import chain
from brushfire import VORONOI

import cv2
import math
import numpy as np
import time

def can_see(mask: np.ndarray, a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    # TODO: Reimplement, LOL
    temp = np.zeros(mask.shape, np.uint8)
    cv2.line(temp, a, b, 255, 1)

    # Set all road pixels to 0 as well
    temp[mask > 0] = 0
    # If any is found, then there is a collision
    if cv2.findNonZero(temp) is not None:
        return False
    return True
    # # TODO Implement
    # # https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
    # x0: float = float(a[0])
    # y0: float = float(a[1])
    # x1: float = float(b[0])
    # y1: float = float(b[1])
    # deltax = x1 - x0
    # deltay = y1 - y0
    # if deltax == 0 and deltay == 0:
    #     return False
    # flip = math.fabs(deltax) < math.fabs(deltay)
    #
    # if flip:
    #     deltaerr: float = math.fabs(float(deltax) / float(deltay))
    #     error: float = 0.0
    #     x: int = int(x0)
    #     for y in range(a[1], b[1] + 1):
    #         if mask[x, y] == 0:
    #             return False
    #         error = error + deltaerr
    #         if error >= 0.5:
    #             x = int(float(x) + np.sign(deltax))
    #             error = error - 1.0
    # else:
    #     deltaerr: float = math.fabs(deltay / deltax)
    #     error: float = 0.0
    #     y: int = int(y0)
    #     for x in range(a[0], b[0] + 1):
    #         # If any of the pixels are not the road
    #         print(str(x) + ", " + str(y) + ": " + str(mask[x, y]))
    #         if mask[x, y] == 0:
    #             return False
    #         error = error + deltaerr
    #         if error >= 0.5:
    #             y = int(float(y) + np.sign(deltay))
    #             error = error - 1.0
    # return True

def distance_squared(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

def distance(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    return math.sqrt(distance_squared(a, b))

def add_edge(positions: Dict[int, Tuple[int, int]], edges: Dict[int, Set[int]], dists: Dict[Tuple[int, int], float], a: int, b: int):
    if b in edges[a]:
        return
    edges[a].add(b)
    edges[b].add(a)
    dists[(a, b)] = dists[(b, a)] = distance(positions[a], positions[b])

def remove_edge(edges: Dict[int, Set[int]], dists: Dict[Tuple[int, int], float], a: int, b: int):
    if b not in a:
        return
    edges[a].remove(b)
    edges[b].remove(a)
    dists.pop((a, b), None)
    dists.pop((b, a), None)

def run(iteration: int, img: np.ndarray, data: Dict[str, Any], global_data: Dict[str, Any]) -> (np.ndarray, bool):
    neighbours: Callable[[int, int, int, int], Iterator[Tuple[int, int]]] = global_data["neighbours"]
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    R, C = img.shape
    road_mask = np.zeros(img.shape[0:2], np.uint8)
    road_mask[img > 0] = 255

    if iteration == 0:
        original_points = []
        # Load data
        if "afterbrush" in global_data:
            ppp = cv2.findNonZero(global_data["afterbrush"]["img"])
            if ppp is not None:
                for i in range(0, ppp.shape[0]):
                    (r, c) = ppp[i, 0, 0:2]
                    original_points.append((r, c))
        elif "brushfire" in global_data:
            groups = global_data["brushfire"]["groups"]
            mask = np.zeros(img.shape[0:2], np.uint8)
            mask[groups == VORONOI] = 255
            voronoi = cv2.findNonZero(mask)
            if voronoi is not None:
                for i in range(0, voronoi.shape[0]):
                    (r, c) = voronoi[i, 0, 0:2]
                    original_points.append((r, c))
            else:
                print("Brushfire found no points!")
        elif "michael" in global_data:
            for point in global_data["michael"]["midpoints"]:
                original_points.append(point)
        else:
            print("No waypoint data present.")
        data["original_points"] = original_points

        points: Set[int] = set()
        positions: Dict[int, Tuple[int, int]] = {}
        positions_reverse: Dict[Tuple(int, int), int] = {}
        edges: Dict[int, Set[int]] = {}
        dists: Dict[Tuple[int, int], float] = {}
        N = len(original_points)
        img = cv2.cvtColor(np.zeros([R, C]).astype(np.uint8), cv2.COLOR_GRAY2BGR)

        # Init positions
        for curr in range(N):
            positions[curr] = original_points[curr]
            positions_reverse[positions[curr]] = curr
            edges[curr] = set()

        test_factor = 11
        test_offset = test_factor // 2
        line_colour = tuple(hsl_to_rgb(0.333, 1.0, 0.5))
        centre_colour = tuple(hsl_to_rgb(0.0, 0.0, 0.2))

        test = cv2.cvtColor(np.zeros([R, C]).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        test[road_mask == 0] = 255
        for curr in range(N):
            (r, c) = original_points[curr]
            test[c, r] = centre_colour
        test = cv2.resize(test, None, fx = test_factor, fy = test_factor, interpolation = cv2.INTER_NEAREST)


        for curr in range(N):
            points.add(curr)
            # Check if it can see any of the other previously added points, creating an edge between them
            # for other in range(curr):
            #     if distance_squared(positions[curr], positions[other]) <= 8*8 and can_see(road_mask, positions[curr], positions[other]):
            #         add_edge(positions, edges, dists, curr, other)
            #         cv2.line(img, positions[curr], positions[other], line_colour, 1)

            for neighbour in neighbours(*positions[curr], R, C):
                if neighbour in positions_reverse:
                    add_edge(positions, edges, dists, curr, positions_reverse[neighbour])
                    (cr, cc) = positions[curr]
                    (nr, nc) = neighbour
                    cv2.line(test, (cr * test_factor + test_offset, cc * test_factor + test_offset) , (nr * test_factor + test_offset, nc * test_factor + test_offset), line_colour, 1)

        data["points"] = points
        data["positions"] = positions
        data["positions_reverse"] = positions_reverse
        data["edges"] = edges
        data["dists"] = dists
        data["N"] = N
        data["img"] = test
        cv2.imwrite("./images/" + time.strftime("waysimp_test_%Y-%m-%d_%H-%M-%S.png"), test)

    points: Set[int] = data["points"]
    positions: Dict[int, Tuple[int, int]] = data["positions"]
    positions_reverse: Dict[Tuple[int, int], int] = data["positions_reverse"]
    edges: Dict[int, Set[int]] = data["edges"]
    dists: Dict[Tuple[int, int], float] = data["dists"]

    return img, True