import math
import time
from typing import Any, Callable, Dict, FrozenSet, Iterator, Set, Tuple

import cv2
import numpy as np
from cython_functions import some_filter

import constants
from common import hsl_to_rgb, eprint
from filters.brushfire import VORONOI


def can_see_cached(mask: np.ndarray, positions: Dict[int, Tuple[int, int]], line_of_sight: Dict[FrozenSet[int], bool], a: int, b: int) -> bool:
    ref = frozenset([a, b])
    if ref in line_of_sight:
        return line_of_sight[ref]
    see = can_see(mask, positions[a], positions[b])
    line_of_sight[ref] = see
    return see

def can_see(mask: np.ndarray, a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    # TODO: Reimplement, LOL
    minr = max(min(a[0], b[0]), 0)
    minc = max(min(a[1], b[1]), 0)
    maxr = min(max(a[0], b[0]) + 1, mask.shape[0])
    maxc = min(max(a[1], b[1]) + 1, mask.shape[1])
    R = maxr - minr
    C = maxc - minc
    if R <= 0 or C <= 0:
        eprint("What is going on?")
    temp = np.zeros([R, C], np.uint8)
    cv2.line(temp, (a[1] - minc, a[0] - minr), (b[1] - minc, b[0] - minr), 255, 1)
    # Set all road pixels to 0 as well
    submask = mask[minr:maxr, minc:maxc]
    if temp.shape[0] != submask.shape[0] or temp.shape[1] != submask.shape[1]:
        eprint("Something went really wrong...")
    temp[submask > 0] = 0

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
    #         eprint(str(x) + ", " + str(y) + ": " + str(mask[x, y]))
    #         if mask[x, y] == 0:
    #             return False
    #         error = error + deltaerr
    #         if error >= 0.5:
    #             y = int(float(y) + np.sign(deltay))
    #             error = error - 1.0
    # return True

def can_tri(mask: np.ndarray, a: Tuple[int, int], b: Tuple[int, int], c: Tuple[int, int]) -> bool:
    # TODO: Reimplement, LOL
    minr = max(min(a[0], b[0], c[0]), 0)
    minc = max(min(a[1], b[1], c[1]), 0)
    maxr = min(max(a[0], b[0], c[0]) + 1, mask.shape[0])
    maxc = min(max(a[1], b[1], c[1]) + 1, mask.shape[1])
    R = maxr - minr
    C = maxc - minc
    if R <= 0 or C <= 0:
        eprint("What is going on?")
    temp = np.zeros([R, C], np.uint8)
    cv2.fillPoly(temp, np.array([[[a[1] - minc, a[0] - minr],[b[1] - minc, b[0] - minr],[c[1] - minc, c[0] - minr]]], 'int32'), 255)
    #cv2.line(temp, (a[1] - minc, a[0] - minr), (b[1] - minc, b[0] - minr), 255, 1)
    # Set all road pixels to 0 as well
    submask = mask[minr:maxr, minc:maxc]
    if temp.shape[0] != submask.shape[0] or temp.shape[1] != submask.shape[1]:
        eprint("Something went really wrong...")
    temp[submask > 0] = 0

    # If any is found, then there is a collision
    if cv2.findNonZero(temp) is not None:
        return False
    return True

def distance_squared(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

def distance(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    return math.sqrt(distance_squared(a, b))

def add_edge(positions: Dict[int, Tuple[int, int]], edges: Dict[int, Set[int]], dists: Dict[FrozenSet[int], float], a: int, b: int):
    if b in edges[a]:
        return
    edges[a].add(b)
    edges[b].add(a)
    dists[frozenset([a, b])] = distance(positions[a], positions[b])

def remove_edge(edges: Dict[int, Set[int]], dists: Dict[FrozenSet[int], float], a: int, b: int):
    ref = frozenset([a, b])
    if b not in edges[a] or a not in edges[b] or ref not in dists:
        return
    edges[a].remove(b)
    edges[b].remove(a)
    dists.pop(ref, None)

def run(iteration: int, img: np.ndarray, data: Dict[str, Any], global_data: Dict[str, Any]) -> (np.ndarray, bool):
    neighbours = global_data["neighbours"] # type: Callable[[int, int, int, int], Iterator[Tuple[int, int]]]
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    R, C = img.shape
    road_mask = np.zeros(img.shape[0:2], np.uint8)
    road_mask[img > 0] = 255

    if iteration == 0:
        point_mask = np.zeros(img.shape[0:2], np.uint8)
        # Load data
        if "afterbrush" in global_data:
            point_mask[global_data["afterbrush"]["img"] != 0] = 255
        elif "brushfire" in global_data:
            groups = global_data["brushfire"]["groups"]
            point_mask[groups == VORONOI] = 255
        elif "michael" in global_data:
            for point in global_data["michael"]["midpoints"]:
                point_mask[point[1], point[0]] = 255
        else:
            eprint("No waypoint data present.")

        point_mask = some_filter(point_mask)

        original_points = []
        ppp = cv2.findNonZero(point_mask)
        if ppp is not None:
            for i in range(0, ppp.shape[0]):
                (c, r) = ppp[i, 0, 0:2] # Open CV functions use x, y, even though indexing for numpy arrays are r, c. r = y, c = x
                original_points.append((r, c))
        else:
            eprint("No points found?")

        data["original_points"] = original_points

        points = set() # type: Set[int]
        positions = {} # type: Dict[int, Tuple[int, int]]
        positions_reverse = {} # type: Dict[Tuple(int, int), int]
        edges = {} # type: Dict[int, Set[int]]
        dists = {} # type: Dict[FrozenSet[int], float]
        N = len(original_points)
        line_of_sight = {} # type: Dict[FrozenSet[int], float]

        # Init positions
        for curr in range(N):
            positions[curr] = original_points[curr]
            positions_reverse[positions[curr]] = curr
            edges[curr] = set()

        for curr in range(N):
            points.add(curr)
            # Check if it can see any of the other previously added points, creating an edge between them
            # for other in range(curr):
            #     if distance_squared(positions[curr], positions[other]) <= 8*8 and can_see(road_mask, line_of_sight, positions[curr], positions[other]):
            #         add_edge(positions, edges, dists, curr, other)
            #         cv2.line(img, positions[curr], positions[other], line_colour, 1)

            for neighbour in neighbours(*positions[curr], R, C):
                if neighbour in positions_reverse:
                    add_edge(positions, edges, dists, curr, positions_reverse[neighbour])


        data["points"] = points
        data["positions"] = positions
        data["positions_reverse"] = positions_reverse
        data["edges"] = edges
        data["dists"] = dists
        data["N"] = N
        data["line_of_sight"] = line_of_sight

    points = data["points"] # type: Set[int]
    positions = data["positions"] # type: Dict[int, Tuple[int, int]]
    positions_reverse = data["positions_reverse"] # type: Dict[Tuple[int, int], int]
    edges = data["edges"] # type: Dict[int, Set[int]]
    dists = data["dists"] # type: Dict[FrozenSet[int], float]
    line_of_sight = data["line_of_sight"] # type: Dict[FrozenSet[int], float]

    # output
    #test_factor = 11
    test_factor = 3
    test_offset = test_factor // 2
    line_colour = tuple(hsl_to_rgb(0.333, 1.0, 0.5))
    centre_colour = tuple(hsl_to_rgb(0.0, 0.0, 0.2))

    test = cv2.cvtColor(np.zeros([R, C]).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    test[road_mask == 0] = 255

    for curr in points:
        (r, c) = positions[curr]
        test[r, c] = centre_colour

    test = cv2.resize(test, None, fx=test_factor, fy=test_factor, interpolation=cv2.INTER_NEAREST)

    for curr in points:
        (cr, cc) = positions[curr]
        for neighbour in edges[curr]:
            (nr, nc) = positions[neighbour]
            cv2.line(test, (cc * test_factor + test_offset, cr * test_factor + test_offset),
                     (nc * test_factor + test_offset, nr * test_factor + test_offset), line_colour, 1)

    data["img"] = test

    # Simplification
    def point_filter(curr):
        if "waysimp_mode" in constants.CONFIG:
            if constants.CONFIG["waysimp_mode"] == "none":
                return True
            elif constants.CONFIG["waysimp_mode"] == "triangle":
                if len(edges[curr]) == 2:
                    neighbours = list(edges[curr])
                    n1 = neighbours[0]
                    n2 = neighbours[1]
                    if can_tri(road_mask, positions[curr], positions[n1], positions[n2]):
                        remove_edge(edges, dists, curr, n1)
                        remove_edge(edges, dists, curr, n2)
                        add_edge(positions, edges, dists, n1, n2)
                        # points.remove(curr)
                        return False
                return True
        if len(edges[curr]) == 2:
            neighbours = list(edges[curr])
            n1 = neighbours[0]
            n2 = neighbours[1]
            if can_see(road_mask, positions[n1], positions[n2]):
                remove_edge(edges, dists, curr, n1)
                remove_edge(edges, dists, curr, n2)
                add_edge(positions, edges, dists, n1, n2)
                #points.remove(curr)
                return False
        return True

    new_points = set(filter(point_filter, points))
    changed = len(new_points) != len(points)
    data["points"] = points = new_points

    if constants.DEBUG:
        #cv2.imwrite("./images/" + time.strftime("waysimp_step_%Y-%m-%d_%H-%M-%S_" + str(iteration) + ".png"), test) # TODO: Remove
        if iteration == 0:
            cv2.imwrite("./images/" + time.strftime("waysimp_test_%Y-%m-%d_%H-%M-%S_begin.png"), test)
        elif not changed:
            cv2.imwrite("./images/" + time.strftime("waysimp_test_%Y-%m-%d_%H-%M-%S_end.png"), test)

    return img, not changed
