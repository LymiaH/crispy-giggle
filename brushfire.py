from common import hsl_to_rgb
from typing import Any, Callable, Dict, Iterator, List, Set, Tuple
from itertools import chain

import cv2
import numpy as np

VORONOI = -2
TOP = -3
BOTTOM = -4
LEFT = -5
RIGHT = -6

def run(iteration: int, img: np.ndarray, data: Dict[str, Any], global_data: Dict[str, Any]) -> (np.ndarray, bool):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    R, C = img.shape
    neighbours: Callable[[int, int, int, int], Iterator[Tuple[int, int]]] = global_data["neighbours"]
    retval, labels, stats, centroids = global_data["connected"]["connected"]
    if iteration == 0:
        # wavefront: List[Tuple[int, int]] = []
        # it = np.nditer(labels, op_flags=['readonly'], flags=['multi_index'])
        # while not it.finished:
        #     r, c = it.multi_index
        #     front = False
        #     for neighbour in neighbours(r, c, R, C):
        #         if labels[neighbour] != labels[r, c]:
        #             front = True
        #             break
        #     if front:
        #         wavefront.append((r, c))
        #     it.iternext()
        # Initialize Components
        distances = np.zeros([R, C], np.int32)
        groups = np.zeros([R, C], np.int32)

        for group in range (1, retval):
            # Find pixels that are part of this group
            mask = np.zeros(img.shape, np.uint8)
            mask[labels == group] = 255
            # Update data
            distances[mask > 0] = 1
            groups[mask > 0] = group

        data["distances"] = distances
        data["groups"] = groups

        # # Find the wavefront
        # mask = np.zeros(img.shape, np.uint8)
        # mask[img > 0] = 255
        # outlined = cv2.morphologyEx(img, cv2.MORPH_ERODE, np.ones((3, 3), np.uint8))
        # mask[outlined > 0] = 0
        # wavefront = cv2.findNonZero(mask)
        # #cv2.imshow("test", mask)
        # data["wavefront"] = wavefront

    distances = data["distances"]
    groups = data["groups"]
    # wavefront = data["wavefront"]

    new_distances = np.copy(distances)
    new_groups = np.copy(groups)

    for group in chain(range(RIGHT, VORONOI), range(1, retval)):
        # Find pixels that are part of this group
        mask = np.zeros(img.shape, np.uint8)
        mask[groups == group] = 255
        # grow it a little
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))

        # Include the edges if this is the first iteration
        if iteration == 0 and RIGHT <= group < VORONOI:
            if group == TOP:
                mask[0, 0:C - 1] = 255
            elif group == RIGHT:
                mask[0:R - 1, C - 1] = 255
            elif group == BOTTOM:
                mask[R - 1, 0:C - 1] = 255
            elif group == LEFT:
                mask[0:R - 1, 0] = 255
            else:
                print("How'd ya get here?")

        # Take away stuff that's already part of any groups since the previous iteration
        mask[groups != 0] = 0

        # Now we want to update the groups, if any of the group expansions intersect with another group's expansion, then it becomes a voronoi point
        # TODO think about the case where two of them touch but don't intersect yet... So touching at the same time


        # Find the new Voronoi points
        vori = np.zeros(img.shape, np.uint8)
        # vori[new_groups != 0] = 255
        # vori[groups != 0] = 0
        # vori[mask == 0] = 0

        # Excude them from the mask
        mask[vori > 0] = 0

        # Update data
        new_distances[mask > 0] = iteration + 2
        new_groups[mask > 0] = group
        new_distances[vori > 0] = iteration
        new_groups[vori > 0] = VORONOI

    data["distances"] = new_distances
    data["groups"] = new_groups

    output = cv2.cvtColor(np.zeros([R, C]).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for group in chain(range(RIGHT, VORONOI + 1), range(1, retval)):
        (r, g, b) = hsl_to_rgb(float(group) / (retval - 1), 1.0, 0.5)
        if RIGHT <= group <= VORONOI:
            if group == TOP:
                (r, g, b) = hsl_to_rgb(0.0, 1.0, 0.2)
            elif group == RIGHT:
                (r, g, b) = hsl_to_rgb(0.25, 1.0, 0.2)
            elif group == BOTTOM:
                (r, g, b) = hsl_to_rgb(0.5, 1.0, 0.2)
            elif group == LEFT:
                (r, g, b) = hsl_to_rgb(0.75, 1.0, 0.2)
            elif group == VORONOI:
                (r, g, b) = hsl_to_rgb(0.0, 1.0, 1.0)
            else:
                print("How'd ya get here...?")
        mask = np.zeros(img.shape, np.uint8)
        mask[new_groups == group] = 255
        output[mask > 0] = (b, g, r)

    data["img"] = output

    # Check for completion
    mask = np.zeros([R, C], np.uint8)
    mask[new_distances == 0] = 255
    zero_distances = cv2.findNonZero(mask)

    # Add to global data
    global_data["brushfire"] = data

    return img, zero_distances is None
