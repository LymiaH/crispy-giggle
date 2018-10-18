from typing import Any, Callable, Dict, Iterator, List, Set, Tuple

import cv2
import numpy as np

def run(iteration: int, img: np.ndarray, data: Dict[str, Any], global_data: Dict[str, Any]) -> (np.ndarray, bool):
    retval, labels, stats, centroids = global_data["connected"]["connected"]
    original = img

    if "canny" in global_data:
        original = global_data["canny"]["input"]

    img = np.copy(img)

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    groups = {} # type: Dict[int, np.ndarray]
    for group_id in range(1, retval):
        mask = np.zeros(labels.shape, np.uint8)
        mask[labels == group_id] = 255
        groups[group_id] = cv2.findNonZero(mask)

    midpoints = []
    for group_id, group in groups.items():
        other_groups = None # type: np.ndarray
        for other_group_id, other_group in groups.items():
            if group_id == other_group_id:
                continue
            if other_groups is None:
                other_groups = other_group
            else:
                other_groups = np.concatenate((other_groups, other_group))

        # From step 3 of Michael's code
        for pixel in group:
            distances = np.sqrt((other_groups[:, :, 0] - pixel[0][0]) ** 2 + (other_groups[:, :, 1] - pixel[0][1]) ** 2)
            nearest_index = np.argmin(distances)
            nearest_inner_pixel = other_groups[nearest_index]
            midpointX = int((pixel[0][0] + nearest_inner_pixel[0][0]) / 2)
            midpointY = int((pixel[0][1] + nearest_inner_pixel[0][1]) / 2)
            midpoint = (midpointX, midpointY)
            if midpoint == (nearest_inner_pixel[0][0], nearest_inner_pixel[0][1]):
                continue
            cv2.circle(img, midpoint, 1, [0, 0, 255], -1)
            midpoints.append(midpoint)

    data["midpoints"] = midpoints
    data["img"] = img
    global_data["michael"] = data
    return original, True
