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
    retval, labels, stats, centroids = global_data["connected"]["connected"]
    groups = global_data["brushfire"]["groups"]

    final = np.zeros(img.shape[0:2], np.uint8)
    for group in range(1, retval):
        mask = np.zeros(img.shape[0:2], np.uint8)
        mask[groups == group] = 255
        canny = cv2.Canny(mask, 100, 200)
        final = cv2.bitwise_or(final, canny)

    data["img"] = final
    return img, True