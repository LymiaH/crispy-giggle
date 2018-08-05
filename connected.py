from common import hsl_to_rgb
from typing import Any, Callable, Dict, Iterator, Set, Tuple

import cv2
import numpy as np

def run(iteration: int, img: np.ndarray, data: Dict[str, Any], global_data: Dict[str, Any]) -> (np.ndarray, bool):
    R, C = img.shape[:2]
    binary = (img > 0).astype(np.uint8)
    retval, labels, stats, centroids = global_data["connected"] = cv2.connectedComponentsWithStats(binary, 8, cv2.CV_32S)

    output = np.zeros([R, C, 3]).astype(np.uint8)
    for i in range(retval - 1):
        fill = hsl_to_rgb(360 * i / retval, 1, 0.5)
        mask = labels == i + 1
        output[mask] = fill

    data["img"] = output
    return img, True
