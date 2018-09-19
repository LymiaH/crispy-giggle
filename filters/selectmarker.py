from common import hsl_to_rgb
from typing import Any, Callable, Dict, Iterator, List, Set, Tuple
from itertools import chain

import cv2
import numpy as np

# Applied after afterbrush to select one connected area to be the desired path
def run(iteration: int, img: np.ndarray, data: Dict[str, Any], global_data: Dict[str, Any]) -> (np.ndarray, bool):
    retval, labels, stats, centroids = data["connected"] = cv2.connectedComponentsWithStats(img, 8, cv2.CV_32S)
    mask = np.zeros(labels.shape[0:2], np.uint8)
    mask[labels > 0] = 255
    points = cv2.findNonZero(mask)
    r, c = labels.shape[0:2] // 2
    # TODO Implement this for correctness
    return img, True
