from typing import Any, Dict

import cv2
import numpy as np


def run(iteration: int, img: np.ndarray, data: Dict[str, Any], global_data: Dict[str, Any]) -> (np.ndarray, bool):
    # for x in np.nditer(img, op_flags=['readwrite']):
    #     x[...] = 255 - x
    yay = np.full_like(img, 255)
    return np.subtract(yay, img), True
