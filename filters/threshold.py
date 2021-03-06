from typing import Any, Dict

import cv2
import numpy as np


def run(iteration: int, img: np.ndarray, data: Dict[str, Any], global_data: Dict[str, Any]) -> (np.ndarray, bool):
    # for x in np.nditer(img, op_flags=['readwrite']):
    #     if x <= 127:
    #         x[...] = 0
    #     else:
    #         x[...] = 255
    new = np.zeros(img.shape[0:2], np.uint8)
    new[img > 127] = 255
    return new, True
