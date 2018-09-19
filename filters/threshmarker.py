from typing import Any, Dict

import cv2
import numpy as np

# Whiteboard marker
def run(iteration: int, img: np.ndarray, data: Dict[str, Any], global_data: Dict[str, Any]) -> (np.ndarray, bool):
    out = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    out = np.subtract(np.full_like(out, 255), out)
    #out = cv2.morphologyEx(out, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    #out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return out, True
