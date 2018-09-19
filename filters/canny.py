from typing import Any, Callable, Dict, Iterator, Set, Tuple

import cv2
import numpy as np

# From Michael's code
def run(iteration: int, img: np.ndarray, data: Dict[str, Any], global_data: Dict[str, Any]) -> (np.ndarray, bool):
    gray = img
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    canny = cv2.Canny(closed, 100, 200)
    return canny, True
