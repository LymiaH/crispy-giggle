from typing import Any, Callable, Dict, Iterator, List, Set, Tuple

import cv2
import numpy as np

def run(iteration: int, img: np.ndarray, data: Dict[str, Any], global_data: Dict[str, Any]) -> (np.ndarray, bool):
    R, C = img.shape[:2]
    neighbours: Callable[[int, int, int, int], Iterator[Tuple[int, int]]] = global_data["neighbours"]
    retval, labels, stats, centroids = global_data["connected"]
    if iteration == 0:
        wavefront: List[Tuple[int, int]] = set()
        # Find the wavefront
        it = np.nditer(labels, op_flags=['readonly'], flags=['multi_index'])
        while not it.finished:
            r, c = it.multi_index
            front = False
            for neighbour in neighbours(r, c, R, C):
                if labels[neighbour] != labels[r, c]:
                    front = True
                    break
            if front:
                wavefront.add((r, c))
            it.iternext()
        data["wavefront"] = wavefront
    wavefront = data["wavefront"]


    output = np.zeros(img.shape[:2]).astype(np.uint8)
    for i in range(retval):
        fill = 255 * i // (retval - 1)
        mask = labels == i
        output[mask] = fill

    output = cv2.color




    global_data["brushfire"] = data
    return output, True
