from argparse import ArgumentParser, ArgumentError
from typing import Any, Callable, Dict, Iterator, Tuple

import cv2
import importlib
import numpy as np
import constants

def neighbours8(r:int, c:int, R:int, C:int) -> Iterator[Tuple[int, int]]:
    up = r > 0
    down = r < R - 1
    left = c > 0
    right = c < C - 1
    if up:
        yield r - 1, c
        if left:
            yield r - 1, c - 1
        if right:
            yield r - 1, c + 1
    if left:
        yield r, c - 1
    if right:
        yield r, c + 1
    if down:
        yield r + 1, c
        if left:
            yield r + 1, c - 1
        if right:
            yield r + 1, c + 1

def neighbours4(r: int, c: int, R: int, C: int) -> Iterator[Tuple[int, int]]:
    if r > 0:
        yield r - 1, c
    if c > 0:
        yield r, c - 1
    if r < R - 1:
        yield r + 1, c
    if c < C - 1:
        yield r, c + 1

def resize(img: np.ndarray, max_side_length: int) -> np.ndarray:
    nr, nc = img.shape[:2]
    longest_side = max(nr, nc)
    if longest_side <= max_side_length:
        return img
    # Open cv functions are width/col first, then height/row second...
    return cv2.resize(img, (nc * max_side_length // longest_side, nr * max_side_length // longest_side))

def wait(window_name: str):
    if constants.QUIET:
        return
    while cv2.getWindowProperty(window_name, 0) >= 0 and cv2.waitKey(50) == -1:
        pass

if __name__ == '__main__':

    args = ArgumentParser()
    func_filter = Callable[[int, np.ndarray, Dict[str, Any], Dict[str, Any]], Tuple[np.ndarray, bool]]


    def check_image(path: str) -> np.ndarray:
        # Not sure if should be reading as grayscale image or colour...
        # Grayscale for now I guess
        img = cv2.imread(path, 0)
        if img is None:
            raise ArgumentError(path, "Could not read %s" % path)
        else:
            return img


    args.add_argument("-i", "--input", type=check_image, default="draw0.png", required=False,
                      help="Input Image")
    args.add_argument("-s", "--size", type=int, default=-1, required=False,
                      help="Maximum side length. Image will be scaled down if larger.")
    args.add_argument("-d", "--debug", action='store_true', required=False,
                      help="Debug Mode (prints debug information to stderr)")
    args.add_argument("-q", "--quiet", action='store_true', required=False,
                      help="Quiet Mode (does not imshow and automatically closes on completion)")

    def check_filter(identifier: str) -> Tuple[str, func_filter]:
        return identifier, importlib.import_module(identifier).run

    args.add_argument('filters', metavar='F', type=check_filter, nargs='*',
                        help='Functions to apply')
    args = vars(args.parse_args())

    # Arguments
    constants.DEBUG = args["debug"]
    constants.QUIET = args["quiet"]
    img: np.ndarray = args["input"]
    if args["size"] > 0:
        img = resize(img, args["size"])

    if not constants.QUIET:
        cv2.imshow("input", img)
        # wait("input")

    name_history: Dict[str, int] = {}

    filt: Tuple[str, func_filter]
    global_data: Dict[str, Any] = {}
    global_data["neighbours"] = neighbours8

    for filt in args["filters"]:
        name: str = filt[0]

        if not name in name_history:
            name_history[name] = 0
        else:
            name_history[name] = name_history[name] + 1
        name = name + str(name_history[name])

        f: func_filter = filt[1]
        i: int = 0
        data: Dict[str, Any] = {}
        data["input"] = np.copy(img)
        while True:
            img, finished = f(i, img, data, global_data)
            data["output"] = np.copy(img)
            global_data[filt[0]] = data
            if not constants.QUIET:
                if "img" in data:
                    cv2.imshow(name, data["img"])
                else:
                    cv2.imshow(name, img)
                cv2.waitKey(1)  # Needed to force image to display...
            if finished:
                break
            i = i + 1
    if not constants.QUIET:
        cv2.imshow("output", img)
        wait("output")
