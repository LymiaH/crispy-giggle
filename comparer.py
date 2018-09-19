import json
from argparse import ArgumentParser, ArgumentError, ArgumentTypeError
from ast import literal_eval
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import cv2
import tempfile as tf
import subprocess
import os

import numpy as np
import time

DEBUG = False
# -d -i 8.png -s -1 invert threshold connected brushfire afterbrush invert waysimp wayprint
PARAMS = ['-q','-s','-1','threshold','invert','connected','brushfire','afterbrush','invert','waysimp','wayprintraw']
PARAMS_GRAPH = ['-q','-s','-1','threshold','invert','connected','brushfire','afterbrush','invert','waysimp','waygraph']
COMMAND = ['python', 'process.py']
WORKING_DIRECTORY = Path("../crispy-giggle/")

def make_a_path(size: int = 512, thickness: int = 7):
    path = np.zeros(shape=[0, 2], dtype=np.int32)

    cv2.namedWindow('drawloop')

    def add_node(event, x, y, flags, param):
        nonlocal path
        if event == cv2.EVENT_LBUTTONDOWN:
            path = np.concatenate((path, np.array([(x, y)], dtype=np.int32)), 0)

    cv2.setMouseCallback('drawloop', add_node)

    while cv2.getWindowProperty('drawloop', 0) >= 0:
        img = np.zeros([size, size, 3], np.uint8)
        if path.shape[0] > 0:
            cv2.polylines(img, [path], True, (255, 255, 255), thickness)
        cv2.imshow('drawloop', img)
        code = cv2.waitKey(20)
        if code == 27:
            break
        elif code == ord('z'):
            if path.shape[0] > 0:
                path = path[:-1, :]
    if path.shape[0] > 0:
        output = []
        for i in range(path.shape[0]):
            output.append((int(path[i, 0]), int(path[i, 1])))
        # Save to a file!
        (WORKING_DIRECTORY / 'paths' / time.strftime("comparer_%Y-%m-%d_%H-%M-%S.txt")).write_text(str(output))
    return path

def run_processor(frame: np.ndarray, params: List[str]):
    print("[CAPWAY] Saving image to temporary file...")
    path = tf.NamedTemporaryFile(suffix='.png').name
    cv2.imwrite(path, frame)

    args = []
    for arg in COMMAND: args.append(arg)
    args.append('-i')
    args.append(path)
    for arg in params: args.append(arg)

    print("[CAPWAY] Running: " + ' '.join(args))
    # result = subprocess.run(args, cwd=WORKING_DIRECTORY, capture_output=True, text=True)
    p = subprocess.Popen(args, cwd=str(WORKING_DIRECTORY), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    stdout, stderr = p.communicate()
    stdout = stdout.decode("utf-8")
    stderr = stderr.decode("utf-8")
    print("[CAPWAY] Result: " + stdout)
    if len(stderr) > 0:
        print("[CAPWAY] Result: " + stderr)
    os.remove(path)
    return stdout

def get_waypoints(frame: np.ndarray) -> List:
    waypoints = []
    stdout = run_processor(frame, PARAMS)
    data = json.loads(stdout)
    if data is None or "waypoints" not in data:
        print("[CAPWAY] Failed to capture waypoints (" + stdout + ")")
    else:
        print("[CAPWAY] Found " + str(len(data["waypoints"])) + " waypoints!")
        waypoints = data["waypoints"]
    return waypoints

def render_waypoints(img: np.ndarray, waypoints: np.ndarray):
    # Draw the result image
    if waypoints.shape[0] > 0:
        cv2.polylines(img, [waypoints], True, 255, THICKNESS)
    return img

def detect_and_render_waypoints(input: np.ndarray, output: np.ndarray):
    # Detect waypoints
    waypoints = np.array(get_waypoints(input), dtype=np.int32)
    print(waypoints)

    # Render waypoints
    return render_waypoints(output, waypoints)

def get_graph(frame: np.ndarray):
    stdout = run_processor(frame, PARAMS_GRAPH)
    data = json.loads(stdout)
    if data is None or "nodes" not in data or "edges" not in data:
        print("[CAPWAY] Failed to capture graph (" + stdout + ")")
        return {}, {}
    else:
        print("[CAPWAY] Found " + str(len(data["nodes"])) + " nodes and " + str(len(data["edges"])) + " edges!")
        output_nodes = {}
        for wid, pos in data["nodes"].items():
            output_nodes[int(wid)] = (int(pos[0]), int(pos[1]))
        output_edges = {}
        for wid, connections in data["edges"].items():
            output_edges[int(wid)] = set([int(cwid) for cwid in connections])

        return output_nodes, output_edges

def render_graph(img: np.ndarray, nodes, edges):
    for wid, connections in edges.items():
        r1 = nodes[wid][0]
        c1 = nodes[wid][1]
        if len(connections) == 0:
            continue
        for cwid in connections:
            r2 = nodes[cwid][0]
            c2 = nodes[cwid][1]
            cv2.line(img, (c1, r1), (c2, r2), 255, thickness=THICKNESS)
    return img


def detect_and_render_graph(input: np.ndarray, output: np.ndarray):
    # Detect graph
    nodes, edges = get_graph(input)
    print("Nodes:")
    print(nodes)
    print("Edges:")
    print(edges)

    # Render graph
    return render_graph(output, nodes, edges)

if __name__ == '__main__':
    args = ArgumentParser()

    def check_array_of_tuples(arg: str) -> Optional[np.ndarray]:
        if arg is None or arg == "":
            return None
        path = arg
        # Try read the string as a file
        try:
            with open(path, 'r') as path_file:
                path = path_file.read()
        except IOError:
            pass
        path = literal_eval(path)

        if not isinstance(path, list):
            raise ArgumentTypeError("Expected a list")

        for node in path:
            if not isinstance(node, (tuple, list)):
                raise ArgumentTypeError("Expected a list of tuples/lists")
            if  len(node) != 2:
                raise ArgumentTypeError("Should have lengths of 2")
            for dim in node:
                if not isinstance(dim, int):
                    raise ArgumentTypeError("Should be integers")

        return np.array(path, dtype=np.int32)

    args.add_argument("-i", "--input", type=check_array_of_tuples, default="",
                      help="Input loop or file containing loop (e.g. [(0,0),(0, 100),(100, 100),(100,0)] or loop.txt). Otherwise a dialog will open to allow you to create a loop.")
    args.add_argument("-t", "--thickness", type=int, default=15,
                      help="Line thickness to use")
    args.add_argument("-s", "--size", type=int, default=1024,
                      help="Width and Height of canvas (only used when making your own path)")
    args.add_argument("-d", "--debug", action='store_true', required=False,
                      help="Extra debug information")
    args.add_argument("-q", "--quiet", action='store_true', required=False,
                      help="Don't show guis")
    MODES = {
        'way': detect_and_render_waypoints,
        'graph': detect_and_render_graph,
    }

    def check_mode(param: str):
        param = param.lower()
        if param not in MODES:
            raise ArgumentTypeError("Must be one of: " + str(MODES))
        return MODES[param]

    args.add_argument("-m", "--mode", type=check_mode, default="way",
                      help="Detection mode")

    args = vars(args.parse_args())
    cv2.waitKey(1)

    DEBUG = args["debug"]
    if DEBUG:
        PARAMS.insert(0, "-d")
        PARAMS.remove('-q')
    THICKNESS = args["thickness"]
    MODE = args["mode"]

    # Make your own path!
    if not "input" in args or args["input"] is None:
        args["input"] = make_a_path(args["size"], THICKNESS)
    LOOP = args["input"]
    if LOOP.shape[0] < 1:
        raise RuntimeError("Loop is empty")
    print(LOOP)

    # Work out dimensions for image with padding
    minx, miny = np.min(LOOP, axis=0)
    maxx, maxy = np.max(LOOP, axis=0)
    minx -= THICKNESS*2
    miny -= THICKNESS*2
    maxx += THICKNESS*2
    maxy += THICKNESS*2

    # Translate all the points
    LOOP[:, 0] -= minx
    LOOP[:, 1] -= miny
    maxx -= minx
    maxy -= miny

    WIDTH = maxx
    HEIGHT = maxy

    # Draw the reference image
    img = np.zeros([HEIGHT, WIDTH], np.uint8)
    if LOOP.shape[0] > 0:
        cv2.polylines(img, [LOOP], True, 255, THICKNESS)
    REFERENCE = img
    cv2.imshow("reference", REFERENCE)
    cv2.waitKey(1)

    # Detect and re-render the image
    img = np.zeros([HEIGHT, WIDTH], np.uint8)
    MODE(REFERENCE, img)
    RESULT = img

    # Display Result
    cv2.imshow("result", RESULT)
    cv2.waitKey(1)

    # Draw the difference image
    img = np.zeros([HEIGHT, WIDTH, 3], np.uint8)
    maskRefRes = np.zeros(img.shape[:2], np.uint8)
    maskRef = np.zeros(img.shape[:2], np.uint8)
    maskRes = np.zeros(img.shape[:2], np.uint8)

    maskRefRes[REFERENCE == 255] = 255
    maskRefRes[RESULT == 255] = 255
    maskRef[REFERENCE == 255] = 255
    maskRef[RESULT == 255] = 0
    maskRes[RESULT == 255] = 255
    maskRes[REFERENCE == 255] = 0

    img[maskRefRes > 0] = (0, 255, 0) # Correct
    img[maskRef > 0] = (255, 0, 0) # Missing
    img[maskRes > 0] = (0, 0, 255)  # Extra
    num_correct = cv2.findNonZero(maskRefRes).shape[0]
    num_missing = cv2.findNonZero(maskRef).shape[0]
    num_extra = cv2.findNonZero(maskRes).shape[0]
    num_total = num_correct + num_extra + num_missing
    DIFFERENCE = img
    print("Correct: %d %d%%" % (num_correct, num_correct * 100 // num_total))
    print("Missing: %d %d%%" % (num_missing, num_missing * 100 // num_total))
    print("Extra: %d %d%%" % (num_extra, num_extra * 100 // num_total))
    print("Total: %d" % num_total)
    cv2.imshow("difference", DIFFERENCE)
    cv2.waitKey(0)
    pass
