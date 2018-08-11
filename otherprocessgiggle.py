import json
from typing import List, Dict

import cv2
import tempfile as tf
import subprocess
import os

CAM = None
PARAMS = "-q -s -1 threshmarker invert connected brushfire afterbrush invert waysimp wayprint"
COMMAND = 'python process.py'
WORKING_DIRECTORY = "../crispy-giggle/"

def capture_waypoints() -> List:
    waypoints = [[640 // 4, 480 // 2], [640 * 3 // 2, 480 // 2]]
    print("[CAPWAY] Capturing image...")
    # if CAM is None:
    #     print("[CAPWAY] No Camera found")
    #     return waypoints
    # frame = CAM.get_frame()
    frame = cv2.imread("squiggly.png")
    if frame is None:
        print("[CAPWAY] Failed to get frame from camera")
        return waypoints
    print("[CAPWAY] Saving frame...")
    path = tf.NamedTemporaryFile(suffix='.png').name
    cv2.imwrite(path, frame)
    params = "-i " + path + " " + PARAMS
    args = COMMAND + " " + params
    print("[CAPWAY] Running: " + args)
    result = subprocess.run(args, cwd=WORKING_DIRECTORY, capture_output=True, text=True)
    print("[CAPWAY] Result: " + result.stdout)
    if len(result.stderr) > 0:
        print("[CAPWAY] Result: " + result.stderr)
    os.remove(path)
    data = json.loads(result.stdout)
    if data is None or "waypoints" not in data:
        print("[CAPWAY] Failed to capture waypoints (" + result.stdout + ")")
    else:
        print("[CAPWAY] Found " + str(len(data["waypoints"])) + " waypoints!")
        waypoints = data["waypoints"]
    return waypoints

if __name__ == '__main__':
    print(capture_waypoints())
    pass
