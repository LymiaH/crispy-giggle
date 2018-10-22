import time
import os

import cv2

CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480

# Michael Finn's camera setup script
camSetupScript = """
v4l2-ctl \
-c auto_exposure=1 \
-c exposure_time_absolute=100 \
-c white_balance_auto_preset=0 \
-c red_balance=2300 \
-c blue_balance=1400
"""
os.system(camSetupScript)

stream = cv2.VideoCapture(0)
stream.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
stream.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
_, frame = stream.read()
stream.release()
cv2.imwrite("./" + time.strftime("capture_%Y-%m-%d_%H-%M-%S.png"), frame)
