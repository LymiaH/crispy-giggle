import numpy as np
import cv2

# Source: https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/

def dist(a, b):
	dx = a[0] - b[0]
	dy = a[1] - b[1]
	return np.sqrt(dx * dx + dy * dy)

def perspective_transform_matrix(corners, h_on_w = -1):
	rect = np.zeros((4, 2), dtype = "float32")

	for i in range(0, 4):
		rect[i] = corners[i]

	w = max(int(dist(rect[0], rect[1])), int(dist(rect[2], rect[3])))
 	h = max(int(dist(rect[1], rect[2])), int(dist(rect[3], rect[0])))
	if h_on_w > 0:
		h = int(h_on_w * w)
	else:
		sq = max(w, h)
		w = sq
		h = sq
		#pass
		

	to = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype = "float32")
 
	M = cv2.getPerspectiveTransform(rect, to)
	return M, w, h
