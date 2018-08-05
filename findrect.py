import numpy as np
import cv2
import f

# img = cv2.imread('rectangle_test_by_otaku_seraph-dbva4dt.png')
# img = cv2.imread('Rectangle-test.jpg')
#img = cv2.imread('checkerboard_transform.png')
img = cv2.imread('checkerboard.png')

found, corners = cv2.findChessboardCorners(img, (10, 7), flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                               cv2.CALIB_CB_NORMALIZE_IMAGE +
                                                               cv2.CALIB_CB_FAST_CHECK)

print(found)

if found:
    for corner in corners.reshape(len(corners), -1):
        print(corner)
        img = cv2.circle(img, tuple(corner), 10, (0, 0, 255), -1)

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
