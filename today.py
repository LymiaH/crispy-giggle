import cv2
import numpy as np

pattern_size = (10, 7)
ref = cv2.imread("checkerboard.png")
reference_found, reference_corners = cv2.findChessboardCorners(ref, pattern_size)
reference_corner_array = np.array(reference_corners).reshape(len(reference_corners), 2)

if not reference_found:
    exit(1)

# Get camera
cap = cv2.VideoCapture(1)

pc = None
homo = None

# Setup window
cv2.namedWindow('image')
sized = False
output_image = None
#chessboard_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
chessboard_flags = cv2.CALIB_CB_FAST_CHECK

while not (cv2.waitKey(1) & 0xFF == ord('q')):
    ret, source = cap.read()

    if not ret:
        exit(1)

    if not sized:
        sized = True
        cv2.resizeWindow('image', source.shape[1] + ref.shape[1], max(source.shape[0], ref.shape[0]))
        output_image = np.zeros((max(source.shape[0], ref.shape[0]), source.shape[1] + ref.shape[1], 3), np.uint8)

    gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    found, projected_corners = cv2.findChessboardCorners(gray, pattern_size, flags=chessboard_flags)

    result = None

    if found:
        pc = np.array(projected_corners).reshape(len(projected_corners), 2)
        homo = cv2.findHomography(pc, reference_corner_array)
        result = cv2.warpPerspective(source, homo[0], (ref.shape[1], ref.shape[0]))
    elif homo is not None:
        result = cv2.warpPerspective(source, homo[0], (ref.shape[1], ref.shape[0]))

    output_image[:source.shape[0], :source.shape[1], :3] = source

    if result is not None:
        output_image[:ref.shape[0], source.shape[1]:source.shape[1] + ref.shape[1], :3] = result

    cv2.imshow("image", output_image)
