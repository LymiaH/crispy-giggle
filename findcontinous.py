import numpy as np
import cv2
import f

testrino = None
#testrino = img = cv2.imread('chessboard.png')

# Get camera
cap = cv2.VideoCapture(0)

# Intialize windows
cv2.namedWindow("source")
cv2.namedWindow("output")

# Last 4 selected points
points = []
ow = 1
oh = 1
ready = False

# Points to transform
point_source = (0,0)
point_output = (0,0)
M = None
Marr = None

# Mouse handlers
def mouse_source(event, x, y, flags, param):
    global points, ready, M, Marr, ow, oh, point_source, point_output

    if event == cv2.EVENT_RBUTTONDOWN:
        points.append((x, y))
        if len(points) > 4:
            points.pop(0)
            # points = points[len(points) - 4:]
        ready = len(points) == 4
        print(points)
        point_source = (0,0)
        point_output = (0,0)

        if ready:
            Marr, ow, oh = f.perspective_transform_matrix(points,
                #-1
                15.5/7.5
                )
            M = np.matrix(Marr)
    
    if not ready:
        return
    
    if event == cv2.EVENT_LBUTTONDOWN:
        point_source = (x, y)
        #vec = np.matmul(M, (x, y, 1))
        vec = cv2.perspectiveTransform(np.float32(point_source).reshape(1, 1, -1), Marr)
        point_output = (int(vec.item(0)), int(vec.item(1)))
        #print(point_output)

def mouse_output(event, x, y, flags, param):
    global points, ready, M, Marr, ow, oh, point_source, point_output

    if not ready:
        return
    
    # Do stuff
    if event == cv2.EVENT_LBUTTONDOWN:
        point_output = (x, y)
        #vec = np.matmul(M.I, (x, y, 1))
        vec = cv2.perspectiveTransform(np.float32(point_output).reshape(1, 1, -1), np.asarray(M.I))
        point_source = (int(vec.item(0)), int(vec.item(1)))
        #print(point_source)

cv2.setMouseCallback("output", mouse_output)
cv2.setMouseCallback("source", mouse_source)

while(True):    
    ret, source = cap.read()
    if testrino is not None:
        source = testrino.copy()

    if ready:
        output = cv2.warpPerspective(source, Marr , (ow, oh))
        M = np.matrix(M)

        #Draw Transformed Points
        source = cv2.circle(source, point_source, 5, (0, 0, 255), -1)
        output = cv2.circle(output, point_output, 5, (0, 0, 255), -1)

        cv2.imshow('output', output)

    for p in points:
        source = cv2.circle(source, p, 3, (0,255,0), -1)
    
    cv2.imshow('source',source)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

cap.release()
cv2.destroyAllWindows()
