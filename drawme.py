import cv2
import numpy as np
import time
import itertools
from unionfind import UnionFind

R = 512
C = 512

# Setup window
cv2.namedWindow('main')
#img_i = np.zeros((R, C), np.uint8)
img_i = cv2.imread("window1.png", cv2.IMREAD_GRAYSCALE)
#img_i = cv2.threshold(img_i, 127, 255, cv2.THRESH_BINARY)[1]

down = False
last_pos = (0,0)
last_time = time.time()

def wtf(img):
    """
    Source: http://opencvpython.blogspot.com.au/2012/05/skeletonization-using-opencv-python.html
    :param img:
    :return: thinned image
    """
    thinned = np.zeros(img.shape, np.uint8)
    ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    iteration = 0
    file_prefix = "./images/" + time.strftime("wtf_%Y-%m-%d_%H-%M-%S_")
    joined = np.zeros((img.shape[0], img.shape[1]*2), np.uint8)
    joined[:img.shape[0], 0:img.shape[1]] = img
    joined[:img.shape[0], img.shape[1]:img.shape[1]*2] = thinned
    cv2.imwrite(file_prefix + str(iteration) + ".png", joined)
    while True:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        thinned = cv2.bitwise_or(thinned, temp)
        img = eroded.copy()
        iteration += 1
        joined[:img.shape[0], 0:img.shape[1]] = img
        joined[:img.shape[0], img.shape[1]:img.shape[1] * 2] = thinned
        cv2.imwrite(file_prefix + str(iteration) + ".png", joined)
        if cv2.countNonZero(img) == 0:
            break

    return thinned

def neighbours8(bounds, pos, repeat_first_last=False):
    # nhood8 = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
    rows, cols = bounds
    r, c = pos
    cup = r > 0
    crh = c < cols - 1
    cdn = r < rows - 1
    clf = c > 0

    if cup:
        yield (r - 1, c)
        if crh:
            yield (r - 1, c + 1)
    if crh:
        yield (r, c + 1)
        if cdn:
            yield (r + 1, c + 1)
    if cdn:
        yield (r + 1, c)
        if clf:
            yield (r + 1, c - 1)
    if clf:
        yield (r, c - 1)
        if cup:
            yield (r - 1, c - 1)
    if repeat_first_last and cup:
        yield (r - 1, c)

def neighbour_transitions_to_white(img, pos):
    last_value = None
    count = 0
    for neighbour in neighbours8((img.shape[0], img.shape[1]), pos, True):
        r, c = neighbour
        if last_value is None:
            last_value = img[r][c]
            continue
        count += last_value == 0 and img[r][c] != 0
        last_value = img[r][c]
    return count

def black_neighbours(img, pos):
    count = 0
    for neighbour in neighbours8((img.shape[0], img.shape[1]), pos):
        r, c = neighbour
        count += img[r][c] == 0
    return count

def hilditch(img):
    """
    Source: http://cgm.cs.mcgill.ca/~godfried/teaching/projects97/azar/skeleton.html
    :param img:
    :return: thinned image
    """
    rows, cols = (img.shape[0], img.shape[1])
    ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    temp = np.copy(img)

    # Repeat these two steps till no changes
    changed = True
    iteration = 0
    file_prefix = "./images/" + time.strftime("hilditch_%Y-%m-%d_%H-%M-%S_")
    cv2.imwrite(file_prefix + str(iteration) + ".png", img)
    while changed:
        changed = False
        # Step 1
        # for each pixel that has 8 neighbours
        for r in range(1, rows - 1):
            for c in range(1, cols - 1):
                # and is black
                if img[r][c] != 0:
                    continue

                # and 2 <= B(Pixel) <= 6
                B = black_neighbours(img, (r, c))
                if B < 2 or B > 6:
                    continue

                # and A(Pixel) = 1
                A = neighbour_transitions_to_white(img, (r, c))
                if A != 1:
                    continue

                # and P2||P4||P8||A(P2)!=1
                if img[r-1][c] == 0 and img[r][c+1] == 0 and img[r][c-1] == 0 and neighbour_transitions_to_white(img, (r - 1, c)) == 1:
                    continue

                # and P2||P4||P6||A(P4)!=1
                if img[r-1][c] == 0 and img[r][c+1] == 0 and img[r+1][c-1] == 0 and neighbour_transitions_to_white(img, (r, c+1)) == 1:
                    continue

                changed = True
                temp[r][c] = 255
        img = np.copy(temp)
        iteration += 1
        cv2.imwrite(file_prefix + str(iteration) + ".png", img)

    return img

def zhangsuen(img):
    """
    Source: http://rosettacode.org/wiki/Zhang-Suen_thinning_algorithm
    :param img:
    :return: thinned image
    """
    rows, cols = (img.shape[0], img.shape[1])
    ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    temp = np.copy(img)

    # Repeat these two steps till no changes
    changed = True
    iteration = 0
    file_prefix = "./images/" + time.strftime("zhangsuen_%Y-%m-%d_%H-%M-%S_")
    cv2.imwrite(file_prefix + str(iteration) + ".png", img)
    while changed:
        changed = False
        # Step 1
        # for each pixel that has 8 neighbours
        for r in range(1, rows - 1):
            for c in range(1, cols - 1):
                # and is black
                if img[r][c] != 0:
                    continue

                # and 2 <= B(Pixel) <= 6
                B = black_neighbours(img, (r, c))
                if B < 2 or B > 6:
                    continue

                # and A(Pixel) = 1
                A = neighbour_transitions_to_white(img, (r, c))
                if A != 1:
                    continue

                # and P2||P4||P6
                if img[r-1][c] == 0 and img[r][c+1] == 0 and img[r+1][c] == 0:
                    continue

                # and P4||P6||P8
                if img[r][c+1] == 0 and img[r+1][c] == 0 and img[r][c-1] == 0:
                    continue

                changed = True
                temp[r][c] = 255
        img = np.copy(temp)
        # Step 2
        # for each pixel that has 8 neighbours
        for r in range(1, rows - 1):
            for c in range(1, cols - 1):
                # and is black
                if img[r][c] != 0:
                    continue

                # and 2 <= B(Pixel) <= 6
                B = black_neighbours(img, (r, c))
                if B < 2 or B > 6:
                    continue

                # and A(Pixel) = 1
                A = neighbour_transitions_to_white(img, (r, c))
                if A != 1:
                    continue

                # and P2||P4||P8 <===
                if img[r-1][c] == 0 and img[r][c+1] == 0 and img[r][c-1] == 0:
                    continue

                # and ===>P2||P6||P8
                if img[r-1][c] == 0 and img[r+1][c] == 0 and img[r][c-1] == 0:
                    continue

                changed = True
                temp[r][c] = 255
        img = np.copy(temp)
        iteration += 1
        cv2.imwrite(file_prefix + str(iteration) + ".png", img)

    return img

class BFCell:
    """Brushfire Cell"""
    def __init__(self, r, c, id, occupied):
        """BFCell(row, col)"""
        self.r = r
        self.c = c
        self.id = id
        self.occupied = occupied

    def __repr__(self):
        return str(self)

    def __str__(self):
        #return "(%d, %d)" % (self.r, self.c)
        return "(%d)" % (self.id)

class BFCounter:
    def __init__(self):
        self.count = 0

    def i(self):
        orig = self.count
        self.count += 1
        return orig

def brushfire(img):
    """
    :param img:
    :return: Output Image
    """
    WALL = 255
    SPACE = 255 - WALL

    colours = BFCounter()

    VORONOI = colours.i()
    LEFT = colours.i()
    RIGHT = colours.i()
    UP =  colours.i()
    DOWN =  colours.i()

    CV = BFCell(-1, -1, -1, False) # Voronoi
    CL = BFCell(-1, -1, -2, True) # Left
    CR = BFCell(-1, -1, -3, True) # Right
    CU = BFCell(-1, -1, -4, True) # Up
    CD = BFCell(-1, -1, -5, True) # Down

    rows, cols = (img.shape[0], img.shape[1])
    ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    regions = UnionFind()
    cells = [[BFCell(r, c, r * cols + c) for c in range(cols)] for r in range(rows)]
    cellsf = [cell for row in cells for cell in row]
    regions.insert_objects(itertools.chain(cellsf, (CV, CL, CR, CU, CD)))

    visited = set()

    # Add the border cells to a set
    for r in range(rows):
        pass

    return img

process = False

def mouse_callback(event, x, y, flags, param):
    global img_i, down, last_pos, last_time, process
    if event == cv2.EVENT_RBUTTONDOWN:
        #img_i = np.zeros((R, C), np.uint8)
        process = True
    elif event == cv2.EVENT_LBUTTONDOWN:
        down = True
        last_pos = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        down = False
        last_pos = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE:
        if down:
            cv2.line(img_i, last_pos, (x, y), 255, 5)
            last_pos = (x, y)
            last_time = time.time()

cv2.setMouseCallback("main", mouse_callback)

edges = []

img_o = np.copy(img_i)

# iterr = None

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    # if (time.time() - last_time) > 1:
    #     last_time = time.time()
    #     del edges[:]
    if process:
        process = False
        #img_o = hilditch(img_i)
        img_o = zhangsuen(img_i)
        #img_o = brushfire(img_i)
    #     iterr = zhangsuen(img_i)
    #     for edge in edges:
    #         cv2.line(img_o, edge[0], edge[1], 127, 1)
    # if iterr is not None:
    #     try:
    #         img_o = iterr.next()
    #     except:
    #         iterr = None

    combined = np.zeros((img_i.shape[0], img_i.shape[1]*2), np.uint8)
    combined[:img_i.shape[0], :img_i.shape[1]] = img_i
    combined[:img_i.shape[0], img_i.shape[1]:img_i.shape[1]*2] = img_o
    cv2.imshow("main", combined)
