import cv2
import numpy as np

# Line Segment Detector
img = cv2.imread("./img/rectangles.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

lsd = cv2.createLineSegmentDetector(0)
lines = lsd.detect(gray)[0]

for l in lines:
    (x1, y1, x2, y2) = l[0]
    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)

cv2.imwrite("./img/rectangles_LSD.jpg", img)

# Hough Algorithm
img = cv2.imread('./img/rectangles.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, 100, 200, apertureSize=3)
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 15, minLineLength=10, maxLineGap=10)

for i in range(0, len(lines)):
    for x1, y1, x2, y2 in lines[i]:
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)

cv2.imwrite("./img/rectangles_Hough.jpg", img)
