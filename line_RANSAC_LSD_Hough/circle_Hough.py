import numpy as np
import cv2

img = cv2.imread("./img/Balls.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 2, 100)
for c in circles[0]:
    c[2] = np.round(c[2])

for (x, y, r) in circles[0]:
    cv2.circle(img, (x, y), r, (0, 0, 0), 2)
    cv2.circle(img, (x, y), 2, (0, 0, 0), 2)

cv2.imwrite("./img/Balls_Output.jpg", img)
