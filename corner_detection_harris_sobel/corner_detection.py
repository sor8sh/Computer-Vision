import cv2
import numpy as np

img = cv2.imread('./img/1.jpg')
gray = np.float32(cv2.imread('./img/1.jpg', 0))
w, h = img.shape[0], img.shape[1]

sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

Ix2 = sobel_x * sobel_x
Iy2 = sobel_y * sobel_y
Ixy = sobel_x * sobel_y
gIx2 = cv2.GaussianBlur(Ix2, (3, 3), 0)
gIy2 = cv2.GaussianBlur(Iy2, (3, 3), 0)
gIxy = cv2.GaussianBlur(Ixy, (3, 3), 0)

m = []
dst = np.copy(gray)
for i in range(w):
    for j in range(h):
        m = np.array([[gIx2[i][j], gIxy[i][j]],
                      [gIxy[i][j], gIy2[i][j]]])
        dst[i][j] = np.linalg.det(m) - 0.04 * np.trace(m)

val = 0.03 * dst.max()
for i in range(w):
    for j in range(h):
        if dst[i][j] > val:
            img[i][j] = [0, 0, 255]

cv2.imwrite('./img/1-corners.jpg', img)
