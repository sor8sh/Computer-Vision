import numpy as np
import cv2

img = cv2.imread("./img/1.jpg", 1)
grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(grey, (3, 3), 0)
edge = cv2.Canny(blur, 0, 10)

_, contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)

doc = np.zeros([4, 2])
for c in contours:
    p = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.05 * p, True)[:5]

    if len(approx) == 4:
        doc = approx.reshape((4, 2))
        break

region = np.zeros((4, 2), dtype="float32")

s, d = doc.sum(axis=1), np.diff(doc, axis=1)
region[0], region[2] = doc[np.argmin(s)], doc[np.argmax(s)]
region[1], region[3] = doc[np.argmin(d)], doc[np.argmax(d)]

(tl, tr, br, bl) = region

d1 = ((tl[0] - tr[0]) ** 2 + (tl[1] - tr[1]) ** 2) ** 0.5
d2 = ((bl[0] - br[0]) ** 2 + (bl[1] - br[1]) ** 2) ** 0.5
width = max(int(d1), int(d2))

d1 = ((tl[0] - bl[0]) ** 2 + (tl[1] - bl[1]) ** 2) ** 0.5
d2 = ((tr[0] - br[0]) ** 2 + (tr[1] - br[1]) ** 2) ** 0.5
length = max(int(d1), int(d2))

dst = np.array([[0, 0], [width - 1, 0], [width - 1, length - 1], [0, length - 1]], dtype="float32")

m = cv2.getPerspectiveTransform(region, dst)
transform = cv2.warpPerspective(img, m, (width, length))

final = cv2.cvtColor(transform, cv2.COLOR_BGR2GRAY)
final = cv2.adaptiveThreshold(final, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 101, 10)

cv2.imwrite("Scanned_Real.jpg", transform)
cv2.imwrite("Scanned_Binary.jpg", final)
