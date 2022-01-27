import cv2
from matplotlib import pyplot as plt

img = cv2.imread('./img/doc_shadow.png', 0)

_, global_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
local_img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 100)

plt.imshow(local_img, cmap='gray')
plt.title('Adaptive Threshold'), plt.xticks([]), plt.yticks([])
plt.show()
