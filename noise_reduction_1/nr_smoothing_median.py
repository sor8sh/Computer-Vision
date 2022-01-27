import cv2
from matplotlib import pyplot as plt

pic5 = cv2.imread('./img/5.jpg')
pic6 = cv2.imread('./img/6.jpg')

smooth5 = cv2.blur(pic5, (3, 3))
median5 = cv2.medianBlur(pic5, 3)
smooth6 = cv2.blur(pic6, (3, 3))
median6 = cv2.medianBlur(pic6, 3)

plt.subplot(231), plt.imshow(pic5), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(232), plt.imshow(smooth5), plt.title('Smoothing')
plt.xticks([]), plt.yticks([])
plt.subplot(233), plt.imshow(median5), plt.title('Median')
plt.xticks([]), plt.yticks([])

plt.subplot(234), plt.imshow(pic6)
plt.xticks([]), plt.yticks([])
plt.subplot(235), plt.imshow(smooth6)
plt.xticks([]), plt.yticks([])
plt.subplot(236), plt.imshow(median6)
plt.xticks([]), plt.yticks([])

plt.show()

cv2.imwrite('./img/5-smooth.jpg', smooth5)
cv2.imwrite('./img/5-median.jpg', median5)
cv2.imwrite('./img/6-smooth.jpg', smooth6)
cv2.imwrite('./img/6-median.jpg', median6)
