import numpy as np
import cv2
from matplotlib import pyplot as plt


def otsu(image):
    history = []
    threshold = 0
    best_sigma = np.var(image) * image.shape[0] * image.shape[1]
    for i in range(256):
        class1 = []
        class2 = []

        for j in image.flatten():
            if j < i:
                class1.append(j)
            else:
                class2.append(j)

        temp = np.var(class1) * len(class1) + np.var(class2) * len(class2)
        history.append(temp)

        if temp < best_sigma:
            threshold = i
            best_sigma = temp

    final_img = image.copy()
    final_img[image > threshold] = 255
    final_img[image < threshold] = 0
    return final_img, history


img = cv2.imread('./img/redBall.png', 0)
newImg, sigma_lst = otsu(img)
row = []
for i in range(256):
    row.append(i)

plt.plot(row, sigma_lst, 'r--')
plt.title('Sigma'), plt.xticks([]), plt.yticks([])
plt.savefig('./img/redBall_Sigma.png')
plt.close()

plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(newImg, cmap='gray')
plt.title('Otsu'), plt.xticks([]), plt.yticks([])
plt.savefig('./img/redBall_Otsu.png')
