import cv2
import numpy as np

# Histogram stretching
stretchPath = './img/1.png'
stretchImg = cv2.imread(stretchPath, cv2.IMREAD_GRAYSCALE)
minValue = np.min(stretchImg)
maxValue = np.max(stretchImg)

newStretchImg = np.zeros(stretchImg.shape)

stretch = 255 / (maxValue - minValue)

for i in range(stretchImg.shape[0]):
    for j in range(stretchImg.shape[1]):
        newStretchImg[i][j] = (stretchImg[i][j] - minValue) * stretch

cv2.imwrite('./img/1-out.png', newStretchImg)

# Histogram clipping
clipPath = './img/2.jpg'
clipImg = cv2.imread(clipPath, cv2.IMREAD_GRAYSCALE)
minValue = np.min(clipImg)
maxValue = np.max(clipImg)
clipArr = np.sort(np.reshape(clipImg, [1, clipImg.shape[0] * clipImg.shape[1]]))

newClipImg = np.zeros(clipImg.shape)

bound = int(0.01 * len(clipArr[0]))
clip = 255 / (clipArr[0][-bound] - clipArr[0][bound])

for i in range(clipImg.shape[0]):
    for j in range(clipImg.shape[1]):
        if clipImg[i][j] >= clipArr[0][-bound]:
            newClipImg[i][j] = 255
            continue
        if clipImg[i][j] <= clipArr[0][bound]:
            newClipImg[i][j] = 0
            continue
        newClipImg[i][j] = (clipImg[i][j] - clipArr[0][bound]) * clip

cv2.imwrite('./img/2-out.jpg', newClipImg)
