import cv2
import numpy as np
from matplotlib import pyplot as plt

# Gaussian picture 1
img = cv2.imread('./img/1.jpg', 0)
img_dft = cv2.dft(np.float32(img), flags=cv2.DFT_REAL_OUTPUT)

dst = np.copy(img_dft)
h, w = img.shape
for i in range(h):
    for j in range(w):
        dst[i][j] = img_dft[i][j] * (np.e ** (-1 * (i ** 2 + j ** 2) / (2 * (70 ** 2))))

dst = cv2.idft(dst)

plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(dst, cmap='gray')
plt.title('Gaussian Filter'), plt.xticks([]), plt.yticks([])
plt.savefig('./img/1_gaussian.jpg')

# Gaussian picture 2
img = cv2.imread('./img/2.jpg', 0)
img_dft = cv2.dft(np.float32(img), flags=cv2.DFT_REAL_OUTPUT)

dst = np.copy(img_dft)
for i in range(h):
    for j in range(w):
        dst[i][j] = img_dft[i][j] * (np.e ** (-1 * (i ** 2 + j ** 2) / (2 * (70 ** 2))))

dst = cv2.idft(dst)

plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(dst, cmap='gray')
plt.title('Gaussian Filter'), plt.xticks([]), plt.yticks([])
plt.savefig('./img/2_gaussian.jpg')

# Mask picture 1
img = cv2.imread('./img/1.jpg', 0)

dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
c_row, c_col = int(h / 2), int(w / 2)

mask = np.zeros((h, w, 2), np.uint8)
margin = 30
mask[c_row - margin:c_row + margin, c_col - margin:c_col + margin] = 1

img_back = dft_shift * mask
img_back = np.fft.ifftshift(img_back)
img_back = cv2.idft(img_back)
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(img_back, cmap='gray')
plt.title('Mask Filter'), plt.xticks([]), plt.yticks([])
plt.savefig('./img/1_mask.jpg')

# Mask picture 1
img = cv2.imread('./img/2.jpg', 0)

dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

mask = np.zeros((h, w, 2), np.uint8)
margin = 30
mask[c_row - margin:c_row + margin, c_col - margin:c_col + margin] = 1

img_back = dft_shift * mask
img_back = np.fft.ifftshift(img_back)
img_back = cv2.idft(img_back)
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(img_back, cmap='gray')
plt.title('Mask Filter'), plt.xticks([]), plt.yticks([])
plt.savefig('./img/2_mask.jpg')
