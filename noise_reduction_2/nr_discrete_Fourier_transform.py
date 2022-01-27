import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('./img/striping.bmp', 0)
height, width = img.shape

dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

for i in range(height):
    total_sum = 0
    for j in range(width):
        total_sum += magnitude_spectrum[i][j]
    if total_sum / width >= 210:
        for j in range(width):
            magnitude_spectrum[i - 2][j] = 0
            magnitude_spectrum[i - 1][j] = 0
            magnitude_spectrum[i][j] = 0
            magnitude_spectrum[i + 1][j] = 0
            magnitude_spectrum[i + 2][j] = 0

for i in range(height):
    total_sum = 0
    for j in range(width):
        if i == height - 1 or j == width - 1:
            continue
        total_sum += magnitude_spectrum[j][i]
    if total_sum / height >= 210:
        for j in range(width - 1):
            magnitude_spectrum[j][i - 2] = 0
            magnitude_spectrum[j][i - 1] = 0
            magnitude_spectrum[j][i] = 0
            magnitude_spectrum[j][i + 1] = 0
            magnitude_spectrum[j][i + 2] = 0

rows, cols = img.shape

# create a mask first, center square is 1, remaining all zeros
mask = np.zeros((rows, cols, 2), np.uint8)
for i in range(height):
    for j in range(width):
        mask[i][j] = magnitude_spectrum[i][j]

# apply mask and inverse DFT
f_shift = dft_shift * mask
f_i_shift = np.fft.ifftshift(f_shift)
img_back = cv2.idft(f_i_shift)
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

plt.subplot(131), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Image Frequency'), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(img_back, cmap='gray')
plt.title('Reconstructed Image'), plt.xticks([]), plt.yticks([])
plt.show()
