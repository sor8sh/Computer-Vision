import cv2

# Stadium 1
img = cv2.imread('./img/Stadium1.jpg', 3)
final = cv2.imread('./img/Stadium1.jpg', 0)

w, h = img.shape[0], img.shape[1]

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if img[i][j][1] > img[i][j][0] and img[i][j][1] > img[i][j][2]:
            final[i][j] = 255
        else:
            final[i][j] = 0

_, labels = cv2.connectedComponents(final)

print(labels[int(w / 2)][int(h / 2)])

for i in range(w):
    for j in range(h):
        if labels[i][j] == labels[int(w / 2)][int(h / 2)]:
            final[i][j] = 255
        else:
            final[i][j] = 0

cv2.imwrite('./img/Stadium1_Grass.jpg', final)

# Stadium 2
img = cv2.imread('./img/Stadium2.png', 3)
final = cv2.imread('./img/Stadium2.png', 0)

w, h = img.shape[0], img.shape[1]

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if img[i][j][1] > img[i][j][0] and img[i][j][1] > img[i][j][2]:
            final[i][j] = 255
        else:
            final[i][j] = 0

_, labels = cv2.connectedComponents(final)

print(labels[int(h / 2)][int(w / 2)])

for i in range(w):
    for j in range(h):
        if labels[i][j] == labels[int(h / 2)][int(w / 2)]:
            final[i][j] = 255
        else:
            final[i][j] = 0

cv2.imwrite('./img/Stadium2_Grass.png', final)
