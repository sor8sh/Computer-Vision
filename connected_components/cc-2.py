import cv2

img = cv2.imread('./img/shapes.png', 0)
w, h = img.shape[0], img.shape[1]


def check_neighbor(image, x, y, q):
    for i in range(x - 1, x + 2):
        for j in range(y - 1, y + 2):
            try:
                if image[i][j][0] != 0 and not image[i][j][2]:
                    q.append((i, j))
            except:
                pass
    return q


def connected_component(image):
    cc = []
    for i in range(w):
        lst = []
        for j in range(h):
            lst.append([image[i][j], 0, False])
        cc.append(lst)

    counter = 0
    queue = []
    for i in range(w):
        for j in range(h):
            if cc[i][j][0] != 0 and not cc[i][j][2]:
                queue.append((i, j))
                counter += 1

            while queue:
                nxt = queue.pop()
                cc[nxt[0]][nxt[1]][1] = counter
                cc[nxt[0]][nxt[1]][2] = True
                queue = check_neighbor(cc, nxt[0], nxt[1], queue)
    return cc, counter


cc, labels = connected_component(img)
print("# of connected components: ", labels)

holes = img.copy()
for i in range(w):
    for j in range(h):
        if holes[i][j] != 0:
            holes[i][j] = 0
        else:
            holes[i][j] = 255

not_cc, not_counter = connected_component(holes)
print("# of holes: ", not_counter - 1)

holed_shape = []
for i in range(w):
    for j in range(h):
        if cc[i][j][1] != 0 and cc[i][j + 1][1] == 0:
            for k in range(j + 1, h):
                if cc[i][k][1] == cc[i][j][1] and cc[i][j][1] not in holed_shape:
                    holed_shape.append(cc[i][j][1])

print("# of shapes with hole: ", len(holed_shape))

circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 5, param1=30, param2=28)
print("# of squares: ", labels - len(circles[0]))
print("# of circles: ", len(circles[0]))
