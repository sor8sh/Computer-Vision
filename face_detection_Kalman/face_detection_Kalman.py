import cv2
import numpy as np


def normal_detect(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                          flags=cv2.CASCADE_SCALE_IMAGE)
    if len(faces) == 0:
        return 0, 0, 0, 0

    return faces[0]


face_cascade = cv2.CascadeClassifier('./face.xml')
video_capture = cv2.VideoCapture(0)

ret, frame = video_capture.read()
c, r, w, h = normal_detect(frame)

state = np.array([c + w / 2, r + h / 2, 0, 0], dtype='float64')  # initial position
kalman = cv2.KalmanFilter(4, 2, 0)
kalman.transitionMatrix = np.array([[1., 0., .1, 0.],
                                    [0., 1., 0., .1],
                                    [0., 0., 1., 0.],
                                    [0., 0., 0., 1.]])
kalman.measurementMatrix = 1. * np.eye(2, 4)
kalman.processNoiseCov = 1e-5 * np.eye(4, 4)
kalman.measurementNoiseCov = 1e-3 * np.eye(2, 2)
kalman.errorCovPost = 1e-1 * np.eye(4, 4)
kalman.statePost = state

x0, y0, c, r = c, r, w, h
while True:
    ret, frame = video_capture.read()

    prediction = kalman.predict()
    x, y, w, h = normal_detect(frame)
    if not (w < 0.1 * c):
        x0, y0, c, r = x, y, w, h
        cv2.rectangle(frame, (int(x0 - 5), int(y0 - 5)), (int(x0 + c + 5), int(y0 + r + 5)),
                      (0, 0, 255), 2)
    measurement = np.array([x + w / 2, y + h / 2], dtype='float64')

    if not (x == 0 and y == 0 and w == 0 and h == 0):
        x, y, w, h = kalman.correct(measurement)
    else:
        x, y, w, h = prediction

    cv2.rectangle(frame, (int(x - c / 2 + 5), int(y - r / 2 + 5)), (int(x + c / 2 - 5), int(y + r / 2 - 5)),
                  (0, 255, 0), 2)
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
