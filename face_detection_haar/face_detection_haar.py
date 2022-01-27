import cv2

face_cascade = cv2.CascadeClassifier('./haar/face.xml')
eye_cascade = cv2.CascadeClassifier('./haar/eye.xml')
mouth_cascade = cv2.CascadeClassifier('./haar/mouth.xml')
img = cv2.imread('./img/face.jpeg')
gray = cv2.imread('./img/face.jpeg', 0)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (face_x, face_y, face_w, face_h) in faces:
    cv2.rectangle(img, (face_x, face_y), (face_x + face_w, face_y + face_h), (255, 0, 0), 2)
    local_face_gray = gray[face_y:face_y + face_h, face_x:face_x + face_w]
    local_face_color = img[face_y:face_y + face_h, face_x:face_x + face_w]

    eyes = eye_cascade.detectMultiScale(local_face_gray)
    for (eye_x, eye_y, eye_w, eye_h) in eyes:
        cv2.rectangle(local_face_color, (eye_x, eye_y), (eye_x + eye_w, eye_y + eye_h), (0, 255, 0), 2)

    mouth = mouth_cascade.detectMultiScale(local_face_gray, 1.7, 11)
    if len(mouth):
        mouth_x, mouth_y, mouth_w, mouth_h = mouth[0]
        cv2.rectangle(local_face_color, (mouth_x, mouth_y), (mouth_x + mouth_w, mouth_y + mouth_h), (0, 0, 255), 2)

cv2.imwrite('./img/face_detected.jpg', img)
