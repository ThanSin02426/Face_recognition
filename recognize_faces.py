import cv2
import numpy as np
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

names = ['None', 'User1', 'User2'] 

cam = cv2.VideoCapture(0)

print("\n [INFO] Starting face recognition.")
while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        _id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

        if  confidence < 80 :
            name = names[_id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            name = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))

        cv2.putText(img, name, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
        cv2.putText(img, str(confidence), (x + 5, y + h - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)

    cv2.imshow('camera', img)
    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break

print("\n [INFO] Exiting Program.")
cam.release()
cv2.destroyAllWindows()