import cv2
import os

if not os.path.exists('dataset'):
    os.makedirs('dataset')

# Load the Haar cascade for face detection
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

face_id = input('\n Enter user id and press <return> ==> ')
cam = cv2.VideoCapture(0)
print("\n [INFO] Initializing face capture. Look at the camera and wait ...")
count = 0

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        count += 1
        cv2.imwrite(f"dataset/user.{face_id}.{count}.jpg", gray[y:y+h, x:x+w])

    cv2.imshow('image', img)
    k = cv2.waitKey(100) & 0xff
    if k == 27:
        break
    elif count >= 10:  
        break

print("\n [INFO] Exiting Program.")
cam.release()
cv2.destroyAllWindows()

success = cv2.imwrite("dataset/user.{}.{}.jpg".format(face_id, count), gray[y:y+h, x:x+w])
if success:
    print(f"Saved image: dataset/user.{face_id}.{count}.jpg")
else:
    print(f"Failed to save image: dataset/user.{face_id}.{count}.jpg")
count += 1