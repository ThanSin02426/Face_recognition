import cv2
import os
import numpy as np
from PIL import Image

path = 'dataset'
recognizer = cv2.face.LBPHFaceRecognizer_create()
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    face_samples = []
    ids = []

    for image_path in image_paths:
        img_pil = Image.open(image_path).convert('L') 
        img_numpy = np.array(img_pil, 'uint8')
        _id = int(os.path.split(image_path)[-1].split(".")[1])
        faces = face_detector.detectMultiScale(img_numpy)

        for (x, y, w, h) in faces:
            face_samples.append(img_numpy[y:y+h, x:x+w])
            ids.append(_id)

    return face_samples, ids

print("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces, ids = get_images_and_labels(path)
recognizer.train(faces, np.array(ids))

recognizer.write('trainer.yml')

print("\n [INFO] {0} faces trained. Exiting Program.".format(len(np.unique(ids))))
