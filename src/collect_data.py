import cv2
import os
import hydra
from omegaconf import DictConfig


def collectImages(config: DictConfig, name):
    face_cascPath = config.haar.frontalface
    print(face_cascPath)
    face_detector = cv2.CascadeClassifier(face_cascPath)
    os.makedirs(config.data.processed+'/faces/'+name, exist_ok=True)
    frames = cv2.VideoCapture(0)
    count = 0
    while True:
        ret, frame = frames.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(50, 50),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        for (x,y,w,h) in faces:
            count += 1
            face = gray[y:y+h,x:x+w]
            filename = f'{config.data.processed}/faces/'+name+'/'+str(count)+'.jpg'
            cv2.imwrite(filename, face)
        cv2.imshow("Eye-Blink based Liveness Detction for Facial Recognition", frame)
        if count > 200:
            cv2.destroyAllWindows()
            break