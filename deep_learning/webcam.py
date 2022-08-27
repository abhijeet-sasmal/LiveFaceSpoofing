import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd

data = []


class_names = ['closed', 'open']

def predict(img, model):
    img = cv2.resize(img, (224,224))
    img_array = tf.expand_dims(img, 0)
    img_array = img_array/255
    predictions = model.predict(img_array)
    # print(predictions)
    return class_names[np.argmax(predictions)]
    

def isBlinking(history, maxFrames):
    if not '1' in history:
        return False

    if not '0' in history:
        return False

    count0 = 0
    count1 = 0
    print(len(history))
    # for i in range(len(history)-1, len(history)-maxFrames, -1):
    #     # print(i)
    #     # if '0' in history:
    #     #     return True
    #     if history[i] == '0':
    #         count0 += 1    
    #     if history[i] == '1':
    #         count1 += 1    
    # if count0 > count1:
    #     return True
    # print("0 :"+str(count0))
    # print("1 :"+str(count1))

    for i in range(maxFrames):
        pattern = '1' + '0'*(i+1) + '1'
        if pattern in history:
            return True
    
    return False

import os

def collectImages():
    face_cascPath = 'Dataset/dataset1/haarcascade_frontalface_alt.xml'
    face_detector = cv2.CascadeClassifier(face_cascPath)
    # name = "Abhijeet"
    name = input("Enter your name: ")
    os.makedirs('Dataset/faces/'+name, exist_ok=True)
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
            filename = 'Dataset/faces/'+name+'/'+str(count)+'.jpg'
            cv2.imwrite(filename, face)
            # print(count)
        cv2.imshow("Eye-Blink based Liveness Detction for Facial Recognition", frame)
        if count > 200:
            cv2.destroyAllWindows()
            break


from PIL import Image

def trainFaces():
    names = os.listdir('Dataset/faces/')
    faces = []
    id = []
    count = 0
    for x in names:
        dir =  os.listdir('Dataset/faces/'+x)
        for y in dir:
            img = Image.open('Dataset/faces/'+x+'/'+y).convert('L')
            faces.append(np.array(img, 'uint8'))
            id.append(count)
        count += 1
        print(count)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(id))
    recognizer.save('Dataset/trained_faces.yml')



def process_and_display():
    face_cascPath = 'Dataset/dataset1/haarcascade_frontalface_alt.xml'

    open_eye_cascPath = 'Dataset/dataset1/haarcascade_eye_tree_eyeglasses.xml'
    left_eye_cascPath = 'Dataset/dataset1/haarcascade_lefteye_2splits.xml'
    right_eye_cascPath ='Dataset/dataset1/haarcascade_righteye_2splits.xml'
    # dataset = 'faces'

    face_detector = cv2.CascadeClassifier(face_cascPath)
    open_eyes_detector = cv2.CascadeClassifier(open_eye_cascPath)
    left_eye_detector = cv2.CascadeClassifier(left_eye_cascPath)
    right_eye_detector = cv2.CascadeClassifier(right_eye_cascPath)

    
    model = load_model('model/eye_status_classifier1.h5')
    eyes_detected = ['']*len(os.listdir('Dataset/faces')) 
    frames = cv2.VideoCapture(0)
    # frames.set(cv2.CAP_PROP_FPS, 10)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('Dataset/trained_faces.yml')

    while True:
        ret, frame = frames.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(50, 50),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        for (x,y,w,h) in faces:
            face = frame[y:y+h,x:x+w]
            gray_face = gray[y:y+h,x:x+w]
            color = (0,255,0)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

            # print(open_eyes_detector)
            id = 0
            id, conf = recognizer.predict(gray_face)

            if (conf < 20 ):
                print("Hi")
                continue

            lb = os.listdir('Dataset/faces')
            
            print("ID: ",lb[id])
            print("Conf: ",conf)

            open_eyes_glasses = open_eyes_detector.detectMultiScale(
                gray_face
            )


            # print(eyes_detected)

            if len(open_eyes_glasses) == 2:
                print("Processing")
                eyes_detected += '1'
                for (ex,ey,ew,eh) in open_eyes_glasses:
                    cv2.rectangle(face,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

                left_face = frame[y:y+h, x+int(w/2):x+w]
                left_face_gray = gray[y:y+h, x+int(w/2):x+w]

                right_face = frame[y:y+h, x:x+int(w/2)]
                right_face_gray = gray[y:y+h, x:x+int(w/2)]

                left_eye = left_eye_detector.detectMultiScale(
                    left_face_gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30),
                    flags = cv2.CASCADE_SCALE_IMAGE
                )

                right_eye = right_eye_detector.detectMultiScale(
                    right_face_gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30),
                    flags = cv2.CASCADE_SCALE_IMAGE
                )

                eye_status='1'
                for (ex,ey,ew,eh) in right_eye:
                    color = (0,255,0)
                    (row, col) = right_face.shape[0:2]
 
                    # Take the average of pixel values of the BGR Channels
                    # to convert the colored image to grayscale image
                    right_face_gray_dim = right_face
                    for i in range(row):
                        for j in range(col):
                            # Find the average of the BGR pixel values
                            right_face_gray_dim[i, j] = sum(right_face_gray_dim[i, j]) * 0.33
                    # print(right_face_gray_dim.shape)
                    pred = predict(right_face_gray_dim[ey:ey+eh,ex:ex+ew],model)
                    if pred == 'closed':
                        eye_status='0'
                        color = (0,0,255)
                    cv2.rectangle(right_face,(ex,ey),(ex+ew,ey+eh),color,2)
                for (ex,ey,ew,eh) in left_eye:

                    (row, col) = left_face.shape[0:2]
 
                    # Take the average of pixel values of the BGR Channels
                    # to convert the colored image to grayscale image
                    left_face_gray_dim = left_face
                    for i in range(row):
                        for j in range(col):
                            # Find the average of the BGR pixel values
                            left_face_gray_dim[i, j] = sum(left_face_gray_dim[i, j]) * 0.33
                    color = (0,255,0)
                    pred = predict(left_face_gray_dim[ey:ey+eh,ex:ex+ew],model)
                    if pred == 'closed':
                        eye_status='0'
                        color = (0,0,255)
                    cv2.rectangle(left_face,(ex,ey),(ex+ew,ey+eh),color,2)
                eyes_detected[id] += eye_status
                

            # print(eyes_detected)
            # print("------------------")
            # If yes, we display its name
            if len(eyes_detected[id]) < 3:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                # Display name
                y = y - 15 if y - 15 > 15 else y + 15
                cv2.putText(frame, 'Processing if '+lb[id]+' is real or fake', (x, y), cv2.FONT_HERSHEY_SIMPLEX,0.75, (255, 0, 0), 2)
            


            if (len(eyes_detected[id]) > 20):
                if isBlinking(eyes_detected[id],3):
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    # Display name
                    y = y - 15 if y - 15 > 15 else y + 15
                    cv2.putText(frame, lb[id]+' Real Image: ', (x, y), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 255, 0), 2)
                    # eyes_detected = '111'
                    print()

                else:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    # Display name
                    y = y - 15 if y - 15 > 15 else y + 15
                    cv2.putText(frame, lb[id]+' Spoofed Image: ', (x, y), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 0, 255), 2, cv2.LINE_AA)


                print(eyes_detected[id])
                data.append(eyes_detected)
                # d = pd.DataFrame([])
            
            if(len(eyes_detected[id]) > 30):
                eyes_detected = ['']*len(os.listdir('Dataset/faces'))
                print('cleared')

        cv2.imshow("Eye-Blink based Liveness Detection for Facial Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

# collectImages()
# trainFaces()
process_and_display()