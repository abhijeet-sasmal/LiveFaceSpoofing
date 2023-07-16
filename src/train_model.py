"""
This is the demo code that uses hy                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      dra to access the parameters in under the directory config.

Author: Khuyen Tran
"""

import hydra
from omegaconf import DictConfig
import os
from PIL import Image
import cv2
import numpy as np

def trainFaces(config: DictConfig):
    names = os.listdir(config.data.processed)
    faces = []
    id = []
    count = 0
    for x in names:
        dir =  os.listdir(config.data.processed+'/'+x)
        for y in dir:
            img = Image.open(config.data.processed+'/'+x+'/'+y).convert('L')
            faces.append(np.array(img, 'uint8'))
            id.append(count)
        count += 1
        print(count)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(id))
    recognizer.save(config.model+'trained_faces.yml')




