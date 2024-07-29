import os
import time
import cv2
import numpy as np
from PIL import Image
from threading import Thread


def getImagesAndLabels(path):
    # path of all the files in the folder
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    # empty ID list
    Ids = []
    for imagePath in imagePaths:
        pilImage = Image.open(imagePath).convert('L')
        imageNp = np.array(pilImage, 'uint8')
        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces.append(imageNp)
        Ids.append(Id)
    return faces, Ids


# def TrainImages():
#     recognizer = cv2.face.LBPHFaceRecognizer_create()
#     harcascadePath = "haarcascade_default.xml"
#     detector = cv2.CascadeClassifier(harcascadePath)
#     faces, Id = getImagesAndLabels("TrainingImage")
#     Thread(target = recognizer.train(faces, np.array(Id))).start()
#     Thread(target = counter_img("TrainingImage")).start()
#     recognizer.save("TrainingImageLabel"+os.sep+"Trainner.yml")
#     print("All Images")

def TrainImages():
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
    except AttributeError:
        recognizer = cv2.face.LBPHFaceRecognizer.create()
    
    harcascadePath = "haarcascade_default.xml"
    if not os.path.exists(harcascadePath):
        print("Error: Haarcascade file not found")
        return
    
    detector = cv2.CascadeClassifier(harcascadePath)
    if detector.empty():
        print("Error: Failed to load Haarcascade classifier")
        return
    
    faces, Id = getImagesAndLabels("TrainingImage")
    if len(faces) == 0 or len(Id) == 0:
        print("No images found or no labels provided")
        return
    
    recognizer.train(faces, np.array(Id))
    
    # Run counter_img in a separate thread if necessary
    counter_thread = Thread(target=counter_img, args=("TrainingImage",))
    counter_thread.start()
    
    if not os.path.exists("TrainingImageLabel"):
        os.makedirs("TrainingImageLabel")
    
    recognizer.save(os.path.join("TrainingImageLabel", "Trainer.yml"))
    print("All Images trained and saved")


def counter_img(path):
    imgcounter = 1
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    for imagePath in imagePaths:
        print(str(imgcounter) + " Images Trained", end="\r")
        time.sleep(0.008)
        imgcounter += 1