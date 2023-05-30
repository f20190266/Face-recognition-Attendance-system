import cv2
import numpy as np
from PIL import Image
import os

path = './images/'
recognizer = cv2.face.LBPHFaceRecognizer_create()

detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

def getimglbl(path):
image = [os.path.join(path,f) for f in os.listdir(path)]
facedb = []
ids = []
for imagePath in image:

21

grayimg = Image.open(imagePath).convert('L')
img_numpy = np.array(grayimg,'uint8')
id = int(os.path.split(imagePath)[-1].split(".")[1])
faces = detector.detectMultiScale(img_numpy)
for (x,y,w,h) in faces:
facedb.append(img_numpy[y:y+h,x:x+w])
ids.append(id)
return facedb,ids

print ("\n Training faces.....")
faces,ids = getimglbl(path)
recognizer.train(faces, np.array(ids))

recognizer.write('trainer.yml')
print("\n {0} faces trained. Exiting Program ......".format(len(np.unique(ids))))
