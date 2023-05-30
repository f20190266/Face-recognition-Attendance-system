import cv2
import numpy as np
import time as t
import os

recog = cv2.face.LBPHFaceRecognizer_create()
recog.read('trainer.yml')

face_cascade_Path = "haarcascade_frontalface_default.xml"

face = cv2.CascadeClassifier(face_cascade_Path)

font = cv2.FONT_HERSHEY_SIMPLEX

id = 0

names = ['None','vijay','nirbhaya','kan','Teja','Sharan','VIJAY']

capture= cv2.VideoCapture(0)
capture.set(3, 640)
capture.set(4, 480)

23

minW = 0.1 * capture.get(3)
minH = 0.1 * capture.get(4)
while True:
ret, img = capture.read()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face.detectMultiScale(
gray,
scaleFactor=1.3,
minNeighbors=1,
minSize=(int(minW), int(minH)),
)

for (x, y, w, h) in faces:
cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
id, accuracy = recog.predict(gray[y:y + h, x:x + w])
if (accuracy < 60):
id = names[id]
accuracy = " {0}%".format(round(100 - accuracy))
elif(accuracy>=60):

24

id = " Unrecognised Face "
accuracy = " {0}%".format(round(100 - accuracy))
#t.sleep(0.5)
cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
cv2.putText(img, str(accuracy), (x + 5, y + h - 5), font, 1, (255, 255, 0),
1)

cv2.imshow('camera', img)
# Escape to exit the webcam / program
k = cv2.waitKey(10) & 0xff
if k == 27:
break
print("\n [INFO] Exiting Program.")
capture.release()
cv2.destroyAllWindows()
