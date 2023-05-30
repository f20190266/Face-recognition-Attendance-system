import numpy as np
import cv2
import os
import time as t

if not os.path.exists('images'):
os.makedirs('images')

face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
capture = cv2.VideoCapture(0)
capture.set(3,640)
capture.set(4,480)
count = 0
face_detector =
cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

faceid = input('\n Enter your Roll no : ')
print("\n Initializing face capture. Look the camera and wait ....")

while(True):
ret, img = capture.read()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
count += 1

cv2.imwrite("./images/Users." + str(faceid) + '.' + str(count) + ".jpg",
gray[y:y+h,x:x+w])
cv2.imshow('image', img)

k = cv2.waitKey(100) & 0xff
if k < 90:
break
elif count >= 90:
break
t.sleep(3)

while(True):
ret, img = capture.read()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
count += 1

cv2.imwrite("./images/Users." + str(faceid) + '.' + str(count) + ".jpg",
gray[y:y+h,x:x+w])
cv2.imshow('image', img)

k = cv2.waitKey(100) & 0xff
if k < 60:
break
elif count >= 60:

break
t.sleep(3)

while(True):
ret, img = capture.read()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
count += 1

cv2.imwrite("./images/Users." + str(faceid) + '.' + str(count) + ".jpg",
gray[y:y+h,x:x+w])
cv2.imshow('image', img)

k = cv2.waitKey(100) & 0xff
if k < 90:
break
elif count >= 90:
break

print("\n Exiting Program......")
capture.release()
cv2.destroyAllWindows()
