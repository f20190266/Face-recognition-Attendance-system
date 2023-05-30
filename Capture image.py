import numpy as np
import cv2

capture = cv2.VideoCapture(0)
face = cv2.CascadeClassifier(cv2.data.haarcascades +
'haarcascade_frontalface_default.xml')
eye = cv2.CascadeClassifier(cv2.data.haarcascades +
'haarcascade_eye.xml')

while True:

ret,box = capture.read()

grayscale = cv2.cvtColor(box, cv2.COLOR_BGR2GRAY)

11
faces = face.detectMultiScale(grayscale,1.3,5) #2nd factor to
reduce the size of our face to match the classifier , the lesser it is the
better accuracy but slower algo perform
for (x,y,w,h) in faces:
cv2.rectangle(box,(x,y),(x+w,y+h),(255,0,0),5) #3rd param colour
BGR and last param line thickness
gray = grayscale[y:y+h , x:x+w]
color = box[y:y+h , x:x+w]
eyes = eye.detectMultiScale(gray,1.3,5)
for (ex,ey,ew,eh) in eyes:
cv2.rectangle(color,(ex,ey),(ex+ew,ey+eh),(0,255,0),5)

cv2.imshow('frame', box)

if cv2.waitKey(1) == ord('x'): # press 'x' to exit the capturing screen
break
capture.release()
cv2.destroyAllWindows()
