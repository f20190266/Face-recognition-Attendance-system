import numpy as np
from PIL import Image, ImageDraw
from IPython.display import display
import cv2
from cv2 import imshow
import os
import face_recognition
from datetime import datetime
import time as t

f1 = face_recognition.load_image_file("vijay.jpg")
f1_encod= face_recognition.face_encodings(f1)[0]

27

f2 = face_recognition.load_image_file("nirbhaya.jpg")
f2_encod= face_recognition.face_encodings(f2)[0]

f3 = face_recognition.load_image_file("kanishkha.jpg")
f3_encod= face_recognition.face_encodings(f3)[0]

known_face = [
f1_encod,
f2_encod,
f3_encod
]
face_names = [
"Vijay Gottipati",
"Nirbhaya Reddy",
"J.Kanishkha"
]
print("Done learning and creating profiles")

if not os.path.exists('image'):
os.makedirs('image')

28

a=(int)(input("Enter \n 1 For marking attendance \n 2 For Exit\n"))
if a!=1:
print("exiting")
while(a==1):
faceCascade =
cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap= cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
c = 0
face_detector =
cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
print("\n [INFO] Initializing face capture. Look the capera and wait ...")

while(True):
t.sleep(2)
ret, img = cap.read()

rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
f = faceCascade.detectMultiScale(g, 1.3, 5)

29

for (x,y,w,h) in f:
cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
c += 1
cv2.imwrite("Users.jpg", g[y:y+h,x:x+w])
cv2.imshow('image', img)
k = cv2.waitKey(300) & 0xff
if k < 2:
break
elif c >= 1:
break
t.sleep(2)
cap.release()
cv2.destroyAllWindows()
curr = datetime.now()
dtS = curr.strftime('%d-%b-%Y')
if not os.path.exists('attendance_list'+dtS+'.csv'):
with open('attendance_list'+dtS+'.csv', 'w') as f:
f.writelines(f'\n Name,Date,Time')
f.close()
def makeAttendanceEntry(name):
curr = datetime.now()

30

dtS= curr.strftime('%d-%b-%Y')
with open('attendance_list'+dtS+'.csv','r+') as f:
allLines = f.readlines()
attendanceList = []
for line in allLines:
entry = line.split(',')
attendanceList.append(entry[0])
if (name not in attendanceList):
curr = datetime.now()
dtS = now.strftime('%d/%b/%Y, %H:%M:%S')
f.writelines(f'\n{name},{dtString}')

fname = "Users.jpg"
unknown = face_recognition.load_image_file(fname)
udraw = cv2.imread(fname)

faceL = face_recognition.face_locations(unknown)
face_encodings = face_recognition.face_encodings(unknown, faceL)

pil_image = Image.fromarray(unknown)

31

draw = ImageDraw.Draw(pil_image)
name="Unknown"
for (top, right, bottom, left), face_encoding in zip(faceL, face_encodings):
matches = face_recognition.compare_faces(known_face,
face_encoding)
name = "Unknown"
face_distances = face_recognition.face_distance(known_face,
face_encoding)
best_match_index = np.argmin(face_distances)
if matches[best_match_index]:
name = face_names[best_match_index]
cv2.rectangle(udraw,(left, top), (right, bottom), (0,255,0),3 )
draw.rectangle(((left, top), (right, bottom)), outline=(0, 255, 255))
cv2.putText(udraw,name,(left,top-20),
cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 2,cv2.LINE_AA)
makeAttendanceEntry(name)
if name=="Unknown":
print(name)
else:
print("welcome! "+name)
display(pil_image)

32

a=(int)(input("enter \n 1 for marking attendance \n 2 for exit\n"))
if a!=1:
print("exiting")
