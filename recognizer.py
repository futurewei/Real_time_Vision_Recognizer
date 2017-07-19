import numpy as np
import cv2
import threading
from similarity import compute_Similarity
from PIL import Image
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from dataProcess import check
import time
import os 

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
book_cascade = cv2.CascadeClassifier('bookcascade.xml')
tLock = threading.Lock()


cap = cv2.VideoCapture(0)
counter =0
obj = 0
obj_similarity = False
similarity = 0
if cap.isOpened():
  while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    book = book_cascade.detectMultiScale(gray, 30, 30)
    for (x,y,w,h) in book:
        name = name = 'obj' + str(obj) +'.png'
        crop_img = img[y:y+250, x:x+250]
        cv2.imwrite(name,crop_img)
        obj = obj+1
        cv2.rectangle(img,(x,y),(x+200,y+200),(255,255,0),2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, check(name), (10,200), font, 2, (0,0,255), 2, cv2.LINE_AA)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        crop_img = img[y:y+h, x:x+w] # Crop from x, y, w, h -> 100, 200, 300, 400
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            tLock.acquire()
            counter = counter+1
            name = 'person' + str(counter) +'.png'
            cv2.imwrite(name,crop_img)
            tLock.release()
            tLock.acquire()
            similarity = compute_Similarity(name)
            tLock.release()
            os.remove(name)
    if similarity >=0.75:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img,'Lai Wei, accuracy: '+str(similarity),(10,500), font, 2, (200,255,155), 2, cv2.LINE_AA)
    cv2.imshow('img',img)
    #cv2.imwrite('person.png',img)
    k = cv2.waitKey(33)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()