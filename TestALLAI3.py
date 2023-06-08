# Face_de
import face_recognition
import cv2
import time

#Fast Ai
from fastbook import *
from glob import glob
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
learn_inf = load_learner('eay_data_resnet18_fastai\eay_data_resnet18_fastai.pkl')

def eye(roi_gray,roi_color) :
    # Detects eyes 
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor = 1.1, minNeighbors = 10)
        #eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor = 1.1, minNeighbors = 10)
  
        #draw a rectangle in eyes 
        for (ex,ey,ew,eh) in eyes: 
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,127,255),2)
        
        eye_images = []
        for (ex, ey, ew, eh) in eyes:
            eye_image = roi_color[ey:ey+eh, ex:ex+ew]
            eye_images.append(eye_image)

        #   ใช้ eye_images ในโมเดลอื่น
        for eye_image in eye_images:
            re = learn_inf.predict(eye_image)
            return re

def mouth(roi_gray,roi_color) :
      # Detects mouth 
        #mouth = mouth_cascade.detectMultiScale(roi_color, scaleFactor = 1.8, minNeighbors = 20)  
        mouth = mouth_cascade.detectMultiScale(roi_color, scaleFactor = 2.2, minNeighbors = 30)  
  
        #draw a rectangle in mouth
        for (mx,my,mw,mh) in mouth: 
            cv2.rectangle(roi_color,(mx,my),(mx+mw,my+mh),(120,60,222),2) 
        #new_mouth(roi_color)

        mouth_images = []
        for (ex, ey, ew, eh) in mouth:
            mouth_image = roi_color[ey:ey+eh, ex:ex+ew]
            mouth_images.append(mouth_image)

        # ใช้ mouth_images ในโมเดลอื่น
        for mouth_image in mouth_images:
            re = learn_inf.predict(mouth_image)
            return re

def face_MO(frame) :
    face_locations = face_recognition.face_locations(frame)
    #face_encodings = face_recognition.face_encodings(frame, face_locations)
    #face_landmarks_list = face_recognition.face_landmarks(frame)
  
    # convert to gray scale of each frames 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
  
    # Detects faces of different sizes in the input image 
    #faces = face_cascade.detectMultiScale(gray, 1.3, 5) 
    faces2 = face_recognition.face_locations(frame)

    for (top, right, bottom, left) in faces2:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
    
        roi_gray = gray[top:top+bottom, left:left+right] 
        roi_color = frame[top:top+bottom, left:left+right] 
        

        re = eye(roi_gray,roi_color)
        #re = mouth(roi_gray,roi_color)
        return re
      

#=========================================================================================================#

eye_cascade = cv2.CascadeClassifier("eye_de_mo\haarcascade_eye_tree_eyeglasses.xml")  
#eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")  

mouth_cascade = cv2.CascadeClassifier("haarcascade_smile.xml")
  
# capture frames from a camera 
cap = cv2.VideoCapture(2) 

#Set_FRANE(cap)

# ตัวแปรสำหรับคำนวณ FPS
start_time = time.time()
frame_count = 0

#startMo(start_time)

print("-------------------------------------------")
print("------------start Ai_sleepdiver------------")
print("-------------------------------------------")

# loop 
while 1:  
    # reads frames from a camera 
    ret, frame = cap.read()  

    # cropped frames from a camera 
    #frame = cropped_frame(frame)

    re = face_MO(frame)
    print(re)
    # Display
    #cv2.imshow('Ai_sleepdiver',frames) 
    frame_count += 1

    if time.time() - start_time >= 1:
        fps = frame_count / (time.time() - start_time)
        print("Processing FPS :", round(fps, 2))
        start_time = time.time()
        frame_count = 0
  
    k = cv2.waitKey(5) # Wait Esc to stop  
    if k == 27: 
        print("-------------------------------------------")
        print("------------close Ai_sleepdiver------------")
        print("-------------------------------------------")
        break
  
# Close the window 
cap.release() 
  
# De-allocate any associated memory usage 
cv2.destroyAllWindows()


#----------------------------------------------------------