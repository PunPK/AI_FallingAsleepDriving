import tkinter as tk
from tkinter import *
import cv2
import cv2 as cv
import os
from PIL import Image, ImageTk
import numpy as np
#import mysql.connector
from tkinter import messagebox

window = tk.Tk()
window.title("AI_FallingAsleepDriving")
window.resizable(0,0)

load1 = Image.open("LOGO.jpg")
photo1 = ImageTk.PhotoImage(load1)


header = tk.Button(window, image=photo1)
header.place(x=5, y=0)

window = tk.Tk()
window.title("Face Recognition system")


b2 = tk.Button(window, text="Detecttion", font=("Algerian",20), bg="green", fg="orange")
b2.grid(column=1, row=4)


window.geometry("800x200")
window.mainloop()

def landmarksDetection(img, results, draw=False):
    img_height, img_width = img.shape[:2]
    # List of (x,y) coordinates
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]
    if draw:
        [cv.circle(img, p, 2, (0,255,0), -1) for p in mesh_coord]

    # Returning the list of tuples for each landmark 
    return mesh_coord

# Euclaidean distance 
def euclaideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
    return distance

def detect_face():
    
    def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
        FACE_points = [landmarks[idx] for idx in FACE]

        # Find the minimum and maximum x and y coordinates of the FACE points
        x_values = [point[0] for point in FACE_points]
        y_values = [point[1] for point in FACE_points]
        FACE_x_min = min(x_values)
        FACE_x_max = max(x_values)
        FACE_y_min = min(y_values)
        FACE_y_max = max(y_values)

        # Increase width and height of the rectangle
        width_increase = 25
        height_increase = 15
        FACE_x_min -= width_increase
        FACE_x_max += width_increase
        FACE_y_min -= height_increase
        FACE_y_max += height_increase

        # Draw rectangle around lips
        cv.rectangle(img, (FACE_x_min, FACE_y_min), (FACE_x_max, FACE_y_max),(255,0,0))

    video_capture = cv2.VideoCapture(1)
    with map_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
     while True:
        ret, img = video_capture.read()
        frame = cv.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)
        rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            mesh_coords = landmarksDetection(frame, results, False)
        cv2.imshow("face Detection", img)
    
        if cv2.waitKey(1)==13:
            break
    video_capture.release()
    cv2.destroyAllWindows()
    

load3= Image.open('LOGO.jpg')
photo3 = ImageTk.PhotoImage(load3)

canvas3 = Canvas(window, width=280, height=530)
canvas3.place(x=515, y=120)
canvas3.create_image(140, 265, image=photo3)

window.geometry("800x680")
window.mainloop()