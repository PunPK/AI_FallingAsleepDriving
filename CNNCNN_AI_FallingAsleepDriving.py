import cv2 as cv
import cv2 as cv2
import mediapipe as mp
import time
import utils, math
import pandas as pd
# Fast Ai
from fastbook import *
import pathlib
import PIL
from PIL import ImageTk
#voice
import pyaudio
from playsound import playsound
import tkinter as tk
import tkinter.filedialog as filedialog

def conv2(ni, nf): return ConvLayer(ni, nf, stride=2)

class ResBlock(Module):
  def __init__(self, nf):
    self.conv1 = ConvLayer(nf, nf)
    self.conv2 = ConvLayer(nf, nf)
  
  def forward(self, x): return x + self.conv2(self.conv1(x))

def conv_and_res(ni, nf): return nn.Sequential(conv2(ni, nf), ResBlock(nf))


temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

learn_inf_eye = load_learner('Model\Teye_ModelsfromScratch.pkl')
learn_inf_yawn = load_learner('Model\yawn_ModelsfromScratch.pkl')

# variables 
frame_counter = 0
CEF_COUNTER = 0
TOTAL_BLINKS = 0
# constants
CLOSED_EYES_FRAME = 3
start = 0
end = 0
ch = 0

FONTS = cv.FONT_HERSHEY_COMPLEX
window = tk.Tk()
window.title("AI_FallingAsleepDriving")

# Face bounder indices 
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
# Lips indices for landmarks
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]
# Left eyes indices 
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
# Right eyes indices
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]  

map_face_mesh = mp.solutions.face_mesh

# Landmark detection function 
def Set_FRANE(cap) :
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # ตั้งค่าเฟรมเรทให้กล้องเว็บแคม
    cap.set(cv2.CAP_PROP_FPS, 60)

    # ตรวจสอบความละเอียดที่กำหนดและเฟรมเรทปัจจุบัน
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("Resolution:", width, "x", height)
    print("FPS:", fps)

def cropped_frame(frames) :
    # คำนวณตำแหน่งที่ต้องตัดวิดีโอเพื่อให้อยู่ตรงกลางของ 1080p
    width = frames.shape[1]
    height = frames.shape[0]
    
    start_x = int((width - 640) / 2)
    start_y = int((height - 480) / 2)
    end_x = start_x + 640
    end_y = start_y + 480

    # ตัดขนาดวิดีโอ
    cropped_frame = frames[start_y:end_y, start_x:end_x]

    return cropped_frame

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

def select_input():
    # สร้างหน้าต่างย่อยเพื่อให้ผู้ใช้เลือก Input
    select_window = tk.Toplevel(window)
    select_window.title("เลือก Input")
    
    # เพิ่มปุ่มเลือกวิดีโอ
    video_button = tk.Button(select_window, text="เลือกวิดีโอ", command=select_video)
    video_button.pack(pady=10)

    # เพิ่มปุ่มเลือกกล้องเว็บแคม
    webcam_button = tk.Button(select_window, text="เลือกกล้องเว็บแคม", command=start_webcam)
    webcam_button.pack(pady=10)

    if ch == 1 :
        file_path = filedialog.askopenfilename(title="เลือกไฟล์วิดีโอ", filetypes=[("Video Files", "*.mp4")])
        if file_path:
            # ทำสิ่งที่ต้องการเมื่อได้รับไฟล์วิดีโอ
            print("Got file video:", file_path)
            # เรียกใช้ฟังก์ชันสำหรับประมวลผลวิดีโอ
            video = cv2.VideoCapture(file_path)
            select_window.destroy()
            return video
    if ch == 2 :
        video = cv2.VideoCapture(0)
        print("Start Webcam...")
        # ส่งค่า video กลับไป
        select_window.destroy()
        return video

def select_video():
    ch = 1

def start_webcam():
    ch = 2

    
def open_program(event=None):
    # ส่วนโค้ดที่ต้องการให้โปรแกรมเริ่มต้นเมื่อกดปุ่มเข้าสู่โปรแกรม
    print("-------------------------------------------")
    print("------------start Ai_sleepdiver------------")
    print("-------------------------------------------")
    video = select_input()
    strat = 1
    return video

def exit_program(event=None):
    # ส่วนโค้ดที่ต้องการเมื่อกดปุ่ม Exit
    print("-------------------------------------------")
    print("------------close Ai_sleepdiver------------")
    print("-------------------------------------------")
    # ทำลายหน้าต่าง GUI
    window.destroy()
    

def start_mode() :
    # เพิ่มรูปภาพหน้าแรก
    image = Image.open("LOGO.jpg")  # ใส่ชื่อไฟล์รูปภาพที่คุณต้องการใช้
    photo = ImageTk.PhotoImage(image)
    label = tk.Label(window, image=photo)
    label.pack()
    # กำหนดขนาดหน้าต่าง GUI
    window.geometry("800x600")
    # เพิ่มปุ่มเข้าสู่โปรแกรม
    button = tk.Button(window, text="เข้าสู่โปรแกรม", command=open_program)
    button.pack()
    exit_button = tk.Button(window, text="Exit", command=exit_program)
    exit_button.pack()
    if start == 1:
        window.destroy()
        return video
    window.mainloop()
    

# FACE detection function 
def detectFACE(img, landmarks, FACE):
    # FACE coordinates
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
    cv.rectangle(img, (FACE_x_min, FACE_y_min), (FACE_x_max, FACE_y_max), utils.GREEN, 2)

def detecteye(img, landmarks, right_indices, left_indices):

    # Right eye
    # Horizontal line 
    rh_right = landmarks[right_indices[0]]
    rh_left = landmarks[right_indices[8]]
    # Vertical line 
    rv_top = landmarks[right_indices[12]]
    rv_bottom = landmarks[right_indices[4]]

    # Left eye 
    # Horizontal line 
    lh_right = landmarks[left_indices[0]]
    lh_left = landmarks[left_indices[8]]
    # Vertical line 
    lv_top = landmarks[left_indices[12]]
    lv_bottom = landmarks[left_indices[4]]

    rhDistance = euclaideanDistance(rh_right, rh_left)
    rvDistance = euclaideanDistance(rv_top, rv_bottom)

    lvDistance = euclaideanDistance(lv_top, lv_bottom)
    lhDistance = euclaideanDistance(lh_right, lh_left)

    reRatio = rhDistance/rvDistance
    leRatio = lhDistance/lvDistance


    # Draw lines on eyes 
    # Right eye
    eye_right_x_min = min([landmarks[idx][0] for idx in RIGHT_EYE])
    eye_right_x_max = max([landmarks[idx][0] for idx in RIGHT_EYE])
    eye_right_y_min = min([landmarks[idx][1] for idx in RIGHT_EYE])
    eye_right_y_max = max([landmarks[idx][1] for idx in RIGHT_EYE])

    # Increase width of rectangle
    width_increase = 20
    eye_right_x_min -= width_increase
    eye_right_x_max += width_increase
    eye_right_y_min -= width_increase
    eye_right_y_max += width_increase
    # Draw rectangle around right eye
    cv.rectangle(img, (eye_right_x_min, eye_right_y_min), (eye_right_x_max, eye_right_y_max), utils.GREEN, 2)

    # Left eye
    eye_left_x_min = min([landmarks[idx][0] for idx in LEFT_EYE])
    eye_left_x_max = max([landmarks[idx][0] for idx in LEFT_EYE])
    eye_left_y_min = min([landmarks[idx][1] for idx in LEFT_EYE])
    eye_left_y_max = max([landmarks[idx][1] for idx in LEFT_EYE])

    # Increase width of rectangle
    eye_left_x_min -= width_increase
    eye_left_x_max += width_increase
    eye_left_y_min -= width_increase
    eye_left_y_max += width_increase

    if eye_right_x_min >= 0 and eye_right_y_min >= 0 and (eye_right_x_max - eye_right_x_min) > 0 and (eye_right_y_max - eye_right_y_min) > 0:
        # Draw rectangle around left eye
        cv.rectangle(img, (eye_right_x_min, eye_right_y_min), (eye_right_x_max, eye_right_y_max), utils.GREEN, 2)

        # Crop eye regions from the image based on the rectangles
        eye_right_image = img[eye_right_y_min:eye_right_y_max, eye_right_x_min:eye_right_x_max]

        try:
            # Perform prediction on cropped eye regions
            re_right = learn_inf_eye.predict(eye_right_image)
            print("Eye right:", re_right)
            re_right_m = reRatio
        except (ValueError, PIL.Image.DecompressionBombError):
            # Handle the specific error and return None
            re_right_m = None
    else:
        # Mouth region is not valid, return None or any appropriate value
        re_right_m = None
    
    if eye_left_x_min >= 0 and eye_left_y_min >= 0 and (eye_left_x_max - eye_left_x_min) > 0 and (eye_left_y_max - eye_left_y_min) > 0:
        # Draw rectangle around left eye
        cv.rectangle(img, (eye_left_x_min, eye_left_y_min), (eye_left_x_max, eye_left_y_max), utils.GREEN, 2)

        # Crop eye regions from the image based on the rectangles
        eye_left_image = img[eye_left_y_min:eye_left_y_max, eye_left_x_min:eye_left_x_max]

        try:
            # Perform prediction on cropped eye regions
            re_left = learn_inf_eye.predict(eye_left_image)
            print("Eye left:", re_left)
            re_left_m = leRatio
        except (ValueError, PIL.Image.DecompressionBombError):
            # Handle the specific error and return None
            re_left_m = None
    else:
        # Mouth region is not valid, return None or any appropriate value
        re_left_m = None
    

    return(re_right_m,re_left_m)

# Yawn detection function 
def detectYawn(img, landmarks, LIPS):
    # Lips coordinates
    lips_points = [landmarks[idx] for idx in LIPS]

    # Find the minimum and maximum x and y coordinates of the lips points
    x_values = [point[0] for point in lips_points]
    y_values = [point[1] for point in lips_points]
    lips_x_min = min(x_values)
    lips_x_max = max(x_values)
    lips_y_min = min(y_values)
    lips_y_max = max(y_values)

    # Increase width and height of the rectangle
    width_increase = 25
    height_increase = 20
    lips_x_min -= width_increase
    lips_x_max += width_increase
    lips_y_min -= height_increase
    lips_y_max += height_increase

    if lips_x_min >= 0 and lips_y_min >= 0 and (lips_x_max - lips_x_min) > 0 and (lips_y_max - lips_y_min) > 0:
        # Draw rectangle around lips
        cv.rectangle(img, (lips_x_min, lips_y_min), (lips_x_max, lips_y_max), utils.GREEN, 2)

        Yawn_image = img[lips_y_min:lips_y_max, lips_x_min:lips_x_max]
        try:
            # Perform prediction on cropped mouth region
            re_yawn = learn_inf_yawn.predict(Yawn_image)
            print("Yawn: ", re_yawn)
            return re_yawn[0]
        except (ValueError, PIL.Image.DecompressionBombError):
            # Handle the specific error and return None
            return None
    else:
        # Mouth region is not valid, return None or any appropriate value
        return None

#=================================================Start========================================================================#
# Variables for counting
blink_right_counter = 0 
blink_left_counter = 0  
blink_right_counter_n = 0 
blink_left_counter_n = 0 
yawn_counter = 0        
blink_right = 0         
blink_left = 0          
re_yawn_counter = 0     
re_yawn_counter_n = 0  
close_eye_right = 0
close_eye_right_counter = 0
close_eye_left = 0
close_eye_left_counter = 0
no_blink_right = 0
no_blink_left = 0
blink_30_right = 0
blink_30_left = 0
yawn_30 = 0
danger = '0 : Alert'   
Sound = "OFF"
n_i = 0
show_txt = ""
# variables 
frame_counter = 0
leCEF_COUNTER = 0
leTOTAL_BLINKS = 0
# constants
leCLOSED_EYES_FRAME = 3
reCEF_COUNTER = 0
reTOTAL_BLINKS = 0
# constants
reCLOSED_EYES_FRAME = 3

# Data list for storing results
data = []

Blinks_right_start = 20
Blinks_left_start = 20
Yawn_start = 15
duration = 3  # ระยะเวลาของเสียง (วินาที)
sampling_rate = 44100  # อัตราการสุ่มตัวอย่างเสียง (หน่วยเป็น Hz)
frequency = 440  # ความถี่ของเสียงที่ต้องการสร้าง (หน่วยเป็น Hz)

# สร้างตัวอย่างเสียง
samples = (np.sin(2 * np.pi * np.arange(sampling_rate * duration) * frequency / sampling_rate)).astype(np.float32)

p = pyaudio.PyAudio()

stream = p.open(format=pyaudio.paFloat32,
                channels=1,
                rate=sampling_rate,
                output=True)

data.append({'Frame': frame_counter, 'Blinks_right': Blinks_right_start, 'Blinks_left': Blinks_left_start, 'Yawns': Yawn_start})

with map_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    # Starting time
    # ตัวแปรสำหรับคำนวณ FPS
   video = start_mode()
   if start == 1 :
    start_time = time.time()
    start_time_right = time.time()
    start_time_left = time.time()
    start_blink_time = time.time()
    start_yawn_time = time.time()
    start_n = time.time()
    frame_count = 0          
    #Set_FRANE(video)
    

    # Starting video loop
    while True:
        frame_counter += 1 # Frame counter
        ret, frame = video.read() # Get frame from camera
        if not ret:
            break # No more frames, break
        n_current_time = time.time()
        n_elapsed_time = n_current_time - start_n
        minutes, seconds = divmod(n_elapsed_time, 60)

        # Resize frame
        frame = cv.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)
        #frame = cropped_frame(frame)
        frame_height, frame_width = frame.shape[:2]
        rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            mesh_coords = landmarksDetection(frame, results, False)
            # Eye and yawn detection
            #gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            #detectEyes(frame, gray)
            reEYE,leEYE = detecteye(frame, mesh_coords, RIGHT_EYE, LEFT_EYE)
            re_yawn = detectYawn(frame, mesh_coords,LIPS)
            #detectFACE(frame, mesh_coords, FACE_OVAL)
            if reEYE > 4.7 :
                re_right = "close eye"
                reCEF_COUNTER += 1
            else:
                re_right = "open eye"
                if reCEF_COUNTER > reCLOSED_EYES_FRAME:
                    reTOTAL_BLINKS += 1
                    reCEF_COUNTER = 0

            if leEYE > 4.7 :
                leCEF_COUNTER += 1
                re_left = "close eye"
            else:
                re_left = "open eye"
                if leCEF_COUNTER > leCLOSED_EYES_FRAME:
                    leTOTAL_BLINKS += 1
                    leCEF_COUNTER = 0

            frame = utils.textWithBackground(frame, f'Eye right : {re_right}', FONTS, 1.0, (30, 100), bgOpacity=0.9, textThickness=2)
            frame = utils.textWithBackground(frame, f'Eye left  : {re_left}', FONTS, 1.0, (30, 150), bgOpacity=0.9, textThickness=2)
            frame = utils.textWithBackground(frame, f'Yawn : {re_yawn}', FONTS, 1.0, (30, 200), bgOpacity=0.9, textThickness=2)
            # Count blink
            if re_right == 'open eye' :
                elapsed_right_time = time.time() - start_time_right
                if re_right == 'close eye':
                    start_time_right_0 = time.time() 
                elif elapsed_right_time >= 60 :
                    elapsed_right_time = time.time() - start_time_right - elapsed_right_time
                    no_blink_right = 0
            if re_left == 'open eye' :
                elapsed_left_time = time.time() - start_time_left
                if re_left == 'close eye':
                    start_time_left = time.time()
                elif elapsed_left_time >= 60 :
                    elapsed_left_time = time.time() - start_time_left - elapsed_left_time
                    no_blink_left = 0

            if re_right == 'close eye' :
                close_eye_right += 1
                blink_right += 1
            if re_left == 'close eye':
                close_eye_left += 1
                blink_left += 1

            if close_eye_right >= 1 :
                if re_right == 'open eye' :
                    close_eye_right = 0 
                elif close_eye_right >= 5 :
                    close_eye_right_counter += 1
                    close_eye_right = 0 

            if close_eye_left >= 1 :
                if re_left == 'open eye' :
                    close_eye_left = 0 
                elif close_eye_left >= 5 :
                    close_eye_left_counter += 1
                    close_eye_left = 0 

            if blink_right >= 2 :
                blink_right_counter += 1
                blink_right_counter_n += 1
                blink_right = 0
            if blink_left >= 2 :
                blink_left_counter += 1
                blink_left = 0
            
            blink_time = time.time() - start_blink_time
            if blink_time <= 59 :
                if blink_right_counter_n >= 30 :
                    blink_30_right += 1
                    blink_right_counter_n = 0
                if blink_left_counter_n >= 30 :
                    blink_30_left += 1
                    blink_left_counter_n = 0
            elif blink_time >= 60 :
                blink_time = time.time() - start_blink_time - blink_time
                blink_right_counter_n = 0
                blink_left_counter_n = 0


            # Count yawns
            if re_yawn == 'yawn':
                yawn_counter += 1
            
            if yawn_counter >= 10 :
                re_yawn_counter += 1
                re_yawn_counter_n += 1
                yawn_counter = 0

            yawn_time = time.time() - start_yawn_time
            if yawn_time <= 59 :
                if re_yawn_counter_n >= 3 :
                    yawn_30 += 1
                    re_yawn_counter_n = 0
            elif yawn_time >= 60 :
                yawn_time = time.time() - start_yawn_time - yawn_time
                re_yawn_counter_n = 0

            if re_yawn_counter >= 12 or close_eye_right_counter >= 2 or close_eye_left_counter >= 2 or no_blink_right >= 3 or no_blink_left >= 3 or blink_30_left >= 3 or blink_30_right >= 3 or yawn_30 >= 3 :
                danger = '4 : Extremely Sleepy, fighting sleep'
                stream.write("test-Video.mp4")
                frame = utils.textWithBackground(frame, f"You must park your car for a break.", FONTS, 1, (200, 300), bgOpacity=0.9, textThickness=2)
            elif 11 >= re_yawn_counter >= 6 or no_blink_right == 2 or no_blink_left == 2 or blink_30_left >= 2 or blink_30_right == 2 or yawn_30 == 2:
                danger = '3 : Sleepy, but no difficulty remaining awake'
            elif 5 >= re_yawn_counter >= 3 or no_blink_left == 1 or no_blink_left == 1 or blink_30_left == 1 or blink_30_right == 1 or yawn_30 == 1:
                danger = '2 : Some signs of sleepiness'
            elif 2 >= re_yawn_counter >= 1:
                danger = '1 : Rather Alert'
                
            frame = utils.textWithBackground(frame, f'Degree of danger :: {danger}', FONTS, 0.5, (500, 50), bgOpacity=0.45, textThickness=1)
            frame = utils.textWithBackground(frame, f'Sum blink Eye right    : {blink_right_counter}', FONTS, 0.5, (650, 95), bgOpacity=0.45, textThickness=1)
            frame = utils.textWithBackground(frame, f'Sum blink Eye left     : {blink_left_counter}', FONTS, 0.5, (650, 140), bgOpacity=0.45, textThickness=1)
            frame = utils.textWithBackground(frame, f'Sum close Eye right    : {close_eye_right_counter}', FONTS, 0.5, (650, 185), bgOpacity=0.45, textThickness=1)
            frame = utils.textWithBackground(frame, f'Sum close Eye left     : {close_eye_left_counter}', FONTS, 0.5, (650, 230), bgOpacity=0.45, textThickness=1)
            #frame = utils.textWithBackground(frame, f'Sum no blink Eye right : {no_blink_right}', FONTS, 0.5, (650, 275), bgOpacity=0.45, textThickness=1)
            #frame = utils.textWithBackground(frame, f'Sum no blink Eye left  : {no_blink_left}', FONTS, 0.5, (650, 320), bgOpacity=0.45, textThickness=1)
            frame = utils.textWithBackground(frame, f'Sum no blink Eye right : 0', FONTS, 0.5, (650, 275), bgOpacity=0.45, textThickness=1)
            frame = utils.textWithBackground(frame, f'Sum no blink Eye left  : 0', FONTS, 0.5, (650, 320), bgOpacity=0.45, textThickness=1)
            frame = utils.textWithBackground(frame, f'Sum 30 blink Eye right : {blink_30_right}', FONTS, 0.5, (650, 365), bgOpacity=0.45, textThickness=1)
            frame = utils.textWithBackground(frame, f'Sum 30 blink Eye left  : {blink_30_left}', FONTS, 0.5, (650, 410), bgOpacity=0.45, textThickness=1)
            frame = utils.textWithBackground(frame, f'Sum Yawn               : {re_yawn_counter}', FONTS, 0.5, (650, 455), bgOpacity=0.45, textThickness=1)
            frame = utils.textWithBackground(frame, f'Sum 30 Yawn            : {yawn_30}', FONTS, 0.5, (650, 500), bgOpacity=0.45, textThickness=1)

        # Calculate frame per second (FPS)
        end_time = time.time() - start_time
        fps = (frame_counter / end_time)

        frame = utils.textWithBackground(frame, f'FPS : {round(fps,1)}', FONTS, 1.0, (30, 50), bgOpacity=0.9, textThickness=2)
        #frame = utils.textWithBackground(frame, f'Sound : {Sound}', FONTS, 1.0, (30, 250), bgOpacity=0.9, textThickness=2)
        frame = utils.textWithBackground(frame, "Elapsed Time: {:02d}:{:02d}".format(int(minutes), int(seconds)), FONTS, 0.5, (10, 495), bgOpacity=0.45, textThickness=1)
        frame = utils.textWithBackground(frame, f"Press the 'q' or 'Q' button to close the program", FONTS, 0.5, (10, 525), bgOpacity=0.45, textThickness=1)

        cv.imshow('AI_FallingAsleepDriving',frame)

        key = cv.waitKey(2)
        if key == ord('q') or key == ord('Q') :
            # Store data
            data.append({'Frame': frame_counter, 'Blinks_right': blink_right_counter, 'Blinks_left': blink_left_counter, 'Yawns': re_yawn_counter})
            data.append({
                'Frame': frame_counter,
                'Blinks_right': '{:.2f}%'.format(blink_right_counter / Blinks_right_start * 100),
                'Blinks_left': '{:.2f}%'.format(blink_left_counter / Blinks_left_start * 100),
                'Yawns': '{:.2f}%'.format(re_yawn_counter / Yawn_start * 100)
                })
            # Create a DataFrame from the data list
            df = pd.DataFrame(data)
            df = df.rename(index={0: "Start"})
            df = df.rename(index={1: "Work"})
            df = df.rename(index={2: "Summarize"})

            # Print the DataFrame
            print(df)
            print("-------------------------------------------")
            print("-------close AI_FallingAsleepDriving-------")
            print("-------------------------------------------")
            break

cv.destroyAllWindows()
if start == 1 :
    video.release()