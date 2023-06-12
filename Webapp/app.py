import gradio as gr
import mediapipe as mp
import cv2 as cv2
import cv2 as cv
import utils
# Fast Ai
from fastbook import *
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

def conv2(ni, nf): return ConvLayer(ni, nf, stride=2)

class ResBlock(Module):
  def __init__(self, nf):
    self.conv1 = ConvLayer(nf, nf)
    self.conv2 = ConvLayer(nf, nf)
  
  def forward(self, x): return x + self.conv2(self.conv1(x))

def conv_and_res(ni, nf): return nn.Sequential(conv2(ni, nf), ResBlock(nf))

learn_inf_eye = load_learner('Webapp\Model\eye_data_resnet18_fastai.pkl')
learn_inf_yawn = load_learner('Webapp\Model\yawn_ModelsfromScratch.pkl')

# Left eyes indices 
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
# Right eyes indices
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]  
mouth = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
map_face_mesh = mp.solutions.face_mesh
FONTS = cv.FONT_HERSHEY_COMPLEX
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

def euclaideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
    return distance

def landmarksDetection(img, results, draw=False):
    img_height, img_width = img.shape[:2]
    # List of (x,y) coordinates
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]
    if draw:
        [cv.circle(img, p, 2, (0,255,0), -1) for p in mesh_coord]

    # Returning the list of tuples for each landmark 
    return mesh_coord

def detecteye(img, landmarks, right_indices, left_indices):
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
    cv.rectangle(img, (eye_right_x_min, eye_right_y_min), (eye_right_x_max, eye_right_y_max), (255, 0, 0))

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
        cv.rectangle(img, (eye_right_x_min, eye_right_y_min), (eye_right_x_max, eye_right_y_max), (255, 0, 0))
        eye_right_image = img[eye_right_y_min:eye_right_y_max, eye_right_x_min:eye_right_x_max]
        re_right = learn_inf_eye.predict(eye_right_image)
        print("Eye right:", re_right)
        re_right_m = re_right[0]
    else:
        # Mouth region is not valid, return None or any appropriate value
        re_right_m = None
    
    if eye_left_x_min >= 0 and eye_left_y_min >= 0 and (eye_left_x_max - eye_left_x_min) > 0 and (eye_left_y_max - eye_left_y_min) > 0:
        # Draw rectangle around left eye
        cv.rectangle(img, (eye_left_x_min, eye_left_y_min), (eye_left_x_max, eye_left_y_max), (255, 0, 0))
    # Crop eye regions from the image based on the rectangles
        eye_left_image = img[eye_left_y_min:eye_left_y_max, eye_left_x_min:eye_left_x_max]

        re_left = learn_inf_eye.predict(eye_left_image)
        print("Eye left:", re_left)
        re_left_m = re_left[0]
        
    else:
        # Mouth region is not valid, return None or any appropriate value
        re_left_m = None
    return(re_right_m,re_left_m)

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
        cv.rectangle(img, (lips_x_min, lips_y_min), (lips_x_max, lips_y_max), (255, 0, 0))

        Yawn_image = img[lips_y_min:lips_y_max, lips_x_min:lips_x_max]
            # Perform prediction on cropped mouth region
        re_yawn = learn_inf_yawn.predict(Yawn_image)
        print("Yawn: ", re_yawn)
        return re_yawn[0]
    else:
        # Mouth region is not valid, return None or any appropriate value
        return None

    
def apply_media_pipe_detection_image(image):
    with map_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_detection:
        # Resize frame
        frame = cv.resize(image, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)
        #frame = cropped_frame(frame)
        frame_height, frame_width = frame.shape[:2]
        rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        results = face_detection.process(rgb_frame)

        if not results.multi_face_landmarks:
            return image
        
        frame = image.copy()
        if results.multi_face_landmarks:
            mesh_coords = landmarksDetection(frame, results, False)
            re_right,re_left = detecteye(frame, mesh_coords, RIGHT_EYE, LEFT_EYE)
            re_yawn = detectYawn(frame, mesh_coords,mouth)
            frame = utils.textWithBackground(frame, f'Yawn : {re_yawn}', FONTS, 0.5, (30, 200), bgOpacity=0.45, textThickness=1)
            frame = utils.textWithBackground(frame, f'Eye right : {re_right}', FONTS, 0.5, (30, 100), bgOpacity=0.45, textThickness=1)
            frame = utils.textWithBackground(frame, f'Eye left  : {re_left}', FONTS, 0.5, (30, 150), bgOpacity=0.45, textThickness=1)

        return frame
    
def apply_media_pipe_detection_video(video,blink_right_counter,blink_left_counter,blink_right_counter_n,blink_left_counter_n,
                                                        yawn_counter,blink_right,blink_left,re_yawn_counter,re_yawn_counter_n,close_eye_right,
                                                        close_eye_right_counter,close_eye_left,close_eye_left_counter,no_blink_right,no_blink_left,
                                                        blink_30_right,blink_30_left,yawn_30):
    with map_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_detection:
        # Resize frame
        frame = cv.resize(video, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)
        #frame = cropped_frame(frame)
        frame_height, frame_width = frame.shape[:2]
        rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        results = face_detection.process(rgb_frame)

        if not results.multi_face_landmarks:
            return video
 
        if results.multi_face_landmarks:
            mesh_coords = landmarksDetection(frame, results, False)
            re_right,re_left = detecteye(frame, mesh_coords, RIGHT_EYE, LEFT_EYE)
            re_yawn = detectYawn(frame, mesh_coords,mouth)

            frame = utils.textWithBackground(frame, f'Eye right : {re_right}', FONTS, 1.0, (30, 100), bgOpacity=0.9, textThickness=2)
            frame = utils.textWithBackground(frame, f'Eye left  : {re_left}', FONTS, 1.0, (30, 150), bgOpacity=0.9, textThickness=2)
            frame = utils.textWithBackground(frame, f'Yawn : {re_yawn}', FONTS, 1.0, (30, 200), bgOpacity=0.9, textThickness=2)

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

            # Count yawns
            if re_yawn == 'yawn':
                yawn_counter += 1
            
            if yawn_counter >= 10 :
                re_yawn_counter += 1
                re_yawn_counter_n += 1
                yawn_counter = 0

            if re_yawn_counter == 6 or close_eye_right_counter >= 2 or close_eye_left_counter >= 2 or no_blink_right >= 4 or no_blink_left >= 4 or blink_30_left >= 4 or blink_30_right >= 4 or yawn_30 >= 4:
                danger = '5 : Extremely Sleepy, fighting sleep'
                frame = utils.textWithBackground(frame, f"You must park your car for a break.", FONTS, 1, (200, 300), bgOpacity=0.9, textThickness=2)
            elif re_yawn_counter == 5 or close_eye_right_counter == 1 or close_eye_left_counter == 1 or no_blink_right == 3 or no_blink_left == 3 or blink_30_left == 3 or blink_30_right == 3 or yawn_30 == 3:
                danger = '4 : Sleepy, some effort to keep alert'
            elif re_yawn_counter == 4 or no_blink_right == 2 or no_blink_left == 2 or blink_30_left >= 2 or blink_30_right == 2 or yawn_30 == 2:
                danger = '3 : Sleepy, but no difficulty remaining awake'
            elif re_yawn_counter == 3 or no_blink_left == 1 or no_blink_left == 1 or blink_30_left == 1 or blink_30_right == 1 or yawn_30 == 1:
                danger = '2 : Some signs of sleepiness'
            elif re_yawn_counter == 2 :
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

        return frame

def process_video(video):
    with map_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_detection:
        for frame in video:
            # Resize frame
            frame = cv.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)
            frame_height, frame_width = frame.shape[:2]
            rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
            results = face_detection.process(rgb_frame)

            if not results.multi_face_landmarks:
                yield frame
            else:
                mesh_coords = landmarksDetection(frame, results, False)
                re_right, re_left = detecteye(frame, mesh_coords, RIGHT_EYE, LEFT_EYE)
                re_yawn = detectYawn(frame, mesh_coords, mouth)
                frame = utils.textWithBackground(frame, f'Yawn : {re_yawn}', FONTS, 0.5, (30, 200), bgOpacity=0.45, textThickness=1)
                frame = utils.textWithBackground(frame, f'Eye right : {re_right}', FONTS, 0.5, (30, 100), bgOpacity=0.45, textThickness=1)
                frame = utils.textWithBackground(frame, f'Eye left  : {re_left}', FONTS, 0.5, (30, 150), bgOpacity=0.45, textThickness=1)
                yield frame

class FaceProcessing(object):
    def __init__(self, ui_obj):
        self.name = "Face Image Processing"
        self.description = "Call for Face Image and video Processing"
        self.ui_obj = ui_obj

    def take_webcam_photo(self, image):
        return image

    def take_webcam_video(self, video_frame):
        video_out2 = process_video(video_frame)
        return video_out2

    def mp_webcam_photo(self, image):
        return image
    
    def mp_webcam_image_detection(self, image):
        detection_image = apply_media_pipe_detection_image(image)
        return detection_image
    
    def mp_webcam_video_detection(self, video):
        detection_video = apply_media_pipe_detection_video(video)
        return detection_video
    
    def webcam_stream_update(self, video_frame,blink_right_counter,blink_left_counter,blink_right_counter_n,blink_left_counter_n,
                                    yawn_counter,blink_right,blink_left,re_yawn_counter,re_yawn_counter_n,close_eye_right,
                                    close_eye_right_counter,close_eye_left,close_eye_left_counter,no_blink_right,no_blink_left,
                                    blink_30_right,blink_30_left,yawn_30) :
        video_out = apply_media_pipe_detection_video(video_frame,blink_right_counter,blink_left_counter,blink_right_counter_n,blink_left_counter_n,
                                                        yawn_counter,blink_right,blink_left,re_yawn_counter,re_yawn_counter_n,close_eye_right,
                                                        close_eye_right_counter,close_eye_left,close_eye_left_counter,no_blink_right,no_blink_left,
                                                        blink_30_right,blink_30_left,yawn_30)
        return video_out
    
    def create_ui(self):
        with self.ui_obj:
            gr.Markdown("AI_FallingAsleepDriving with Webcam/Video")
            with gr.Tabs():
                with gr.TabItem("Eye/Yawn detection Image with Webcam"):
                    with gr.Row():
                        with gr.Column():
                            mp_image_in = gr.Image(label="Webcam Image Input", source="webcam")
                        with gr.Column():
                            mp_photo_action = gr.Button("Take the Photo")
                            mp_apply_fm_action = gr.Button("Apply detection Image the Photo")
                            gr.Text("Please use it in a well lit place. Because the model may guess wrong.")
                    with gr.Row():
                        mp_photo_out = gr.Image(label="Webcam Photo Output")
                        mp_fm_photo_out = gr.Image(label="Face detection Image Photo Output")

                with gr.TabItem("Eye/Yawn detection on Live Webcam Stream"):
                    with gr.Row():
                        webcam_stream_in = gr.Image(label="Webcam Stream Input",
                                                    source="webcam",
                                                    streaming=True)
                        webcam_stream_out = gr.Image(label="Webcam Stream Output")
                        webcam_stream_in.change(
                            self.webcam_stream_update,
                            inputs=webcam_stream_in,
                            outputs=webcam_stream_out
                        )
                    with gr.Row():
                        gr.Text("Please use it in a well lit place. Because the model may guess wrong.")

            mp_photo_action.click(
                self.mp_webcam_photo,
                [
                    mp_image_in 
                ],
                [
                    mp_photo_out
                ]
            )
            
            mp_apply_fm_action.click(
                self.mp_webcam_image_detection,
                [
                    mp_image_in
                ],
                [
                    mp_fm_photo_out
                ]
            )

    
    def launch_ui(self):
        self.ui_obj.launch()

if __name__ == '__main__':
    my_app = gr.Blocks()
    face_ui = FaceProcessing(my_app)
    face_ui.create_ui()
    face_ui.launch_ui()

    print("AI_FallingAsleepDriving")
