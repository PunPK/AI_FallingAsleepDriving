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
    cv.rectangle(img, (FACE_x_min, FACE_y_min), (FACE_x_max, FACE_y_max), (255, 0, 0))

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
    
def apply_media_pipe_detection_video(video):
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
            frame = utils.textWithBackground(frame, f'Yawn : {re_yawn}', FONTS, 0.5, (30, 200), bgOpacity=0.45, textThickness=1)
            frame = utils.textWithBackground(frame, f'Eye right : {re_right}', FONTS, 0.5, (30, 100), bgOpacity=0.45, textThickness=1)
            frame = utils.textWithBackground(frame, f'Eye left  : {re_left}', FONTS, 0.5, (30, 150), bgOpacity=0.45, textThickness=1)

        return frame

class FaceProcessing(object):
    def __init__(self, ui_obj):
        self.name = "Face Image Processing"
        self.description = "Call for Face Image and video Processing"
        self.ui_obj = ui_obj

    def take_webcam_photo(self, image):
        return image

    def take_webcam_video(self, images):
        return images

    def mp_webcam_photo(self, image):
        return image
    
    def mp_webcam_image_detection(self, image):
        detection_image = apply_media_pipe_detection_image(image)
        return detection_image
    
    def mp_webcam_video_detection(self, video):
        detection_video = apply_media_pipe_detection_video(video)
        return detection_video
    
    def webcam_stream_update(self, video_frame):
        video_out = apply_media_pipe_detection_video(video_frame)
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
