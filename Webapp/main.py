import gradio as gr
import mediapipe as mp
import cv2 as cv2
import cv2 as cv

FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
map_face_mesh = mp.solutions.face_mesh

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

def apply_media_pipe_face_detection(image):
    with map_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_detection:
        # Resize frame
        frame = cv.resize(image, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)
        #frame = cropped_frame(frame)
        frame_height, frame_width = frame.shape[:2]
        rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        results = face_detection.process(rgb_frame)

        if not results.multi_face_landmarks:
            return image
        
        annotated_image = image.copy()
        if results.multi_face_landmarks:
            mesh_coords = landmarksDetection(annotated_image, results, False)
            detectFACE(annotated_image, mesh_coords, FACE_OVAL)
        return annotated_image
    
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

    def mp_webcam_face_detection(self, image):
        face_detection_img = apply_media_pipe_face_detection(image)
        return face_detection_img
    
    def create_ui(self):
        with self.ui_obj:
            gr.Markdown("Face Analysis with Webcam/Video")
            with gr.Tabs():
                with gr.TabItem("Playing with Webcam"):
                    with gr.Row():
                        webcam_image_in = gr.Image(label="Webcam Image Input", source="webcam")
                        webcam_video_in = gr.Video(label="Webcam Video Input", source="webcam")
                    with gr.Row():
                        webcam_photo_action = gr.Button("Take the Photo")
                        webcam_video_action = gr.Button("Take the Video")
                    with gr.Row():
                        webcam_photo_out = gr.Image(label="Webcam Photo Output")
                        webcam_video_out = gr.Video(label="Webcam Video")

                with gr.TabItem("Mediapipe Facemesh with Webcam"):
                    with gr.Row():
                        with gr.Column():
                            mp_image_in = gr.Image(label="Webcam Image Input", source="webcam")
                        with gr.Column():
                            mp_photo_action = gr.Button("Take the Photo")
                            mp_apply_fm_action = gr.Button("Apply Face Mesh the Photo")
                            mp_apply_landmarks_action = gr.Button("Apply Face Landmarks the Photo")
                    with gr.Row():
                        mp_photo_out = gr.Image(label="Webcam Photo Output")
                        mp_fm_photo_out = gr.Image(label="Face Mesh Photo Output")
                        mp_lm_photo_out = gr.Image(label="Face Landmarks Photo Output")
            
            mp_photo_action.click(
                self.mp_webcam_photo,
                [
                    mp_image_in
                ],
                [
                    mp_photo_out
                ]
            )
            webcam_photo_action.click(
                self.take_webcam_photo,
                [
                    webcam_image_in
                ],
                [
                    webcam_photo_out
                ]
            )
            webcam_video_action.click(
                self.take_webcam_video,
                [
                    webcam_video_in
                ],
                [
                    webcam_video_out
                ]
            )

            mp_apply_landmarks_action.click(
                self.mp_webcam_face_detection,
                [
                    mp_image_in
                ],
                [
                    mp_lm_photo_out
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
