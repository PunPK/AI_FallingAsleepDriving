import streamlit as st
import av
import mediapipe as mp
import cv2 as cv
import utils, math
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
FACE = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
map_face_mesh = mp.solutions.face_mesh
i = 'p'
def landmarksDetection(img, results, draw=False):
    img_height, img_width = img.shape[:2]
    # List of (x,y) coordinates
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]
    if draw:
        [cv.circle(img, p, 2, (0,255,0), -1) for p in mesh_coord]

    # Returning the list of tuples for each landmark 
    return mesh_coord

    # Euclidean distance 
def euclideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
    return distance

with map_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
  class VideoTransformer(VideoTransformerBase):
    def __init__(self, video_source):
        self.i = 0
        self.video_source = video_source

    def transform(self, frame):
        frame = frame.to_ndarray(format="bgr24")
        self.i += 1
        frame = cv.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)
        #frame = cropped_frame(frame)
        frame_height, frame_width = frame.shape[:2]
        rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        results = face_mesh.process(rgb_frame)
        landmarks = landmarksDetection(frame, results, False)
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
        i = 1
        cv.rectangle(frame, (FACE_x_min, FACE_y_min), (FACE_x_max, FACE_y_max), (0, 127, 255), 2)
        return frame

# Use Streamlit to create a selectbox for choosing the video source
video_source = st.selectbox(
    "Select video source",
    options=["Webcam", "Video File (.mp4)"]
)

if video_source == "Webcam":
    # Start the WebRTC stream with the video transformer using the webcam as the source
    webrtc_streamer(key="example", video_transformer_factory=lambda: VideoTransformer("Webcam"))    
else:
    # Use Streamlit to upload a video file
    video_file = st.file_uploader("Upload a video file", type=["mp4"])

    if video_file is not None:
        # Start the WebRTC stream with the video transformer using the uploaded video file as the source
        webrtc_streamer(key="example", video_transformer_factory=lambda: VideoTransformer("Video File"))
