import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

# Create a MediaPipe Hands instance
mp_hands = mp.solutions.hands.Hands()

# Use Streamlit to create a selectbox for choosing the video source
video_source = st.selectbox(
    "Select video source",
    options=["Webcam", "Video File (.mp4)"]
)

if video_source == "Webcam":
    # Open the webcam
    cap = cv2.VideoCapture(0)
else:
    # Use Streamlit to upload a video file
    video_file = st.file_uploader("Upload a video file", type=["mp4"])

    if video_file is not None:
        # Open the video file
        file_bytes = video_file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        cap = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

# Read and process frames from the video
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5
) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the BGR frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Hands
        results = hands.process(frame_rgb)

        # Display the processed frame
        annotated_image = frame_rgb.copy()
        st.image(annotated_image, channels="RGB")

# Release the video capture
cap.release()
