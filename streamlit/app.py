import streamlit as st
import mediapipe as mp
import numpy as np

# Create a MediaPipe Hands instance
mp_hands = mp.solutions.hands.Hands()

# Use Streamlit to create a selectbox for choosing the video source
video_source = st.selectbox(
    "Select video source",
    options=["Webcam", "Video File (.mp4)"]
)
