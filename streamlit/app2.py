import streamlit as st
import av
import numpy as np
import cv2 as cv
import cv2 as cv2
import mediapipe as mp
import time
import utils, math
import numpy as np
import keyboard
import pandas as pd
# Fast Ai
from fastbook import *
from glob import glob
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score
import pathlib
import PIL

def conv2(ni, nf): return ConvLayer(ni, nf, stride=2)

class ResBlock(Module):
  def __init__(self, nf):
    self.conv1 = ConvLayer(nf, nf)
    self.conv2 = ConvLayer(nf, nf)
  
  def forward(self, x): return x + self.conv2(self.conv1(x))

def conv_and_res(ni, nf): return nn.Sequential(conv2(ni, nf), ResBlock(nf))


temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath



from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

class VideoTransformer(VideoTransformerBase):
    def __init__(self, video_source):
        self.i = 0
        self.video_source = video_source

        # Load your model here
        self.model = load_learner('yawn_ModelsfromScratch.pkl')

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.i += 1

        # Preprocess the image (resize, normalize, etc.)
        img = self.model.predict(img)

        # Make predictions using your model
        predictions = self.model.predict(np.expand_dims(img, axis=0))

        # Process the predictions and display the results
        # ...

        return img

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
