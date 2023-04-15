# model = YOLO(r"C:\Users\snehal\PycharmProjects\PlitterDetectionUsingYolo\pretrained\yolov8n.pt")
# # # accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
# # results = model.predict(source="0")
# # results = model.predict(source="folder", show=True) # Display preds. Accepts all YOLO predict arguments
#
# # from PIL
# im1 = Image.open("dog.jpg")
# results = model.predict(source=im1, save=True)  # save plotted images
#
# # from ndarray
# im2 = cv2.imread("dog.jpg")
# results = model.predict(source=im2, save=True, save_txt=True)  # save predictions as labels
#
# # from list of PIL/ndarray
# results = model.predict(source=[im1, im2])

from PIL import Image
import streamlit as st
import cv2
import torch
import numpy as np
import tempfile
import time
from collections import Counter
import json
import pandas as pd
from model_utils import get_yolo, color_picker_fn, get_system_stat
from ultralytics import YOLO

p_time = 0

st.title("Litter Detection using Yolo V8")
sample_img = cv2.imread('c2.jpg')
FRAME_WINDOW = st.image(sample_img, channels='BGR')
cap = None


# YOLOv8 Model
model_type = "YoloV8"
model = YOLO("litterDetectionCustomModelV8.pt")

# Load Class names
class_labels = model.names

# Inference Mode
options = st.sidebar.radio(
    'Options:', ('Webcam', 'Image', 'Video'), index=1)

# Confidence
confidence = st.sidebar.slider(
    'Detection Confidence', min_value=0.0, max_value=1.0, value=0.25)

# Draw thickness
draw_thick = st.sidebar.slider(
    'Draw Thickness:', min_value=1,
    max_value=20, value=3
)

color_pick_list = []
for i in range(len(class_labels)):
    classname = class_labels[i]
    color = color_picker_fn(classname, i)
    color_pick_list.append(color)

# Image
if options == 'Image':
    upload_img_file = st.sidebar.file_uploader(
        'Upload Image', type=['jpg', 'jpeg', 'png'])
    if upload_img_file is not None:
        pred = st.checkbox(f'Predict Using {model_type}')
        file_bytes = np.asarray(
            bytearray(upload_img_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        FRAME_WINDOW.image(img, channels='BGR')

        if pred:
            img, current_no_class = get_yolo(img, model_type, model, confidence, color_pick_list, class_labels,
                                             draw_thick)
            FRAME_WINDOW.image(img, channels='BGR')

            # Current number of classes
            class_fq = dict(Counter(i for sub in current_no_class for i in set(sub)))
            class_fq = json.dumps(class_fq, indent=4)
            class_fq = json.loads(class_fq)
            df_fq = pd.DataFrame(class_fq.items(), columns=['Class', 'Number'])

            # Updating Inference results
            with st.container():
                st.markdown("<h2>Inference Statistics</h2>", unsafe_allow_html=True)
                st.markdown("<h3>Detected objects in curret Frame</h3>", unsafe_allow_html=True)
                st.dataframe(df_fq, use_container_width=True)

# Video
if options == 'Video':
    upload_video_file = st.sidebar.file_uploader(
        'Upload Video', type=['mp4', 'avi', 'mkv'])
    if upload_video_file is not None:
        pred = st.checkbox(f'Predict Using {model_type}')

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(upload_video_file.read())
        cap = cv2.VideoCapture(tfile.name)
        # if pred:


# Web-cam
if options == 'Webcam':
    cam_options = st.sidebar.selectbox('Webcam Channel',
                                       ('Select Channel', '0', '1', '2', '3'))

    if not cam_options == 'Select Channel':
        pred = st.checkbox(f'Predict Using {model_type}')
        cap = cv2.VideoCapture(int(cam_options))


if (cap != None) and pred:
    stframe1 = st.empty()
    stframe2 = st.empty()
    stframe3 = st.empty()
    while True:
        success, img = cap.read()
        if not success:
            st.error(
                f"{options} NOT working\nCheck {options} properly!!", icon="🚨"
            )
            break

        img, current_no_class = get_yolo(img, model_type, model, confidence, color_pick_list, class_labels, draw_thick)
        FRAME_WINDOW.image(img, channels='BGR')

        # FPS
        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time

        # Current number of classes
        class_fq = dict(Counter(i for sub in current_no_class for i in set(sub)))
        class_fq = json.dumps(class_fq, indent=4)
        class_fq = json.loads(class_fq)
        df_fq = pd.DataFrame(class_fq.items(), columns=['Class', 'Number'])

        # Updating Inference results
        get_system_stat(stframe1, stframe2, stframe3, fps, df_fq)
