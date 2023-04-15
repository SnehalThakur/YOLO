import streamlit as st
import numpy as np
from PIL import Image, ImageOps  # Streamlit works with PIL library very easily for Images
import cv2
from ultralytics import YOLO
import os

st.title("Litter Detection using Yolo V8")

option = st.selectbox("Select the option for Litter Detection: ", ["Image", 'Video', 'Camera'])
'''
#video file
python yolo\v8\detect\detect_and_trk.py model=yolov8s.pt source="test.mp4" show=True

#imagefile
python yolo\v8\detect\detect_and_trk.py model=yolov8m.pt source="path to image"

#Webcam
python yolo\v8\detect\detect_and_trk.py model=yolov8m.pt source=0 show=True

#External Camera
python yolo\v8\detect\detect_and_trk.py model=yolov8m.pt source=1 show=True
'''
model = YOLO(r"litterDetectionCustomModelV8.pt")
model = model.predict(r"C:\Users\snehal\PycharmProjects\PlitterDetectionUsingYolo\datasets\valid\images\000101_JPG_jpg.rf.2d6a1751dd92974abbf7c50d97eb714d.jpg", show=True, save=True, hide_labels=False, hide_conf=False, save_txt=True, conf=0.5, line_thickness=3)
model
if option == "Image":
    uploaded_image = st.file_uploader('Upload an Trash Image', type=['jpg', 'png', 'jpeg'])
    if uploaded_image \
            is not None:
        img = Image.open(uploaded_image)
        st.image(img, caption='Uploaded Image', width=300)
        if st.button('Detect'):
            path_dir = os.path.join(os.getcwd(), 'upload')
            print("path_dir =", path_dir)
            upload_path = os.path.join(path_dir, uploaded_image.name)
            print("upload_path=", upload_path)
            st.title("result")

elif option == "Video":
    uploaded_video = st.file_uploader("Upload Video", type=['mp4', 'mpeg', 'mov'])


