import streamlit as st
import cv2
import torch
import tempfile

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')
model.conf = 0.25

st.title("Retail Object Detection")

uploaded_file = st.file_uploader("Upload Image or Video", type=["jpg", "png", "mp4"])

if uploaded_file is not None:

    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())

    if uploaded_file.type.startswith("image"):

        image = cv2.imread(temp_file.name)

        results = model(image)

            # Count detections
        count = len(results.xyxy[0])

# Draw boxes
        output = results.render()[0]

        st.image(output, caption=f"Detected Image (Total Objects: {count})", channels="BGR")

    elif uploaded_file.type.startswith("video"):

        cap = cv2.VideoCapture(temp_file.name)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

           
            results = model(frame)

# Count detections
            count = len(results.xyxy[0])

# Draw boxes
            output = results.render()[0]

            stframe.image(output, caption=f"Objects: {count}", channels="BGR")

        cap.release()