import streamlit as st
from ultralytics import YOLO
from PIL import Image

model = YOLO("yolov8n.pt")

st.title("AI Object Detection")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)

    st.image(image, caption="Original Image")

    results = model(image)

    result_image = results[0].plot()

    st.image(result_image, caption="Detection Result")