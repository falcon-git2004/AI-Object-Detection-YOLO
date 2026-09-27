import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import cv2

model = YOLO("yolov8n.pt")

st.title("AI Object Detection using YOLOv8")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:

    image = Image.open(uploaded_file)

    st.subheader("Original Image")
    st.image(image)

    results = model(image)

    result_image = results[0].plot()

    st.subheader("Detection Result")
    st.image(result_image)

    detections = []

    for box in results[0].boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        detections.append({
            "Object": model.names[class_id],
            "Confidence": f"{confidence*100:.2f}%"
        })

    if detections:
        df = pd.DataFrame(detections)

        st.subheader("Detected Objects")

        st.dataframe(df)

        st.success(f"Total Objects Detected: {len(detections)}")

    result_bytes = cv2.imencode(
        ".jpg",
        result_image
    )[1].tobytes()

    st.download_button(
        label="Download Result Image",
        data=result_bytes,
        file_name="detection_result.jpg",
        mime="image/jpeg"
    )