import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import cv2
import tempfile
model = YOLO(r"X:\best.pt")
st.set_page_config(page_title="E-Waste Detection", layout="wide")
st.title("♻️ E-Waste Component Detection")
st.markdown("Upload an image or use your webcam to detect e-waste components.")
st.sidebar.header("Settings")
confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25)
use_webcam = st.sidebar.checkbox("Use Webcam", value=False)

def display_results(image_path, results):
    st.image(Image.open(image_path), caption="Uploaded Image", use_column_width=True)
    for r in results:
        st.image(r.plot(), caption="Predictions", use_column_width=True)
if use_webcam:
    stframe = st.empty()
    cap = cv2.VideoCapture(0)
    st.sidebar.markdown("Press 'q' in the window to stop webcam.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Unable to access camera.")
            break
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        tmp_path = tmp_file.name
        cv2.imwrite(tmp_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        results = model.predict(source=tmp_path, conf=confidence)
        annotated_frame = results[0].plot()
        stframe.image(annotated_frame, channels="BGR", use_column_width=True)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
else:
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image_path = os.path.join("temp.jpg")
        with open(image_path, "wb") as f:
            f.write(uploaded_file.read())
        results = model.predict(source=image_path, conf=confidence)
        display_results(image_path, results)
st.markdown("---")
