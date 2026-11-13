import streamlit as st
from ultralytics import YOLO
from PIL import Image
import torch

# -----------------------------------
# Page Config
# -----------------------------------

st.set_page_config(
    page_title="VisionPort",
    page_icon="🚗",
    layout="wide"
)

st.title("🚦 VisionPort v0.1")
st.write("GPU-Accelerated Traffic Object Detection")

device = "cuda" if torch.cuda.is_available() else "cpu"
st.write(f"Running on: **{device.upper()}**")

# -----------------------------------
# Load Model
# -----------------------------------

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# -----------------------------------
# Upload Image
# -----------------------------------

upload_file = st.file_uploader(
    label="Upload your image",
    type=["png", "jpg", "jpeg"]
)

if upload_file is not None:

    image = Image.open(upload_file)

    st.subheader("Original Image")
    st.image(
        image,
        caption="Uploaded Traffic Image",
        use_container_width=True
    )

    # Prediction
    results = model.predict(
        source=image,
        device=device
    )

    result_image = results[0].plot()

    st.subheader("Detection Result")
    st.image(
        result_image,
        caption="Detected Objects",
        channels="BGR",
        use_container_width=True
    )