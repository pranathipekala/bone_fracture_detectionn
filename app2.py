import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import gdown
import os

st.set_page_config(page_title="Bone Fracture X-ray Detection", layout="centered")

IMG_SIZE = 128
MODEL_PATH = "bone_fracture_model.h5"

# 🔹 Replace this with your actual Google Drive file ID
FILE_ID = "PASTE_YOUR_FILE_ID_HERE"

@st.cache_resource
def load_cnn_model():
    # Download model if not exists
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/file/d/1LCn6cSasIcN2UcYi0e3DwxMurOQryZ8e/view?usp=sharing"
        gdown.download(url, MODEL_PATH, quiet=False)

    return tf.keras.models.load_model(MODEL_PATH)

model = load_cnn_model()

st.title("🦴 Bone Fracture X-ray Detection")
st.write("Upload an X-ray image to detect **Fractured** or **Normal** bone.")

uploaded_file = st.file_uploader(
    "Upload X-ray Image",
    type=["jpg", "jpeg", "png"]
)

def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0
    image = image.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    return image

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(image, caption="Uploaded X-ray Image", use_container_width=True)

    processed_image = preprocess_image(image)

    prediction = model.predict(processed_image)[0][0]

    st.subheader("Prediction Result")

    if prediction > 0.5:
        st.error("🔴 FRACTURED BONE DETECTED")
    else:
        st.success("🟢 NORMAL BONE")

    st.write(f"Confidence Score: **{prediction:.2f}**")