import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

MODEL_PATH = "results/emotion_cnn_model.h5"
IMG_SIZE = 48
CLASS_NAMES = [
    "Angry", "Disgust", "Fear",
    "Happy", "Neutral", "Sad", "Surprise"
]

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(img):
    img = img.convert("L")  # grayscale
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img) / 255.0
    img = img.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    return img

def render():
    st.title("Base CNN Model")
    st.write("Model CNN dari awal (non-pretrained)")

    model = load_model()

    uploaded = st.file_uploader(
        "Upload gambar wajah",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded:
        image = Image.open(uploaded)
        st.image(image, caption="Input Image", width=250)

        img_tensor = preprocess_image(image)

        preds = model.predict(img_tensor)[0]
        idx = np.argmax(preds)

        st.subheader("Hasil Prediksi")
        st.success(f"**{CLASS_NAMES[idx]}**")
        st.write(f"Confidence: **{preds[idx]*100:.2f}%**")

        st.subheader("Probabilitas Kelas")
        for i, cls in enumerate(CLASS_NAMES):
            st.progress(float(preds[i]), text=f"{cls} ({preds[i]*100:.2f}%)")
