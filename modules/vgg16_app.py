import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.vgg16 import preprocess_input

# =========================
# CONFIG
# =========================
MODEL_PATH = "results_vgg/vgg_model.h5"
IMG_SIZE = 128

CLASS_NAMES = [
    "Angry", "Disgust", "Fear",
    "Happy", "Neutral", "Sad", "Surprise"
]

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

# =========================
# PREPROCESS
# =========================
def preprocess_image(img: Image.Image):
    img = img.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

# =========================
# UI
# =========================
def render():
    st.title("🧠 VGG16 (Transfer Learning)")
    st.write("""
    Model **VGG16 pretrained ImageNet**  
    - Input: 128×128 RGB  
    - Fine-tuning classifier head  
    """)

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

        idx = int(np.argmax(preds))
        confidence = preds[idx] * 100

        st.subheader("Hasil Prediksi")
        st.success(f"**{CLASS_NAMES[idx].capitalize()}**")
        st.write(f"Confidence: **{confidence:.2f}%**")

        st.subheader("Probabilitas Kelas")
        for i, cls in enumerate(CLASS_NAMES):
            st.progress(
                float(preds[i]),
                text=f"{cls.capitalize()} ({preds[i]*100:.2f}%)"
            )
