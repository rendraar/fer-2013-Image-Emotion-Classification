import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model

from tensorflow.keras.applications.efficientnet import (
    EfficientNetB0,
    preprocess_input
)

# =========================
# CONFIG
# =========================
IMG_SIZE = 128
MODEL_PATH = "results_effnet/efficientnet_model.h5"

CLASS_NAMES = [
    "Angry", "Disgust", "Fear",
    "Happy", "Neutral", "Sad", "Surprise"
]

# =========================
# LOAD MODEL
# =========================
def build_effnet():
    base = EfficientNetB0(
        weights=None,
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    out = Dense(len(CLASS_NAMES), activation="softmax")(x)

    model = Model(inputs=base.input, outputs=out)
    return model


@st.cache_resource
def load_model():
    model = build_effnet()
    model.load_weights(MODEL_PATH)
    return model

# =========================
# IMAGE PREPROCESS
# =========================
def preprocess_image(img: Image.Image):
    img = img.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

# =========================
# PREDICTION
# =========================
def predict_image(model, img):
    preds = model.predict(img)
    class_id = np.argmax(preds, axis=1)[0]
    confidence = preds[0][class_id]
    return CLASS_NAMES[class_id], confidence, preds[0]

# =========================
# STREAMLIT UI
# =========================
def render():
    st.title("EfficientNetB0 (Transfer Learning)")
    st.write("Prediksi emosi menggunakan model EfficientNetB0")

    uploaded_file = st.file_uploader(
        "Upload image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        with st.spinner("Predicting..."):
            model = load_model()
            img_tensor = preprocess_image(image)
            label, conf, probs = predict_image(model, img_tensor)

        st.success(f"Prediction: **{label}**")
        st.write(f"Confidence: **{conf:.5f}**")

        st.subheader("Class Probabilities")
        for i, cls in enumerate(CLASS_NAMES):
            st.progress(
                float(probs[i]),
                text=f"{cls} ({probs[i]*100:.2f}%)"
            )
