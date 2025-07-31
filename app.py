import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image
from PIL import Image
import requests
import re
import os

st.set_page_config(page_title="Dog Breed Classification", page_icon="🐶")
st.title("🐕 End-to-End Dog Breed Classification")

# ==============================
# 1. Download Model from Google Drive
# ==============================
@st.cache_resource
def load_model():
    model_path = "dog_vision_model.h5"
    if not os.path.exists(model_path):
        drive_link = "https://drive.google.com/file/d/1YourModelID/view?usp=sharing"  # CHANGE THIS
        match = re.search(r"/d/([a-zA-Z0-9_-]+)", drive_link)
        if match:
            file_id = match.group(1)
            download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
            st.write("📥 Downloading model from Google Drive...")
            r = requests.get(download_url)
            with open(model_path, "wb") as f:
                f.write(r.content)
        else:
            st.error("❌ Invalid Google Drive link for model.")
            st.stop()
    return tf.keras.models.load_model(model_path)

model = load_model()

# ==============================
# 2. Load Class Labels
# ==============================
@st.cache_data
def load_labels():
    df = pd.read_csv("labels.csv")
    if "breed" in df.columns:
        return sorted(df["breed"].unique().tolist())
    else:
        st.error("❌ 'labels.csv' must contain a 'breed' column.")
        st.stop()

class_names = load_labels()

# ==============================
# 3. Image Upload
# ==============================
uploaded_file = st.file_uploader("Upload a dog image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_resized = img.resize((224, 224))  # Resize to match training
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("Predict"):
        preds = model.predict(img_array)
        top_indices = preds[0].argsort()[-3:][::-1]  # Top 3 predictions
        st.subheader("🔮 Top Predictions:")
        for idx in top_indices:
            breed = class_names[idx]
            confidence = preds[0][idx] * 100
            st.write(f"✅ {breed}: **{confidence:.2f}%**")
