import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import torch
import cv2
import numpy as np

from src.training.train_model import model

IMG_SIZE = 128

st.title("🧠 Image Authenticity Detector")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, caption="Uploaded Image", use_column_width=True)

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0

    img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
    img = img.unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)

    if predicted.item() == 0:
        st.success("Prediction: REAL")
    else:
        st.error("Prediction: AI")