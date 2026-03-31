import sys
import os

# Fix import path for src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import torch
import cv2
import numpy as np

from src.model_loader import load_model

# Load model once
model = load_model()

IMG_SIZE = 128

st.title("🧠 Image Authenticity Detector")
st.write("Detect whether an image is REAL, AI-generated, or MORPHED")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert uploaded file to image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # Show image
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_resized = img_resized / 255.0

    img_tensor = torch.tensor(img_resized, dtype=torch.float32).permute(2, 0, 1)
    img_tensor = img_tensor.unsqueeze(0)

    # Prediction
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    classes = ["REAL", "AI", "MORPH"]

    prediction = classes[predicted.item()]
    conf = confidence.item() * 100

    # Display result
    if prediction == "REAL":
        st.success(f"Prediction: {prediction}")
    elif prediction == "AI":
        st.warning(f"Prediction: {prediction}")
    else:
        st.error(f"Prediction: {prediction}")

    st.write(f"Confidence: {conf:.2f}%")