import torch
import cv2
import numpy as np

from src.training.train_model import model

IMG_SIZE = 128

def predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0

    img = np.array(img)
    img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
    img = img.unsqueeze(0)  # add batch dimension

    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)

    if predicted.item() == 0:
        print("Prediction: REAL")
    else:
        print("Prediction: AI")


# Test image
predict_image("test.jpg")