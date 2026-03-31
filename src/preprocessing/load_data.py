import os
import cv2
import numpy as np

# Path to dataset
DATA_DIR = "data"
CATEGORIES = ["real", "ai", "morph"]

IMG_SIZE = 128

data = []
labels = []

for category in CATEGORIES:
    path = os.path.join(DATA_DIR, category)
    label = CATEGORIES.index(category)  # real=0, ai=1

    for img_name in os.listdir(path):
        try:
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path)

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            data.append(img)
            labels.append(label)

        except Exception as e:
            pass

# Convert to numpy arrays
data = np.array(data)
labels = np.array(labels)

print("Data shape:", data.shape)
print("Labels shape:", labels.shape)

# Normalize data (0-255 → 0-1)
data = data / 255.0

# Split into training and testing
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)