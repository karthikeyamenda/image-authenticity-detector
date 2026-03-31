# 🧠 Image Authenticity Detector

A deep learning-based web application that detects whether an image is:

* ✅ **Real**
* 🤖 **AI Generated**
* 🔀 **Morphed / Manipulated**

---

## 🌐 Live Demo

👉 **Try the app here:**
https://image-appenticity-detector-ydslpepx3qsq9rqleggqu7.streamlit.app/

---

## 🚀 Features

* 🧠 CNN-based image classification (PyTorch)
* 🌐 Interactive web UI using Streamlit
* 📊 Confidence score for predictions
* ⚡ Fast inference using saved model
* 🖼️ Supports JPG, PNG image formats

---

## 🏗️ Project Structure

```
image-authenticity-detector/
│
├── app/
│   └── app.py              # Streamlit UI
│
├── src/
│   ├── preprocessing/
│   ├── training/
│   ├── model_loader.py     # Load trained model
│
├── models/
│   └── model.pth           # Trained model file
│
├── data/                   # (Not uploaded to GitHub)
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation (Run Locally)

### 1️⃣ Clone the repository

```
git clone https://github.com/your-username/image-authenticity-detector.git
cd image-authenticity-detector
```

---

### 2️⃣ Create virtual environment

```
python -m venv venv
venv\Scripts\activate
```

---

### 3️⃣ Install dependencies

```
pip install -r requirements.txt
```

---

### 4️⃣ Run the app

```
streamlit run app/app.py
```

---

## 📸 How to Use

1. Open the web app
2. Upload an image (`.jpg`, `.png`)
3. Wait for prediction
4. View result:

* **REAL** → Original image
* **AI** → AI-generated image
* **MORPH** → Manipulated/blended image

👉 You’ll also see a **confidence score (%)**

---

## 🧠 Model Details

* Framework: **PyTorch**
* Architecture: Custom CNN
* Input size: **128 × 128**
* Classes:

  * Real (0)
  * AI (1)
  * Morph (2)

---

## ⚠️ Limitations

* Small dataset → may overfit
* Accuracy depends on image quality
* Not yet robust for all real-world cases

---

## 🔥 Future Improvements

* Transfer learning (ResNet / EfficientNet)
* Grad-CAM visualization (fake region detection)
* Larger dataset training
* Mobile app integration
* Cloud-based model hosting

---

## 👨‍💻 Author

**Karthikeya**

---

## ⭐ Support

If you like this project:

* ⭐ Star the repository
* 🍴 Fork it
* 🚀 Share it

---

## 📜 License

This project is for educational and research purposes.
