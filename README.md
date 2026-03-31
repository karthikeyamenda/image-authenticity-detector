# Image Authenticity Detector

This project is a simple deep learning web application that classifies an image as real, AI-generated, or morphed. It is built using PyTorch for the model and Streamlit for the interface, so it can be tested easily in a browser.

## Live App

You can try the app here:
https://image-appenticity-detector-ydslpepx3qsq9rqleggqu7.streamlit.app/

## What it does

* Takes an input image
* Processes it using a trained CNN model
* Predicts whether the image is real, AI-generated, or morphed
* Displays the prediction along with a confidence score

## Project Structure

image-authenticity-detector/
├── app/
│   └── app.py
├── src/
│   ├── preprocessing/
│   ├── training/
│   ├── model_loader.py
├── models/
│   └── model.pth
├── requirements.txt
├── README.md
└── .gitignore

## How to run locally

1. Clone the repository

git clone https://github.com/your-username/image-authenticity-detector.git
cd image-authenticity-detector

2. Create a virtual environment

python -m venv venv
venv\Scripts\activate

3. Install dependencies

pip install -r requirements.txt

4. Run the app

streamlit run app/app.py

## Notes

* The model is trained on a relatively small dataset, so accuracy may vary
* Results depend on image quality and type
* This project is mainly built for learning and experimentation

## Author

Karthikeya
