import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import json

# Load model
model = load_model("plant_classification.h5")

# Load class indices from training
with open(r"C:\Users\janap\Desktop\class_indices.json", "r") as f:
    class_indices = json.load(f)
class_labels = {v: k for k, v in class_indices.items()}

# Prediction function
def predict(img):
    img = img.resize((128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)
    return class_labels[predicted_class], confidence

# Streamlit UI
st.title("🌿 Plant Leaf Classification")

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    predicted_class, confidence = predict(img)
    st.markdown(f"### ✅ Predicted Leaf: `{predicted_class}`")
    st.markdown(f"### 🔍 Confidence: `{confidence:.4f}`")
