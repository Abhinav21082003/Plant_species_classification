import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load model
model = load_model("plant_classification.h5")

# Class labels (hardcoded instead of loading from JSON)
class_labels = {
    0: "Alstonia Scholaris (P2)",
    1: "Arjun (P1)",
    2: "Chinar (P11)",
    3: "Gauva (P3)",
    4: "Jamun (P5)",
    5: "Jatropha (P6)",
    6: "Lemon (P10)",
    7: "Mango (P0)",
    8: "Pomegranate (P9)",
    9: "Pongamia Pinnata (P7)"
}

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
