import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. CACHING: Ensures the model loads only ONCE to save RAM
@st.cache_resource
def load_cropsync_model():
    # Loading the universally compatible .h5 format!
    return tf.keras.models.load_model("rice_stage_model.h5")

model = load_cropsync_model()

# Class labels
classes = ["flowering", "germination", "noise", "tillering"]
CONFIDENCE_THRESHOLD = 0.6

st.title("🌾 Rice Crop Growth Stage Classifier")
st.write("Upload a rice crop image to detect its growth stage.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert to RGB to safely handle PNGs, then resize
    image_rgb = image.convert("RGB")
    image_resized = image_rgb.resize((224, 224))
    
    # Preprocess
    img = np.array(image_resized) / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img)[0]
    predicted_index = np.argmax(prediction)
    confidence = float(prediction[predicted_index])
    predicted_label = classes[predicted_index]

    # Display Results
    if predicted_label == "noise":
        st.warning("⚠️ Invalid image. Please upload a clear rice crop image from the field.")
    elif confidence < CONFIDENCE_THRESHOLD:
        st.warning("⚠️ Model uncertain. Please upload a clearer crop photo.")
    else:
        st.success(f"🌱 Predicted Stage: **{predicted_label.capitalize()}**")
        st.write(f"Confidence: {confidence:.2f}")

    st.subheader("Prediction Probabilities")
    for i, label in enumerate(classes):
        st.write(f"{label.capitalize()}: {prediction[i]:.2f}")
        st.progress(float(prediction[i]))
