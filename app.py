import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf 

# 1. CACHING: Initialize the TFLite Interpreter only once
@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path="rice_stage_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()

# Get model input and output requirements
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Class labels
classes = ["flowering", "germination", "noise", "tillering"]
CONFIDENCE_THRESHOLD = 0.6

st.title("🌾 Cropsync Rice Growth Stage Classifier")
st.write("Upload a rice crop image from the field to instantly detect its growth stage.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert to RGB to safely handle PNGs, then resize
    image_rgb = image.convert("RGB")
    image_resized = image_rgb.resize((224, 224))
    
    # Preprocess (crucial: TFLite expects strict float32 data types)
    img = np.array(image_resized, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    # Run TFLite Prediction
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0]

    # Extract Results
    predicted_index = np.argmax(prediction)
    confidence = float(prediction[predicted_index])
    predicted_label = classes[predicted_index]

    # Display Results
    if predicted_label == "noise":
        st.warning("⚠️ Invalid image. Please upload a clear rice crop image.")
    elif confidence < CONFIDENCE_THRESHOLD:
        st.warning("⚠️ Model uncertain. Please upload a clearer crop photo.")
    else:
        st.success(f"🌱 Predicted Stage: **{predicted_label.capitalize()}**")
        st.write(f"Confidence: {confidence:.2f}")

    st.subheader("Prediction Probabilities")
    for i, label in enumerate(classes):
        st.write(f"{label.capitalize()}: {prediction[i]:.2f}")
        st.progress(float(prediction[i]))
