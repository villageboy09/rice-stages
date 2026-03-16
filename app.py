import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import keras 

# 1. CACHING: This decorator ensures the model loads only ONCE when the kiosk boots up.
# It saves massive amounts of RAM and makes predictions instant for the farmers.
@st.cache_resource
def load_cropsync_model():
    # 2. THE PATCH: This custom class intercepts the Dense layer load 
    # and throws away the 'quantization_config' argument that crashes older cloud servers.
    class SafeDense(keras.layers.Dense):
        def __init__(self, *args, **kwargs):
            kwargs.pop("quantization_config", None)
            super().__init__(*args, **kwargs)
            
    # Load the model using our patch
    with keras.saving.custom_object_scope({'Dense': SafeDense}):
        return tf.keras.models.load_model("rice_stage_model.keras")

# Load model using the cached function
model = load_cropsync_model()

# Class labels (must match training order)
classes = ["flowering", "germination", "noise", "tillering"]
CONFIDENCE_THRESHOLD = 0.6

# --- UI Setup ---
st.title("🌾 Rice Crop Growth Stage Classifier")
st.write("Upload a rice crop image to detect its growth stage.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # 3. SAFETY NET: Convert to RGB to drop alpha channels from PNGs before processing
    image_rgb = image.convert("RGB")
    image_resized = image_rgb.resize((224, 224))
    
    # Preprocess image array
    img = np.array(image_resized) / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img)[0]
    predicted_index = np.argmax(prediction)
    confidence = float(prediction[predicted_index])
    predicted_label = classes[predicted_index]

    # --- Results Display ---
    if predicted_label == "noise":
        st.warning("⚠️ Invalid image. Please upload a clear rice crop image from the field.")
    elif confidence < CONFIDENCE_THRESHOLD:
        st.warning("⚠️ Model uncertain. Please upload a clearer crop photo.")
    else:
        st.success(f"🌱 Predicted Stage: **{predicted_label.capitalize()}**")
        st.write(f"Confidence: {confidence:.2f}")

    # Show probability distribution
    st.subheader("Prediction Probabilities")
    for i, label in enumerate(classes):
        # Streamlit's st.progress requires a float between 0.0 and 1.0
        st.write(f"{label.capitalize()}: {prediction[i]:.2f}")
        st.progress(float(prediction[i]))
