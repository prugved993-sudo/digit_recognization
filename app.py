import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from streamlit_drawable_canvas import st_canvas
from PIL import Image

# -------------------------------
# Load Trained Model
# -------------------------------
model = tf.keras.models.load_model("my_model.keras")

st.set_page_config(page_title="Digit Recognition", page_icon="✍️")

st.title("✍️ Handwritten Digit Recognition")
st.write("Draw a digit (0-9) in the canvas and click **Predict**.")

# -------------------------------
# Canvas
# -------------------------------
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

# -------------------------------
# Predict Button
# -------------------------------
if st.button("Predict"):

    if canvas_result.image_data is not None:

        # Convert RGBA to grayscale
        img = canvas_result.image_data.astype(np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)

        # Resize to 28x28
        resized = cv2.resize(gray, (28, 28))

        # Normalize
        resized = resized.astype("float32") / 255.0

        # Reshape for model
        # Flatten image to 784 values
        input_img = resized.reshape(1, 784)

# Prediction
        prediction = model.predict(input_img, verbose=0)

        # Prediction
        prediction = model.predict(input_img, verbose=0)
        predicted_digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        st.image(resized, width=150, caption="Processed Image (28×28)")
        st.success(f"Predicted Digit: **{predicted_digit}**")
        st.info(f"Confidence: **{confidence:.2f}%**")

    else:
        st.warning("Please draw a digit first.")