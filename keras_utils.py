import tensorflow as tf
import numpy as np
from PIL import Image
import streamlit as st

# Define the class names (labels)
classes = {
    0: 'Better',
    1: 'LeMinerale',
    2: 'Oreo',
    3: 'Pocari Sweat',
    4: 'YouC1000'
}

# Model utils
@st.cache_resource
def load_model(model_path):
    # Load the Keras model from the .h5 file
    model = tf.keras.models.load_model(model_path)
    return model

def predict_image(model, image):
    # Get predictions from the model
    prediction = model.predict(image)

    # Get the predicted class index and label
    predicted_class = np.argmax(prediction)
    prediction_str = classes[predicted_class]

    confidence = prediction[0, predicted_class] * 100

    return prediction_str, confidence

# Image utils
def preprocess_image(image, model):
    # Get the input shape of the model
    input_shape = model.input_shape  # e.g., (None, 224, 224, 3)

    target_size = (input_shape[1], input_shape[2])  # Resize image to match model input shape

    image = image.resize(target_size)
    image = np.asarray(image).astype('float32') / 255.0  # Normalize image
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    return image

#