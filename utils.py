import tensorflow as tf
import numpy as np
from PIL import Image
import streamlit as st

classes = {
    0: 'Better',
    1: 'Pocari Sweat',
    2: 'YouC1000'
}

# Model utils
@st.cache_resource
def load_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def predict_image(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    
    prediction = interpreter.get_tensor(output_details[0]['index'])
    
    # If the output is not already a probability, apply softmax
    prediction_probs = tf.nn.softmax(prediction).numpy()  # Convert to probabilities if needed
    predicted_class = np.argmax(prediction_probs)
    prediction_str = classes[predicted_class]
    return prediction_str

# Image utils
def preprocess_image(image, interpreter):
    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']  # Get the input shape from the model
    target_size = (input_shape[1], input_shape[2])  # Resize image to match model input shape

    image = image.resize(target_size)
    image = np.asarray(image).astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)

    return image
