<<<<<<< HEAD
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import keras_utils

# Title
st.title('Gloo Packaged Product Detection')
st.write("Upload an image or take a picture using your device's camera and get a prediction!")

# Header
st.header("Upload an Image or Use Your Camera")

# Loading model
model_path = './model/modelv3.h5'
# interpreter = utils.load_model(model_path)

model = keras_utils.load_model(model_path)

# Taking data
camera_image = st.camera_input("Take a picture")
uploaded_file = st.file_uploader("Or upload an image", type=["jpg", "jpeg", "png"])   

image = None

if camera_image is not None or uploaded_file is not None:
    if camera_image is not None:
        image = Image.open(camera_image)
    else:
        image = Image.open(uploaded_file)

if image:
    st.image(image, caption="Selected Image", use_container_width=True)
    st.write("Classifying...")

    # Preprocess image
    inpt_image = keras_utils.preprocess_image(image, model)

    # Model inference
    prediction, confidence = keras_utils.predict_image(model=model, image=inpt_image)

    # Show inference result
    st.write(f"Prediction: {prediction}")
    st.write(f"Confidence : {confidence:.2f}%")
=======
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import utils

# Title
st.title('Gloo Packaged Product Detection')
st.write("Upload an image or take a picture using your device's camera and get a prediction!")

# Header
st.header("Upload an Image or Use Your Camera")

# Loading model
model_path = './model/modelv2.tflite'
interpreter = utils.load_model(model_path)

# Taking data
camera_image = st.camera_input("Take a picture")
uploaded_file = st.file_uploader("Or upload an image", type=["jpg", "jpeg", "png"])   

image = None

if camera_image is not None or uploaded_file is not None:
    if camera_image is not None:
        image = Image.open(camera_image)
    else:
        image = Image.open(uploaded_file)

if image:
    st.image(image, caption="Selected Image", use_container_width=True)
    st.write("Classifying...")

    # Preprocess image
    inpt_image = utils.preprocess_image(image, interpreter)

    # Model inference
    prediction = utils.predict_image(interpreter, inpt_image)

    # Show inference result
    st.write(f"Prediction: {prediction}")
>>>>>>> bb1e94c0849ded5c88784b521cf2aa39ae46a557
