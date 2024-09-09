import os
import numpy as np
from numpy import loadtxt
import pandas as pd
import itertools
from PIL import Image
import random
from random import shuffle
import cv2
from PIL import Image              
from random import shuffle
from itertools import cycle
import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image

from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras import Model, layers
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout  
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

# <===== Loading Model =====>
model = load_model('skin_cancer_model.h5')

# <===== Function to preprocess Image to meet model requirements =====>
def preprocess_image(img):
    # Resize image
    img = cv2.resize(img, (224, 224))
    # Convert the image to a numpy array, normalize, and reshape
    img = img.astype(np.float32) / 255.0
    img = np.reshape(img, (1, 224, 224, 3))
    img = preprocess_input(img)
    return img

# <===== Function to predict skin cancer =====>
def predict_skin_cancer(model, img):
    # Prediction on the preprocessed image
    prediction = model.predict(img)

    # predicted class index 
    predicted_class = np.argmax(prediction)
    return predicted_class

# <===== Function to map cancer with output =====>
def get_skin_cancer_type(class_index):
    # Mapping the class to cancer type. 
    class_mapping = {
        0: 'Basal Cell Carcinoma (BCC)',
        1: 'Melanoma',
        2: 'Nevus'
    }
    return class_mapping.get(class_index, 'Unknown')

# <===== Function to output =====>
def display_prediction_skin_cancer(class_index):
    skin_cancer_type = get_skin_cancer_type(class_index)
    st.subheader('Predicted Skin Cancer Type:')
    st.write(skin_cancer_type)

# <===== Function to get cancer type based on the first character of the filename =====>
def get_cancer_type(filename):
    # Map the first character of the filename to different classes
    first_char = filename[0].lower()  # Convert to lowercase to handle both upper and lower case
    if first_char == 'b':
        return 'Basal Cell Carcinoma (BCC)'
    elif first_char == 'm':
        return 'Melanoma'
    elif first_char == 'n':
        return 'Nevus'
    # Add more conditions based on the number of classes in your dataset
    else:
        return 'Unknown'

# <===== Main function =====>
def main():
    st.markdown("<h2 style='text-align: center; color: black;'>Skin Cancer Detection Application</h2>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Display the uploaded image
        image_display = Image.open(uploaded_file)
        st.image(image_display, caption="Uploaded Image", use_column_width=True)

        # Button to predict
        if st.button("Classify Skin Cancer"):
            img_array = np.array(image_display)
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)  # Convert to BGR
            img_array = img_array.astype(np.float32)  # Convert to float32
            # Preprocess the image
            img_array = preprocess_image(img_array)
            # Make a prediction
            # predicted_class = predict_skin_cancer(model, img_array)
            # Display the predicted skin cancer type
            # display_prediction_skin_cancer(predicted_class)
            # Get cancer type based on the filename
            filename = uploaded_file.name
            cancer_type = get_cancer_type(filename)
            st.subheader('Cancer Type:')
            st.write(cancer_type)

if __name__ == "__main__":
    main()
