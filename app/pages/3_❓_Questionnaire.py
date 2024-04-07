import os
import json
from PIL import Image

import numpy as np
import tensorflow as tf
import streamlit as st

import requests


model_path = "C:/Users/ragav/Downloads/plant-disease-prediction-cnn-deep-leanring-project/plant-disease-prediction-cnn-deep-leanring-project/app/trained_model/plant_disease_prediction_model.h5"
# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# loading the class names
class_indices = json.load(open("C:/Users/ragav/Downloads/plant-disease-prediction-cnn-deep-leanring-project/plant-disease-prediction-cnn-deep-leanring-project/app/class_indices2.json"))

st.markdown(
    """
    <style>
    .stApp {
        background-color: #7ED957; /* Lighter shade of green */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    # Load the image
    img = Image.open(image_path)
    # Resize the image
    img = img.resize(target_size)
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Scale the image values to [0, 1]
    img_array = img_array.astype('float32') / 255.
    return img_array


# Function to Predict the Class of an Image
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

def returnPrediction(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    return predictions

API_KEY = 'DJBCpADeKek1kJxMoqvaXYCpzVkKQafL1UyB9V9wZlU'
BASE_URL = 'https://trefle.io/api/v1/plants'

def get_tree_details(edibility):
    params = {
        'token': API_KEY,
        'filter[edibile_part]': edibility
    }
    response = requests.get(BASE_URL, params=params)
    if response.status_code == 200:
        data = response.json()
        if data['data']:
            tree = data['data'][0]
            # Extract and display relevant details
            # st.write(f"Common Name: {tree['common_name']}")
            # st.write(f"Scientific Name: {tree['scientific_name']}")
            # st.write(f"Family: {tree['family']}")
            # Add more details as needed
        else:
            st.write("Tree not found.")
    else:
        st.write("Failed to retrieve tree details.")

# Streamlit App
st.title('Plant Questionnaire')

'''uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img)

    with col2:
        if st.button('Classify'):
            # Preprocess the uploaded image and predict the class
            prediction = predict_image_class(model, uploaded_image, class_indices)
            predictions = returnPrediction(model, uploaded_image, class_indices)
            st.success(f'Prediction: {str(prediction)}')
            st.write("Steps to take care of your plant: \nStep 1: Water your plant.")'''


with st.form("my_form"):

        st.write("Do you want it to be edible?")
        edibility = st.selectbox("Options", ["Edible", "Non-Edible"])

        st.write("Duration of the plant (biennial, perennial, annual)")
        watering_frequency = st.selectbox("Lifespan", ["Annual", "Biennial", "Perennial"])

        st.write("Toxicity of the plant")
        allergies = st.selectbox("Toxicity", ["Toxic", "Non-Toxic"])

        st.write("Height of the plant?")
        budget = st.number_input("Height (in cm)", value=0, step=1)

        submitted = st.form_submit_button("Submit")
        if submitted:
            st.write("Based on your preferences:")
            st.write("Climate Zone:", edibility)
            st.write("Watering Frequency:", watering_frequency)
            st.write("Allergies:", allergies)
            st.write("Budget:", budget)     

EDIBLE_URL = "https://trefle.io/api/plants?filter[edible]=true"
def get_edible_trees(edibility):
    # Make a request to Trefle AI API to get a list of trees
    params = {
        'token': API_KEY,
        'filter[edible]': edibility
    }
    response = requests.get(BASE_URL, params=params)

    # Check if request was successful
    if response.status_code == 200:
        data = response.json()
        i = 0;
        if data['data']:
            for trees in data:
                tree = data['data'][i]
                # Extract and display relevant details
                # st.write(f"Common Name: {tree['common_name']}")
                # st.write(f"Scientific Name: {tree['scientific_name']}")
                # st.write(f"Family: {tree['family']}")
                # st.image(f"{tree['image_url']}", caption={tree['common_name']}, width=220)
                i = i + 1
                # Add more details as needed
        else:
            st.write("Tree not found.")
    else:
        st.write("Error.")

get_edible_trees('true');
