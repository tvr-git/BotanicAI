import streamlit as st
import requests
import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf


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


# Function to fetch weather forecast data from Open-Meteo API
def fetch_weather_forecast(latitude, longitude):
   url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
   response = requests.get(url)
   if response.status_code == 200:
       return response.json()
   else:
       st.error("Failed to fetch weather forecast data. Please try again.")
       return None


# Function to provide recommendations based on weather
def get_weather_recommendations(humidity, temperature):
   humidity_recommendation = ""
   temperature_recommendation = ""


   # Humidity-based recommendations
   if humidity >= 70:
       humidity_recommendation = "High humidity levels detected. Consider reducing watering frequency to avoid overwatering your plants."
   elif 50 <= humidity < 70:
       humidity_recommendation = "Moderate humidity levels detected. Water your plants as usual."
   else:
       humidity_recommendation = "Low humidity levels detected. Ensure your plants are adequately watered to prevent dehydration."


   # Temperature-based recommendations
   if temperature >= 25:
       temperature_recommendation = "High temperatures detected. Consider providing shade or moving plants indoors to prevent heat stress."
   elif 10 <= temperature < 25:
       temperature_recommendation = "Moderate temperatures detected. Plants can be kept outdoors or indoors depending on preference."
   else:
       temperature_recommendation = "Low temperatures detected. Keep plants indoors to protect them from cold temperatures."


   return humidity_recommendation, temperature_recommendation


# Streamlit App
st.title('Plant Disease Classifier and Weather Forecast')


# Load the pre-trained model
model_path = r"C:\Users\ragav\Downloads\plant-disease-prediction-cnn-deep-leanring-project\plant-disease-prediction-cnn-deep-leanring-project\app\trained_model\plant_disease_prediction_model.h5"
model = tf.keras.models.load_model(model_path)


# loading the class names
class_indices = json.load(open(r"C:\Users\ragav\Downloads\plant-disease-prediction-cnn-deep-leanring-project\plant-disease-prediction-cnn-deep-leanring-project\app\class_indices.json"))


uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])


if uploaded_image is not None:
   image = Image.open(uploaded_image)
   col1, col2 = st.columns(2)


   with col1:
       resized_img = image.resize((150, 150))
       st.image(resized_img)


   with col2:
       if st.button('Classify'):
           prediction = predict_image_class(model, uploaded_image, class_indices)
           st.success(f'Prediction: {str(prediction)}')


# Input fields for latitude and longitude
latitude = st.text_input('Enter Latitude', value='')
longitude = st.text_input('Enter Longitude', value='')


# Button to fetch weather forecast
if st.button('Get Forecast'):
   if not latitude or not longitude:
       st.error("Please provide latitude and longitude.")
   else:
       weather_forecast = fetch_weather_forecast(latitude, longitude)
       if weather_forecast:
           st.write("Forecast for the next 5 days:")
           for i in range(5):
               date = weather_forecast['hourly']['time'][i]
               temperature = weather_forecast['hourly']['temperature_2m'][i]
               humidity = weather_forecast['hourly']['relative_humidity_2m'][i]
               wind_speed = weather_forecast['hourly']['wind_speed_10m'][i]
               st.write(f"Date: {date}")
               st.write(f"Temperature: {temperature} Â°C")
               st.write(f"Humidity: {humidity}%")
               st.write(f"Wind Speed: {wind_speed} m/s")
               st.write("--------------------")


           # Provide recommendations based on humidity and temperature
           humidity_recommendation, temperature_recommendation = get_weather_recommendations(humidity, temperature)
           st.write("Humidity Recommendation:", humidity_recommendation)
           st.write("Temperature Recommendation:", temperature_recommendation)
       else:
           st.error("Failed to fetch weather forecast data. Please try again.")
