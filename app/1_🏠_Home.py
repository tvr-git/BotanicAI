import streamlit as st
import requests

# Initialize Streamlit page
st.set_page_config(
    page_title="BotanicEye",
    page_icon="🪴",
)

# Streamlit app layout
st.title("Home")
st.sidebar.success("Select a page above.")

# Main content of the app
st.write("""
Our AI, BotanicEye, is dedicated to helping you take care of your plants effectively. Whether you're a seasoned plant enthusiast or just starting out, BotanicEye provides valuable suggestions for plant survival, weakness identification, optimal conditions, and more.

### Features:
- **Plant Care Suggestions:** Receive tailored recommendations for maintaining your plants based on their specific needs.
- **Weakness Identification:** Quickly identify any weaknesses or issues your plants may be experiencing, along with actionable solutions.
- **Optimal Conditions:** Learn about the optimal conditions required for various types of plants to thrive.
- **Plant Recommendations:** Get personalized recommendations on which plants to buy according to your preferences and requirements.

With BotanicEye, you can ensure that your plants receive the care they deserve, enhancing their health and beauty while bringing joy to your home.
""")

# Display image
st.image(r"C:\Users\ragav\Downloads\plant-disease-prediction-cnn-deep-leanring-project\plant-disease-prediction-cnn-deep-leanring-project\app\image.png", use_column_width=True)
