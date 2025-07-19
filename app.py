import streamlit as st
import joblib
import tensorflow as tf
import pandas as pd
import numpy as np
def inject_background():
    st.markdown("""
    <style>
    body {
      overflow: hidden;
    }
    .rocket-background {
      position: fixed;
      top: 0;
      left: 0;
      width: 100vw;
      height: 100vh;
      z-index: 0;
      pointer-events: none;
      background-color: black;
      animation: skyToSpace 2s ease-in-out forwards 1s, spaceToSky 2s ease-in-out forwards 10s;
    }

    .rocket-background .space-bg {
      background: url('https://raw.githubusercontent.com/Ibadmoin/spacex-launch-predictor/main/assets/space.jpg') center center no-repeat;
      background-size: cover;
      opacity: 0;
      position: absolute;
      width: 100%;
      height: 100%;
      z-index: 1;
      animation: fadeIn 1s ease-in-out forwards 1s, fadeOut 1s ease-in-out forwards 10s;
    }

    .rocket-background .earth {
      position: absolute;
      bottom: 0;
      opacity: 0;
      width: 40%;
      z-index: 3;
      animation: fadeIn 1s ease-in-out forwards 1s,
                 rotateEarth 5s linear 2s infinite,
                 earthDown 5s ease-in-out forwards 3s,
                 fadeOut 1s ease-in-out forwards 10s;
    }

    .rocket-background .rocket {
      position: absolute;
      bottom: 100px;
      left: 5%;
      mix-blend-mode: lighten;
      opacity: 0;
      width: 30%;
      z-index: 2;
      animation: fadeIn 2s ease-in-out forwards 3s,
                 launchRocket 6s ease-in-out forwards 4s,
                 fadeOut 1s ease-in-out forwards 14s;
    }

    @keyframes fadeIn { to { opacity: 1; } }
    @keyframes fadeOut { to { opacity: 0; } }
    @keyframes skyToSpace { to { background: black; } }
    @keyframes spaceToSky { from { background: black; } to { background:#0e1117; } }
    @keyframes rotateEarth { 0% { transform: rotate(0deg); } 100% { transform: rotate(50deg); } }
    @keyframes earthDown { 0% { bottom: 0; } 100% { bottom: -400px; } }
    @keyframes launchRocket { 0% { bottom: 100px; } 100% { bottom: 1000px; } }
    </style>

    <div class="rocket-background">
      <div class="space-bg"></div>
      <img class="earth" src="https://raw.githubusercontent.com/Ibadmoin/spacex-launch-predictor/main/assets/earth.png" />
      <img class="rocket" src="https://raw.githubusercontent.com/Ibadmoin/spacex-launch-predictor/main/assets/rocket.gif" />
    </div>
    """, unsafe_allow_html=True)


st.set_page_config(page_title="SpaceX Launch Success Prediction", page_icon="🚀", layout="centered", initial_sidebar_state="collapsed") 
inject_background()

rocket_animation_html = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <style>
    html, body {
      margin: 0;
      padding: 0;
      overflow: hidden;
      height: 100%;
      background: transparent;
    }

    .launch-scene {
      position: fixed;
      top: 0;
      left: 0;
      width: 100vw;
      height: 100vh;
      background: #0e1117;
      z-index: 9999;
      pointer-events: none;
      animation: skyToSpace 2s ease-in-out forwards 1s, spaceToSky 2s ease-in-out forwards 10s;
    }

    .space-bg {
      background: url('https://github.com/Ibadmoin/spacex-launch-predictor/blob/main/assets/space.jpg') center center no-repeat;
      background-size: cover;
      opacity: 0;
      position: absolute;
      width: 100%;
      height: 100%;
      z-index: 1;
      animation: fadeIn 1s ease-in-out forwards 1s, fadeOut 1s ease-in-out forwards 10s;
    }

    .earth {
      position: absolute;
      bottom: 0;
      opacity: 0;
      width: 40%;
      z-index: 3;
      animation: fadeIn 1s ease-in-out forwards 1s,
                 rotateEarth 5s linear 2s infinite,
                 earthDown 5s ease-in-out forwards 3s,
                 fadeOut 1s ease-in-out forwards 10s;
    }

    .rocket {
      position: absolute;
      bottom: 100px;
      left: 5%;
      background: transparent;
      mix-blend-mode: lighten;
      opacity: 0;
      width: 30%;
      z-index: 2;
      animation: fadeIn 2s ease-in-out forwards 3s,
                 launchRocket 6s ease-in-out forwards 4s,
                 fadeOut 1s ease-in-out forwards 14s;
    }

    @keyframes fadeIn { to { opacity: 1; } }
    @keyframes fadeOut { to { opacity: 0; } }
    @keyframes skyToSpace { to { background: black; } }
    @keyframes spaceToSky { from { background: black; } to { background:#0e1117; } }
    @keyframes rotateEarth { 0% { transform: rotate(0deg); } 100% { transform: rotate(50deg); } }
    @keyframes earthDown { 0% { bottom: 0; } 100% { bottom: -400px; } }
    @keyframes launchRocket { 0% { bottom: 100px; } 100% { bottom: 900px; } }
  </style>
</head>
<body>
  <div class="launch-scene">
    <div class="space-bg"></div>
    <img class="earth" src="https://github.com/Ibadmoin/spacex-launch-predictor/blob/main/assets/earth.jpg" alt="Earth" />
    <img class="rocket" src="https://github.com/Ibadmoin/spacex-launch-predictor/blob/main/assets/rocket.gif" alt="Rocket" />
  </div>
</body>
</html>
"""


# Load model and preprocessor
@st.cache_resource
def load_model_and_preprocessor():
    model = tf.keras.models.load_model('model/final_model.keras')
    preprocessor = joblib.load('model/preprocessor.joblib')
    return model, preprocessor

model, preprocessor = load_model_and_preprocessor()


st.title("SpaceX Launch Success Prediction 🚀")

# User inputs
flight_number = st.number_input("Flight Number", min_value=1, max_value=300, value=130)
launch_hour = st.slider("Launch Hour (0-23)", 0, 23, 17)
primary_payload_mass = st.number_input("Primary Payload Mass (kg)", min_value=0, max_value=150000, value=50000)
rocket_reusable = st.selectbox("Rocket Reusable?", options=[0, 1], index=1)
launchpad_latitude = st.number_input("Launchpad Latitude", value=28.5623)
launchpad_longitude = st.number_input("Launchpad Longitude", value=-80.5774)
temperature = st.number_input("Temperature (°C)", value=26.0)
wind_speed = st.number_input("Wind Speed (km/h)", value=14.0)
cloud_cover = st.slider("Cloud Cover (%)", 0, 100, 20)
is_night_launch = st.selectbox("Is Night Launch?", options=[0, 1], index=1)

# Adjustable threshold slider in sidebar
threshold = st.sidebar.slider("Classification Threshold", min_value=0.0, max_value=1.0, value=0.374, step=0.01)

def predict(input_dict, threshold):
    input_df = pd.DataFrame([input_dict])
    input_processed = preprocessor.transform(input_df)
    prob = model.predict(input_processed)[0][0]
    prediction = int(prob > threshold)
    return prediction, prob

if st.button("Predict Launch Success"):
    user_input = {
        'flight_number': flight_number,
        'launch_hour': launch_hour,
        'primary_payload_mass': primary_payload_mass,
        'rocket_reusable': rocket_reusable,
        'launchpad_latitude': launchpad_latitude,
        'launchpad_longitude': launchpad_longitude,
        'temperature': temperature,
        'wind_speed': wind_speed,
        'cloud_cover': cloud_cover,
        'is_night_launch': is_night_launch
    }

    pred, prob = predict(user_input, threshold)

    st.write(f"**Prediction:** {'Success 🚀' if pred == 1 else 'Failure ❌'}")
    st.write(f"**Probability:** {prob:.4f}")

    

