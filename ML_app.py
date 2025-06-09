import streamlit as st
import tensorflow as tf
import joblib as jb
import numpy as np
from PIL import Image


# load models from C drive. both cnn and mlp model
model = tf.keras.models.load_model("cleave_model_best_6_6.keras")
mlp_model = tf.keras.models.load_model("best_mlp.keras")

# load scalers
scaler = jb.load("minmax_scaler.pkl")
tension_scaler = jb.load("tension_scaler.pkl")


def preprocess_image(uploaded_file):
    # resize image and normalize
    try:
        img = Image.open(uploaded_file)  
        img = img.resize((224, 224))
        img = img.convert("RGB")  # Convert to 3 channels
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 224, 224, 3)
        return img_array
    except Exception as e:
        st.error(f"Image preprocessing failed: {e}")
        return None


def test_prediction(image, tension, cleave_angle):
    '''
    Test function for generating prediction

    Parameters:
    ----------------------------------------------

    image_path: str
      - path to image to predict
    tension: int
      - tension value in grams
    cleave_angle: float
      - angle that was achieved from cleave

    Return: tf.keras.Model
      - predicition from new image of good or bad cleave
    '''
    image = preprocess_image(image)
    def process_features(tension, cleave_angle):
      features = np.array([[cleave_angle, tension]])
      features = scaler.transform(features)
      return features
    features = process_features(tension, cleave_angle)
    prediction = model.predict([image, features])
    return prediction

def PredictTension(model, image, angle):
    # Process image and convert angle and image to tensor with dimension for single batch
    image = preprocess_image(image)
    angle = tf.convert_to_tensor(np.array([[angle]]), dtype=tf.float32)


    predicted_tension = mlp_model.predict([image, angle])
    # Scale tension back to normal units
    predicted_tension = tension_scaler.inverse_transform(predicted_tension)
    # Print tensions
    return predicted_tension[0][0]


# streamlit main page
st.image("https://www.thorlabs.com/images/thorlabs-logo.png", width=250)
st.title("Welcome to the Cleave Analyzer")
st.subheader("Please upload an image to be analyzed along with tension and angle")
uploaded_file = st.file_uploader("Upload Image: ")
tension = st.number_input("Enter cleave tension (g):", 400, 750, step=1)
angle = st.number_input("Enter cleave angle (deg):", 0.0, 6.0, step=0.01)

if uploaded_file is not None and tension and angle:
   
    # pass to cnn model
    prediction = test_prediction(uploaded_file, tension, angle)

    st.write("Prediction:", "Good Cleave" if prediction > 0.5 else "Bad Cleave")

    # if bad cleave, pass to mlp model
    if(prediction < 0.5):
       optimal_tension = PredictTension(model, uploaded_file, angle)
       st.write(f"Optimal Tension: {optimal_tension:.0f}")


