import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load the saved model
loaded_model = load_model("my_model.h5")

# Define a function for predicting on new images using the loaded model
def predict_new_image(image_path, model):
    if not os.path.exists(image_path):
        return 'Invalid image path. Please provide a valid path to the image.'

    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    test_img_input = img_array.reshape((1, img_array.shape[0], img_array.shape[1], img_array.shape[2]))
    
    predicted_class = np.argmax(model.predict(test_img_input))
    return predicted_class

# Streamlit app
st.title('Image Classification App')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    # Make prediction when the user clicks the button
    if st.button('Predict'):
        prediction = predict_new_image("temp_image.jpg", loaded_model)
        st.write(f"Predicted Class: {prediction}")
