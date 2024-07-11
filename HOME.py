
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import joblib
from PIL import Image

st.title("Residual Frictional Angle Prediction")
nav = st.sidebar.radio("Navigation",["Home","Prediction","Info"])

if nav=="Home":
    image = Image.open('image.png') 
    st.image(image, width = 800)
    st.markdown(
    """📈**Residual Shear Strength Prediction**
### **Predicting Residual Frictional Angle with ML**

**This web application predicts the residual frictional angle of soil based on key soil index properties: Liquid Limit, Plastic Limit, and Change in Plasticity Index. The residual frictional angle is a crucial parameter in geotechnical engineering, influencing the stability of slopes, retaining walls, and foundations. Our prediction model is built using advanced machine learning techniques to provide accurate and reliable estimates..**

##  **How It's Built**

Is built with these core frameworks and modules:

- **Data Collection and Preprocessing** - To gather quality data for model training 
- **Training Machine Learning Model** - To build ML model utilising ensemble methods
- **Saving Model as .H5 Format** - Saved bulit model and unpickling it with 
- **Building Web Application with Streamlit** - To create the web app UI and interactivity

The app workflow is:

1. Click on Make Predictions Button 
2. Enter numeric values for LL, PI, Change in PI and CL
3. The applicaions runs your values and brings out the final answer 
5. Results are given in degrees 

##  **Key Features**

- **Real-time data** - Instantly predict the residual frictional angle by entering the soil index properties.
- **Accurate Predictions** - Model has the highest accuracy rate 
- **Model Improvement** - Model is regularly accessed and improved on




"""
)

if nav== "Prediction":
    st.header("   Predict Your Frictional Angle with Index Properties")
    st.markdown(
        """
        <div style="background-color:tomato;padding:10px">
        <h2 style="color:white;text-align:center;">Streamlit Geotech Machine Learning App</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    def predict_residues(Liquid_Limit, Plasticity_Index, Change_in_Plasticity_Index, Clay_Fraction):
    # Load model
      model = tf.keras.models.load_model("model.h5")
      scaler = joblib.load("scaler.pkl")
      
       # Preprocess the input (without scaling)
      input_data = np.array([[Liquid_Limit, Plasticity_Index, Change_in_Plasticity_Index, Clay_Fraction]])
      scaled_input = scaler.transform(input_data)

    # Make a prediction using the model
      prediction = model.predict(scaled_input)
      
      return prediction  # Return the prediction
    # Function to handle user input and prediction
    def main():
         Liquid_Limit= st.text_input("Liquid Limit (%)", "")
         Plasticity_Index = st.text_input("Plasticity index (%)", "")
         Change_in_Plasticity_Index = st.text_input("Change in Plasticity Index (%)", "")
         Clay_Fraction = st.text_input("Clay Fraction (%)", "")

         try:
            Liquid_Limit = float(Liquid_Limit)
            Plasticity_Index = float(Plasticity_Index)
            Change_in_Plasticity_Index = float(Change_in_Plasticity_Index)
            Clay_Fraction = float( Clay_Fraction)
         except ValueError:
            st.error("Please enter numeric value percentages for LL, PI, DPI, and CF.")
            st.stop()

         result = ""
         if st.button("Predict"):
            result = predict_residues(Liquid_Limit, Plasticity_Index, Change_in_Plasticity_Index, Clay_Fraction)
            st.success("The Predicted Frictional Angle is {}".format(result))
           
         
    # Call the main function to display the input section and prediction output
    main()

elif nav== "Info":
    st.markdown(
        """
        <div style="background-color:tomato;padding:10px">
        <h2 style="color:white;text-align:center;">University of Mines and Technology</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.header("Geological Engineering Department")
    image = Image.open('Lmg.jpg') 
    st.image(image, width = 800)