# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 01:21:15 2024

@author: Ijat
"""

pip install -r requirements.txt

import streamlit as st
import numpy as np
import tensorflow as tf
from joblib import load

# Load the models without passing weight_decay argument
model1 = tf.keras.models.load_model('Trained_model_DNN_Toddler.h5', compile=False)
model2 = load('knn_model.h5')
model3 = tf.keras.models.load_model('Trained_model_DNN_Adult.h5', compile=False)
model4 = load('best_model.h5')

def predict(input_data, model):
    input_as_numpy = np.asarray(input_data[:53], dtype=np.float32)  # Convert to float array
    input_data_reshaped = input_as_numpy.reshape(1, -1)
    # Use the model to predict
    y_pred = model.predict(input_data_reshaped)
    y_pred = (y_pred > 0.5)
    return y_pred

def predict1(input_data, model):
    input_as_numpy = np.asarray(input_data, dtype=np.float32)  # Convert to float array
    input_data_reshaped = input_as_numpy.reshape(1, -1)
    # Use the model to predict
    y_pred = model.predict(input_data_reshaped)
    y_pred = (y_pred > 0.5)
    return y_pred[0]  # Return the predicted class instead of a boolean array

# Add decoration to the Streamlit page
st.markdown(
    """
    <style>
        .title {
            font-size: 32px;
            color: #336699;
            text-align: center;
            padding-top: 20px;
            padding-bottom: 20px;
        }
        .sidebar .sidebar-content {
            background-color: #f0f2f6;
        }
        .sidebar .sidebar-content .block-container {
            margin-top: 20px;
        }
        .main .block-container {
            padding-top: 20px;
            padding-bottom: 20px;
        }
        .btn {
            background-color: #336699;
            color: #ffffff;
            font-size: 18px;
            font-weight: bold;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
        }
        .btn:hover {
            background-color: #254e77;
        }
    </style>
    """,
    unsafe_allow_html=True
)

def main():
    st.title('Autism Spectrum Disorder Predictor')

    page = st.sidebar.radio("Select Page", ("ASD Predictor", "Multi-class Predictor"))

    if page == "ASD Predictor":
        predictor = st.selectbox("Select ASD Predictor", ["ASD Toddler", "ASD Adolescence", "ASD Adult"])

        if predictor == "ASD Toddler":
            st.header("ASD Toddler Predictor")
            st.write('Enter the input data for ASD Toddler (0 or 1):')
            input_data = []
            for i in range(10):
                input_data.append(st.number_input(f'Feature {i+1}', min_value=0, max_value=1, step=1, value=0))

            if st.button('Predict'):
                y_pred = predict(input_data, model1)
                if y_pred[0] == 0:
                    st.write('Prediction: Non-ASD')
                else:
                    st.write('Prediction: ASD')

        elif predictor == "ASD Adolescence":
            st.header("ASD Adolescence Predictor")
            st.write('Enter the input data for ASD Adolescence (0 or 1):')
            input_data = []
            for i in range(10):
                input_data.append(st.number_input(f'Feature {i+1}', min_value=0, max_value=1, step=1, value=0))

            if st.button('Predict'):
                y_pred = predict(input_data, model2)
                if y_pred[0] == 0:
                    st.write('Prediction: Non-ASD')
                else:
                    st.write('Prediction: ASD')

        elif predictor == "ASD Adult":
            st.header("ASD Adult Predictor")
            st.write('Enter the input data for ASD Adult (0 or 1):')
            input_data = []
            for i in range(10):
                input_data.append(st.number_input(f'Feature {i+1}', min_value=0, max_value=1, step=1, value=0))

            if st.button('Predict'):
                y_pred = predict(input_data, model3)
                if y_pred[0] == 0:
                    st.write('Prediction: Non-ASD')
                else:
                    st.write('Prediction: ASD')
        
    elif page == "Multi-class Predictor":
        st.header("Multi-class Predictor")
        st.write('Enter the input data for Multi-class Predictor (0 or 1):')
        input_data = []
        for i in range(54):
            input_data.append(st.number_input(f'Feature {i+1}', min_value=0, max_value=1, step=1, value=0))

        if st.button('Predict'):
            y_pred = predict1(input_data, model4)
            if y_pred == 0:
                st.write('Prediction: Mild')
            elif y_pred == 1:
                st.write('Prediction: Moderate')
            else:
                st.write('Prediction: Severe')


if __name__ == '__main__':
    main()
