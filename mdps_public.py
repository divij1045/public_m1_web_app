# -*- coding: utf-8 -*-
"""
Updated: Added COVID-19 Prediction
"""

import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# Load the saved models
diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open('heart_disease_model.sav', 'rb'))
parkinsons_model = pickle.load(open('parkinsons_model.sav', 'rb'))
covid_model = pickle.load(open('disease_prediction_model.sav', 'rb'))  # Load COVID-19 model

# Sidebar for navigation
with st.sidebar:
    selected = option_menu(
        'Disease Predictor App',
        ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinson’s Prediction', 'COVID-19 Prediction'],
        icons=['activity', 'heart', 'person', 'virus'],
        default_index=0
    )

# ========================== Diabetes Prediction ==========================
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using ML')

    col1, col2, col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
    with col2:
        Glucose = st.text_input('Glucose Level')
    with col3:
        BloodPressure = st.text_input('Blood Pressure value')
    with col1:
        SkinThickness = st.text_input('Skin Thickness value')
    with col2:
        Insulin = st.text_input('Insulin Level')
    with col3:
        BMI = st.text_input('BMI value')
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    with col2:
        Age = st.text_input('Age of the Person')

    diab_diagnosis = ''

    if st.button('Diabetes Test Result'):
        diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        diab_diagnosis = 'The person is diabetic' if diab_prediction[0] == 1 else 'The person is not diabetic'

    st.success(diab_diagnosis)

# ========================== Heart Disease Prediction ==========================
if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.text_input('Age')
    with col2:
        sex = st.text_input('Sex')
    with col3:
        cp = st.text_input('Chest Pain types')
    with col1:
        trestbps = st.text_input('Resting Blood Pressure')
    with col2:
        chol = st.text_input('Serum Cholesterol in mg/dl')
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')
    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')
    with col3:
        exang = st.text_input('Exercise Induced Angina')
    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')
    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')
    with col3:
        ca = st.text_input('Major vessels colored by fluoroscopy')
    with col1:
        thal = st.text_input('Thal: 0 = normal; 1 = fixed defect; 2 = reversible defect')

    heart_diagnosis = ''

    if st.button('Heart Disease Test Result'):
        heart_prediction = heart_disease_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])                          
        heart_diagnosis = 'The person has heart disease' if heart_prediction[0] == 1 else 'The person does not have any heart disease'

    st.success(heart_diagnosis)

# ========================== Parkinson's Prediction ==========================
if selected == "Parkinson’s Prediction":
    st.title("Parkinson's Disease Prediction using ML")

    cols = st.columns(5)
    features = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)',
                'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)',
                'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR',
                'HNR', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE']
    
    input_data = []
    for i, feature in enumerate(features):
        with cols[i % 5]:
            input_data.append(st.text_input(feature))

    parkinsons_diagnosis = ''

    if st.button("Parkinson's Test Result"):
        parkinsons_prediction = parkinsons_model.predict([input_data])
        parkinsons_diagnosis = "The person has Parkinson's disease" if parkinsons_prediction[0] == 1 else "The person does not have Parkinson's disease"

    st.success(parkinsons_diagnosis)

# ========================== COVID-19 Prediction ==========================
if selected == 'COVID-19 Prediction':
    st.title("COVID-19 Prediction using ML")

    cols = st.columns(4)
    covid_features = [
        'Fever', 'Tiredness', 'Dry-Cough', 'Difficulty-in-Breathing', 'Sore-Throat',
        'None_Symptom', 'Pains', 'Nasal-Congestion', 'Runny-Nose', 'Diarrhea',
        'None_Experiencing', 'Age_0-9', 'Age_10-19', 'Age_20-24', 'Age_25-59',
        'Age_60+', 'Gender_Female', 'Gender_Male', 'Gender_Transgender',
        'Contact_Dont-Know', 'Contact_No', 'Contact_Yes'
    ]
    
    covid_input_data = []
    for i, feature in enumerate(covid_features):
        with cols[i % 4]:
            covid_input_data.append(st.text_input(feature))

    covid_diagnosis = ''

    if st.button("COVID-19 Test Result"):
        covid_prediction = covid_model.predict([covid_input_data])
        covid_diagnosis = "The person is COVID-19 Positive" if covid_prediction[0] == 1 else "The person is COVID-19 Negative"

    st.success(covid_diagnosis)
















