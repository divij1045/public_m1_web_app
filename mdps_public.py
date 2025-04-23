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

# Utility function to safely convert input to float
def safe_float(val):
    try:
        return float(val)
    except:
        return 0.0

# ========================== Diabetes Prediction ==========================
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using ML')

    inputs = [
        st.text_input('Number of Pregnancies'),
        st.text_input('Glucose Level'),
        st.text_input('Blood Pressure value'),
        st.text_input('Skin Thickness value'),
        st.text_input('Insulin Level'),
        st.text_input('BMI value'),
        st.text_input('Diabetes Pedigree Function value'),
        st.text_input('Age of the Person')
    ]

    diab_diagnosis = ''

    if st.button('Diabetes Test Result'):
        input_data = [safe_float(i) for i in inputs]
        diab_prediction = diabetes_model.predict([input_data])
        diab_diagnosis = 'The person is diabetic' if diab_prediction[0] == 1 else 'The person is not diabetic'

    st.success(diab_diagnosis)

# ========================== Heart Disease Prediction ==========================
if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using ML')

    features = [
        'Age', 'Sex', 'Chest Pain types', 'Resting Blood Pressure', 'Serum Cholesterol in mg/dl',
        'Fasting Blood Sugar > 120 mg/dl', 'Resting Electrocardiographic results',
        'Maximum Heart Rate achieved', 'Exercise Induced Angina',
        'ST depression induced by exercise', 'Slope of the peak exercise ST segment',
        'Major vessels colored by fluoroscopy', 'Thal: 0 = normal; 1 = fixed defect; 2 = reversible defect'
    ]

    inputs = [st.text_input(f) for f in features]
    heart_diagnosis = ''

    if st.button('Heart Disease Test Result'):
        input_data = [safe_float(i) for i in inputs]
        heart_prediction = heart_disease_model.predict([input_data])
        heart_diagnosis = 'The person has heart disease' if heart_prediction[0] == 1 else 'The person does not have any heart disease'

    st.success(heart_diagnosis)

# ========================== Parkinson's Prediction ==========================
if selected == "Parkinson’s Prediction":
    st.title("Parkinson's Disease Prediction using ML")

    features = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)',
                'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)',
                'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR',
                'HNR', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE']

    inputs = [st.text_input(f) for f in features]
    parkinsons_diagnosis = ''

    if st.button("Parkinson's Test Result"):
        input_data = [safe_float(i) for i in inputs]
        parkinsons_prediction = parkinsons_model.predict([input_data])
        parkinsons_diagnosis = "The person has Parkinson's disease" if parkinsons_prediction[0] == 1 else "The person does not have Parkinson's disease"

    st.success(parkinsons_diagnosis)

# ========================== COVID-19 Prediction ==========================
if selected == 'COVID-19 Prediction':
    st.title("COVID-19 Prediction using ML")

    features = [
        'Fever', 'Tiredness', 'Dry-Cough', 'Difficulty-in-Breathing', 'Sore-Throat',
        'None_Symptom', 'Pains', 'Nasal-Congestion', 'Runny-Nose', 'Diarrhea',
        'None_Experiencing', 'Age_0-9', 'Age_10-19', 'Age_20-24', 'Age_25-59',
        'Age_60+', 'Gender_Female', 'Gender_Male', 'Gender_Transgender',
        'Contact_Dont-Know', 'Contact_No', 'Contact_Yes'
    ]

    inputs = [st.text_input(f) for f in features]
    covid_diagnosis = ''

    if st.button("COVID-19 Test Result"):
        input_data = [safe_float(i) for i in inputs]
        covid_prediction = covid_model.predict([input_data])
        covid_diagnosis = "The person is COVID-19 Positive" if covid_prediction[0] == 1 else "The person is COVID-19 Negative"

    st.success(covid_diagnosis)
















