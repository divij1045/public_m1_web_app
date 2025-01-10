# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 12:33:59 2025

@author: Divij
"""

import numpy as np
import pickle
import streamlit as st

# loading the saved model
loaded_model = pickle.load(open('E:/Deploying Machine Learning Model/trained_model1.sav', 'rb'))

def heartdisease_prediction(input_data):

    input_data = (62,0,0,140,268,0,0,160,0,3.6,0,2,2)

# change the input data to a numpy array
    input_data_as_numpy_array= np.asarray(input_data)

# reshape the numpy array as we are predicting for only on instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0]== 0):
     return 'The Person does not have a Heart Disease'
    else:
     return 'The Person has Heart Disease'
 
def main():
         
         #giving a title
         st.title('Heart Disease prediction Web App')
        
         
         #getting the input data from the user
         
         Age = st.text_input('Age of the Person')
         Sex = st.text_input('Sex')
         cp = st.text_input('cp Value')
         trestbps = st.text_input('trestbps Value')
         Chol = st.text_input('Chol Level')
         fbs = st.text_input('fbs Value')
         restecg = st.text_input('restecg Value')
         thalach = st.text_input('thalach Value')
         exang = st.text_input('exang Value')
         oldpeak = st.text_input('oldpeak Value')
         slope = st.text_input('slope Value')
         ca = st.text_input('ca Value')
         thal = st.text_input('thal Value')
         
         
         
         #code for prediction
         diagnosis = ''
         
         #creating a button for prediction
         
         if st.button('Heart Disease Test Result'):
             diagnosis = heartdisease_prediction([ Age,Sex,cp,trestbps,Chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal])
             
             st.success(diagnosis)
             
             
if __name__=='__main__':
     main()
     
    
