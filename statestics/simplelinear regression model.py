import streamlit as st
import pickle
import numpy as np

#pickle load the saved model 
model = pickle.load(open(r'/Users/Himavaishnavi/Documents/statestics/linear_regression_model.pkl'))

#set the title of streamlit app
st.title("Salary Prediction App")

#Add a breif description
st.write("This app predicts salary based on years of experience using a simple linearregression model")

#Add input widget for user to enter years of expereince
years_expereince =st.number_input("Enter years of experience:",min_value=0.0, max_value=50.0,value=1.0,step=0.5)

#when the button is clicked, make predictions
if st.button("Predict Salary"):
    #make prediction using trained model
    experience_input = np.array([[years_expereince]]) #convert the input to 2Darray for prediction
    prediction = model.predict(experience_input)
    
    #Display the result
    st.success(f"The predicted salary for{years_expereince} years of expereince is :${prediction[0]:,.2f}")

#Display information about model
st.write("the model was trained using dataset of salaries and years of exp build by vaish")
    