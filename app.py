import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import pickle 

model = tf.keras.models.load_model('model.h5')

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
    
with open('onehot_encoder_geography.pkl', 'rb') as f:
    onehot_encoder_geo = pickle.load(f) 
     
with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)  
    
    
st.sidebar.title("Customer Churn Prediction")
st.sidebar.write("Please enter the following information:")

age = st.sidebar.slider("Age", 0, 100, 25)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
geography = st.sidebar.selectbox("Geography", ["France", "Spain", "Germany"])
balance = st.sidebar.number_input("Balance", min_value=0.0, value=1000.0)
credit_score = st.sidebar.slider("Credit Score", 300, 850, 600)
tenure = st.sidebar.slider("Tenure (years)", 0, 10, 3)
num_of_products = st.sidebar.slider("Number of Products", 1, 4, 1)
has_credit_card = st.sidebar.selectbox("Has Credit Card", ["Yes", "No"])
is_active_member = st.sidebar.selectbox("Is Active Member", ["Yes", "No"])
estimated_salary = st.sidebar.number_input("Estimated Salary", min_value=0.0, value=50000.0)

input_data = np.array([[credit_score,geography,gender,age,tenure,  balance,   num_of_products, has_credit_card, is_active_member, estimated_salary]])

input_data_df = pd.DataFrame(input_data, columns=['CreditScore','Geography','Gender','Age','Tenure',  'Balance',   'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'])

input_data_df['Gender'] = label_encoder_gender.transform(input_data_df['Gender'])
input_data_df['HasCrCard'] = input_data_df['HasCrCard'].map({'Yes': 1, 'No': 0})
input_data_df['IsActiveMember'] = input_data_df['IsActiveMember'].map({'Yes': 1, 'No': 0})
onehot_enc_geo = onehot_encoder_geo.transform(input_data_df[['Geography']])   
geo_encoded_df = pd.DataFrame(onehot_enc_geo, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
input_data_df = pd.concat([input_data_df.drop('Geography', axis=1), geo_encoded_df], axis=1)    

input_data_df = scaler.transform(input_data_df)

prediction = model.predict(input_data_df)

st.write(prediction)
st.write("Prediction")
if prediction[0][0] > 0.5:
    st.write("The customer is likely to churn.")
else:
    st.write("The customer is likely to stay with the bank.")
