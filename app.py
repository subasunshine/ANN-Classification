import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

## Load the trained model
model=tf.keras.models.load_model('model.h5')

## load the encoders and scalar
with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)

with open('onehot_encoder_geo.pkl','rb') as file:
    onehot_encoder_geo=pickle.load(file)

with open('scalar.pkl','rb') as file:
    scalar=pickle.load(file)

## Streamlit app
st.title('Customer Churn Prediction')

# Collect user input
geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
gender = st.selectbox("Gender",label_encoder_gender.classes_)
age = st.slider("Age", 18, 92)
credit_score = st.number_input("Credit Score")
balance = st.number_input("Balance")
estimated_salary = st.number_input("Estimated Salary")
tenure = st.slider("Tenure (years with bank)", 0, 10)
num_of_products = st.slider("Number of Products", 1, 4)
has_cr_card = st.radio("Has Credit Card?", [0, 1])
is_active_member = st.radio("Is Active Member?", [0, 1])

# Prepare the input data
input_data = pd.DataFrame([{
    'Gender': label_encoder_gender.transform([gender])[0],
    'Age': age,
    'CreditScore': credit_score,
    'Balance': balance,
    'EstimatedSalary': estimated_salary,
    'Tenure': tenure,
    'NumOfProducts': num_of_products,
    'HasCrCard': has_cr_card,
    'IsActiveMember': is_active_member
}])


# One-hot encode 'Geography'
geo_encoder=onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoder_df=pd.DataFrame(geo_encoder,columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

## Concatenation one-hot encoded columns with input data
input_df=pd.concat([input_data.reset_index(drop=True),geo_encoder_df],axis=1)

## Scaling the input_data
input_data_scaled = scalar.transform(input_df)


## Predict churn
prediction=model.predict(input_data_scaled)

prediction_probability=prediction[0][0]

st.write(f'**Churn Probability:** {prediction_probability:.2f}')

if prediction_probability>0.5:
    print('The customer is likely to churn.')
else:
    print('The customer is not likely to churn.')