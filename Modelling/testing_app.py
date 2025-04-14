import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load the model
model = pickle.load(open("D:/E-Commerce churn project/Modelling/churn_prediction_model.pkl", "rb"))
x_train = pd.read_csv("x_train.csv")

# Prediction function
def pred(category, item, quantity, price, payment_method, card_type, shopping_mall, city, province_state,
         country, gender, age, days_since_last_purchase, tenure, discount_used, purchase_frequency,
         avg_purchase_value, recency, purchase_per_tenure, discount_ratio):
    
    features = np.array([[category, item, quantity, price, payment_method, card_type, shopping_mall, city,
                          province_state, country, gender, age, days_since_last_purchase, tenure, discount_used,
                          purchase_frequency, avg_purchase_value, recency, purchase_per_tenure, discount_ratio]])
    
    prediction = model.predict(features)
    return prediction[0]

# Streamlit app UI
st.title("🧪 Customer Churn Model Testing (Using Training Data)")

# Dropdowns for categorical data
category = st.selectbox("Product Category", sorted(x_train["category"].unique()))
item = st.selectbox("Item", sorted(x_train["item"].unique()))
shopping_mall = st.selectbox("Shopping Mall", sorted(x_train["shopping_mall"].unique()))
city = st.selectbox("City", sorted(x_train["city"].unique()))
province_state = st.selectbox("Province/State", sorted(x_train["province_state"].unique()))
country = st.selectbox("Country", sorted(x_train["country"].unique()))
gender = st.selectbox("Gender", sorted(x_train["gender"].unique()))
payment_method = st.selectbox("Payment Method", sorted(x_train["payment_method"].unique()))
card_type = st.selectbox("Card Type", sorted(x_train["card_type"].unique()))
discount_used = st.selectbox("Used Discount?", sorted(x_train["discount_used"].unique()))

# Numeric Inputs
quantity=st.selectbox("quantity",x_train["quantity"])
price=st.selectbox("price",x_train["price"])
age=st.selectbox("age",x_train["age"])
days_since_last_purchase=st.selectbox("days_since_last_purchase",x_train["days_since_last_purchase"])
tenure=st.selectbox("tenure",x_train["tenure"])
purchase_frequency=st.selectbox("purchase_frequency",x_train["purchase_frequency"])
avg_purchase_value=st.selectbox("avg_purchase_value",x_train["avg_purchase_value"])
recency=st.selectbox("recency",x_train["recency"])
purchase_per_tenure=st.selectbox("purchase_per_tenure",x_train["purchase_per_tenure"])
discount_ratio=st.selectbox("discount_ratio",x_train["discount_ratio"])

# Predict button
if st.button("Predict"):
    result = pred(category, item, quantity, price, payment_method, card_type, shopping_mall, city,province_state, country, gender, age, days_since_last_purchase, tenure, discount_used,purchase_frequency, avg_purchase_value, recency, purchase_per_tenure, discount_ratio)
    
    if result == 1:
        st.error("⚠️ The customer is likely to churn.")
    else:
        st.success("✅ The customer is likely to stay.")
