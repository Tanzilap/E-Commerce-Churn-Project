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
st.title("🔮 Predict Customer Churn (Custom Inputs)")

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
# Quantity and Price
quantity = st.number_input("Quantity", min_value=1, max_value=100, value=1, step=1)
price = st.number_input("Price", min_value=0.0, value=50.0, step=0.5)

# Customer Demographics
age = st.slider("Age", min_value=10, max_value=100, value=30)

# Time-based metrics
days_since_last_purchase = st.slider("Days Since Last Purchase", min_value=0, max_value=365, value=30)
tenure = st.slider("Tenure (in days)", min_value=1, max_value=3650, value=365)

# Purchase behavior
purchase_frequency = st.number_input("Purchase Frequency", min_value=0.0, value=2.0, step=0.1)
avg_purchase_value = st.number_input("Average Purchase Value", min_value=0.0, value=100.0, step=0.5)
recency = st.number_input("Recency", min_value=0.0, value=15.0, step=1.0)
purchase_per_tenure = st.number_input("Purchases per Tenure", min_value=0.0, value=1.0, step=0.1)
discount_ratio = st.slider("Discount Ratio", min_value=0.0, max_value=1.0, value=0.2, step=0.01)


# Predict button
if st.button("Predict"):
    result = pred(category, item, quantity, price, payment_method, card_type, shopping_mall, city,province_state, country, gender, age, days_since_last_purchase, tenure, discount_used,purchase_frequency, avg_purchase_value, recency, purchase_per_tenure, discount_ratio)
    
    if result == 1:
        st.error("⚠️ The customer is likely to churn.")
    else:
        st.success("✅ The customer is likely to stay.")
