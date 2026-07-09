import streamlit as st
import pandas as pd
import pickle

# Load model and encoders
model = pickle.load(open("churn_model.pkl", "rb"))
label_encoders = pickle.load(open("label_encoders.pkl", "rb"))

st.title("Customer Churn Prediction")

st.write("Enter the customer details below:")

# -----------------------------
# User Inputs
# -----------------------------

gender = st.selectbox("Gender", ["Female", "Male"])

senior = st.selectbox("Senior Citizen", [0, 1])

partner = st.selectbox("Partner", ["Yes", "No"])

dependents = st.selectbox("Dependents", ["Yes", "No"])

tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100)

phone = st.selectbox("Phone Service", ["Yes", "No"])

multiple = st.selectbox(
    "Multiple Lines",
    ["No", "Yes", "No phone service"]
)

internet = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

security = st.selectbox(
    "Online Security",
    ["Yes", "No", "No internet service"]
)

backup = st.selectbox(
    "Online Backup",
    ["Yes", "No", "No internet service"]
)

device = st.selectbox(
    "Device Protection",
    ["Yes", "No", "No internet service"]
)

support = st.selectbox(
    "Tech Support",
    ["Yes", "No", "No internet service"]
)

tv = st.selectbox(
    "Streaming TV",
    ["Yes", "No", "No internet service"]
)

movies = st.selectbox(
    "Streaming Movies",
    ["Yes", "No", "No internet service"]
)

contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

paperless = st.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)

payment = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

monthly = st.number_input("Monthly Charges")

total = st.number_input("Total Charges")

# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict"):

    input_data = pd.DataFrame({

        "gender":[gender],
        "SeniorCitizen":[senior],
        "Partner":[partner],
        "Dependents":[dependents],
        "tenure":[tenure],
        "PhoneService":[phone],
        "MultipleLines":[multiple],
        "InternetService":[internet],
        "OnlineSecurity":[security],
        "OnlineBackup":[backup],
        "DeviceProtection":[device],
        "TechSupport":[support],
        "StreamingTV":[tv],
        "StreamingMovies":[movies],
        "Contract":[contract],
        "PaperlessBilling":[paperless],
        "PaymentMethod":[payment],
        "MonthlyCharges":[monthly],
        "TotalCharges":[total]

    })

    # Encode categorical columns
    for column in input_data.columns:
        if column in label_encoders:
            input_data[column] = label_encoders[column].transform(input_data[column])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("Customer is likely to Churn.")
    else:
        st.success("Customer is Not likely to Churn.")