import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import gdown

# Download model from Google Drive if not already present
model_path = "salary_modell.pkl"
if not os.path.exists(model_path):
    file_id = "1g2crsr1nXl7Rq2lWETEzE2y519FkB8SL"
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, model_path, quiet=False)

# Load model and scaler
model = joblib.load("salary_modell.pkl")
scaler = joblib.load("scalerr.pkl")  # Ensure this is under 25MB or repeat same logic if needed

# Page Config
st.set_page_config(
    page_title="AI Salary Predictor",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling
st.markdown("""
<style>
body {
    background-color: #f4f4f4;
}
h1, h2, h3 {
    color: #1e1e1e;
    font-family: 'Segoe UI', sans-serif;
}
div[data-testid="stSidebar"] {
    background-color: #2c3e50;
}
.sidebar .sidebar-content {
    color: white;
}
.big-font {
    font-size: 20px !important;
}
.result-box {
    background: linear-gradient(to right, #6dd5fa, #2980b9);
    padding: 20px;
    border-radius: 15px;
    color: white;
    text-align: center;
    font-size: 24px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}
.button-style div.stButton > button {
    background-color: #2ecc71;
    color: white;
    font-weight: bold;
    padding: 10px 20px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# Sidebar Navigation
with st.sidebar:
    st.title("💼 Salary Predictor")
    page = st.radio("Navigate to", ["🏠 Home", "📊 Predict Salary", "ℹ️ About"])

# Home Page
if page == "🏠 Home":
    st.markdown("## 👋 Welcome to the Smart Salary Estimator!")
    st.markdown("""
    This tool uses a machine learning model trained on realistic employee data to estimate salaries based on:
    - 📚 Education
    - 🧠 Experience
    - 💼 Job Level
    - 🌆 Location
    - 🏢 Industry
    - 👨‍💻 Work Habits

    Explore the sections on the left to predict or learn more.
    """)

# Prediction Page
elif page == "📊 Predict Salary":
    st.markdown("## 📊 Enter Employee Details Below")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("👤 Age", 20, 60, 30)
        education_level = st.selectbox("🎓 Education Level", ["High School", "Diploma", "Bachelor's", "Master's", "PhD"])
        job_level = st.selectbox("💼 Job Level", ["Junior", "Mid", "Senior", "Lead", "Executive"])

    with col2:
        experience = st.slider("📈 Years of Experience", 0, 40, 5)
        gender = st.selectbox("⚧️ Gender", ["Male", "Female", "Other"])
        industry = st.selectbox("🏭 Industry", ["IT", "Finance", "Healthcare", "Manufacturing", "Education"])

    with col3:
        city = st.selectbox("🏙️ City", ["Bangalore", "Mumbai", "Hyderabad", "Delhi", "Chennai"])
        hours_per_week = st.slider("🕒 Hours per Week", 30, 60, 40)
        remote_ratio = st.selectbox("🏡 Remote Ratio", [0, 50, 100])

    st.markdown("### 👇 Click to Predict Salary")

    with st.container():
        if st.button("💰 Predict Salary", use_container_width=True):
            input_df = pd.DataFrame([{
                "age": age,
                "education_level": ["High School", "Diploma", "Bachelor's", "Master's", "PhD"].index(education_level) + 1,
                "years_experience": experience,
                "job_level": ["Junior", "Mid", "Senior", "Lead", "Executive"].index(job_level) + 1,
                "gender": {"Male": 0, "Female": 1, "Other": 2}[gender],
                "industry": {"IT": 2, "Finance": 1, "Healthcare": 3, "Manufacturing": 4, "Education": 0}[industry],
                "city": {"Bangalore": 0, "Mumbai": 3, "Hyderabad": 2, "Delhi": 1, "Chennai": 4}[city],
                "hours_per_week": hours_per_week,
                "remote_ratio": remote_ratio,
            }])

            scaled_input = scaler.transform(input_df)
            prediction = model.predict(scaled_input)[0]

            st.markdown(f'<div class="result-box">💸 Estimated Salary: ₹ {int(prediction):,}</div>', unsafe_allow_html=True)

# About Page
elif page == "ℹ️ About":
    st.markdown("## ℹ️ About This App")
    st.markdown("""
    This AI app uses a machine learning model to help:
    - 🎯 HRs evaluate compensation offers
    - 👨‍💼 Employees benchmark salaries
    - 💻 Students understand industry standards

    **Built with:** Streamlit + Random Forest + Love ❤️
    """)
