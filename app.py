import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Page config
st.set_page_config(page_title="Credit Risk Predictor", layout="centered")

# ---- 3D STYLE BACKGROUND CSS ----
st.markdown("""
<style>
.stApp {
    background: linear-gradient(-45deg, #0f2027, #203a43, #2c5364, #1c1c1c);
    background-size: 400% 400%;
    animation: gradient 10s ease infinite;
    color: white;
}

@keyframes gradient {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* Card effect */
.block-container {
    background: rgba(255,255,255,0.05);
    padding: 2rem;
    border-radius: 15px;
    backdrop-filter: blur(10px);
}

/* Buttons */
.stButton>button {
    background-color: #00c6ff;
    color: white;
    border-radius: 10px;
    padding: 0.5rem 1rem;
    border: none;
}

.stButton>button:hover {
    background-color: #0072ff;
}

/* Input fields */
.stNumberInput input {
    background-color: rgba(255,255,255,0.1);
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ---- TITLE ----
st.title("💳 Credit Default Risk Predictor")
st.markdown("### AI-powered risk detection with modern UI")

# ---- INPUT ----
col1, col2 = st.columns(2)

with col1:
    revolving = st.number_input("Revolving Utilization", min_value=0.0)
    age = st.number_input("Age", min_value=18)
    income = st.number_input("Monthly Income", min_value=0.0)

with col2:
    late_30 = st.number_input("30-59 Days Late", min_value=0)
    late_60 = st.number_input("60-89 Days Late", min_value=0)
    late_90 = st.number_input("90 Days Late", min_value=0)

# ---- PREDICT ----
if st.button("🔍 Predict Risk"):
    try:
        input_data = np.array([[revolving, age, income, late_30, late_60, late_90]])

        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0][1]

        st.markdown("## 📊 Result")

        if prediction[0] == 1:
            st.error(f"⚠️ High Risk\nProbability: {probability:.2f}")
        else:
            st.success(f"✅ Low Risk\nProbability: {probability:.2f}")

        st.progress(int(probability * 100))

    except Exception as e:
        st.warning("⚠️ Feature mismatch or model error")
        st.text(str(e))

# ---- FOOTER ----
st.markdown("---")
st.markdown("✨ Designed with 3D-style UI using Streamlit")
