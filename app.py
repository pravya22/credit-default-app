import streamlit as st
import pickle
import numpy as np

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Credit Risk App", layout="centered")

# ---------------- STYLING ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}

.block-container {
    background: rgba(0,0,0,0.65);
    padding: 2rem;
    border-radius: 15px;
}

h1, h2, h3 {
    color: white;
    text-align: center;
}

p {
    color: #e0e0e0;
}

.stButton>button {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "🔮 Predictor"])

# ---------------- HOME PAGE ----------------
if page == "🏠 Home":

    st.title("💳 Credit Default Risk Predictor")

    st.markdown("### About This App")

    st.markdown("""
This application uses **Machine Learning (XGBoost)** to predict whether a customer is likely to default on a loan.

### 🔍 What it does:
- Analyzes customer financial data
- Predicts default risk
- Helps in decision making

### ⚙️ Features:
- Clean modern UI
- Fast predictions
- Real-world dataset

### 🎯 Use Case:
Banks and financial institutions can use this model to:
- Reduce loan risk
- Improve approval decisions
- Detect risky customers early
    """)

    st.markdown("---")
    st.info("👉 Go to Predictor from sidebar to test the model")

# ---------------- PREDICTOR PAGE ----------------
elif page == "🔮 Predictor":

    st.title("🔮 Credit Risk Prediction")

    # Load model
    try:
        model = pickle.load(open("model.pkl", "rb"))
    except:
        st.error("⚠️ model.pkl not found. Upload it to GitHub.")
        st.stop()

    # Inputs
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 18, 100, 25)
        income = st.number_input("Monthly Income", 0, 100000, 20000)

    with col2:
        debt = st.number_input("Debt Ratio", 0.0, 10.0, 1.0)
        dependents = st.number_input("Dependents", 0, 10, 1)

    st.markdown("---")

    # Prediction
    if st.button("🚀 Predict Risk"):
        data = np.array([[age, income, debt, dependents]])

        try:
            prediction = model.predict(data)

            if prediction[0] == 1:
                st.error("⚠️ High Risk of Default")
            else:
                st.success("✅ Low Risk Customer")

        except Exception as e:
            st.error(f"Error: {e}")
