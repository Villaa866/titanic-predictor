import streamlit as st
import numpy as np
import pickle

# App config
st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")

# Sidebar
with st.sidebar:
    st.title("About")
    st.markdown("""
    **Titanic Survival Predictor**  
    Built with Machine Learning  
    Deployed via Streamlit  
    Trained on real Titanic dataset  
    """)
    st.markdown("**Creator:** Eben Villa")
    st.markdown("[View on GitHub](https://github.com/Villaa866/titanic-predictor)")

# Main header
st.title("🚢 Titanic Survival Predictor")
st.markdown("Enter passenger details below to predict their survival on the Titanic.")

# Load model
@st.cache_resource
def load_model():
    with open("titanic_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# Input layout
col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Passenger Class", [1, 2, 3])
    sex = st.radio("Sex", ["Male", "Female"])
    age = st.slider("Age", 0, 100, 25)

with col2:
    sibsp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)
    parch = st.number_input("Parents/Children Aboard", 0, 10, 0)
    fare = st.number_input("Fare", 0.0, 600.0, 50.0)
    embarked = st.selectbox("Embarked", ["Southampton", "Cherbourg", "Queenstown"])

# Encode inputs
sex = 0 if sex == "Male" else 1
embarked = {"Southampton": 0, "Cherbourg": 1, "Queenstown": 2}[embarked]

input_data = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])

# Predict
if st.button("Predict Survival"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]  # Survival probability

    if prediction == 1:
        st.success(f"✅ This passenger would have survived! (Probability: {probability:.2%})")
    else:
        st.error(f"❌ Unfortunately, this passenger would not survive. (Survival Probability: {probability:.2%})")
