import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

@st.cache_data
def load_model():
    df = sns.load_dataset("titanic").dropna(subset=["age", "sex", "fare", "embarked", "pclass", "sibsp", "parch"])
    df = df[["survived", "pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]]
    df["sex"] = df["sex"].map({"male": 0, "female": 1})
    df["embarked"] = df["embarked"].map({"S": 0, "C": 1, "Q": 2})
    
    X = df.drop("survived", axis=1)
    y = df["survived"]
    
    model = RandomForestClassifier()
    model.fit(X, y)
    return model

model = load_model()

st.title("Titanic Survival Predictor")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.radio("Sex", ["Male", "Female"])
age = st.slider("Age", 0, 100, 25)
sibsp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)
parch = st.number_input("Parents/Children Aboard", 0, 10, 0)
fare = st.number_input("Fare", 0.0, 600.0, 50.0)
embarked = st.selectbox("Embarked", ["Southampton", "Cherbourg", "Queenstown"])

sex = 0 if sex == "Male" else 1
embarked = {"Southampton": 0, "Cherbourg": 1, "Queenstown": 2}[embarked]

input_data = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])

if st.button("Predict Survival"):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.success("This passenger would survive!")
    else:
        st.error("This passenger would not survive.")
