import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

# App config
st.set_page_config(page_title="Titanic Survival App", page_icon="🚢", layout="wide")

# Sidebar
with st.sidebar:
    st.title("About the App")
    st.markdown("""
    **Titanic Survival Predictor**  
    Built with Machine Learning  
    Trained on the original Titanic dataset  
    Deployed with Streamlit  
    """)
    st.markdown("**Created by:** Eben Villa")
    st.markdown("[GitHub Repo](https://github.com/Villaa866/titanic-predictor)")

    st.markdown("---")
    st.info("Predict if a Titanic passenger would have survived based on your input.")

# Load model
@st.cache_resource
def load_model():
    with open("titanic_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("train.csv")

model = load_model()
df = load_data()

# Tabs
tab1, tab2, tab3 = st.tabs(["🧠 Prediction", "📊 Insights", "📁 Data Preview"])

# ---------------- Prediction Tab ----------------
with tab1:
    st.header("🧠 Predict Titanic Passenger Survival")

    col1, col2 = st.columns(2)

    with col1:
        pclass = st.selectbox("Passenger Class", [1, 2, 3])
        sex = st.radio("Sex", ["Male", "Female"])
        age = st.slider("Age", 0, 100, 25)

    with col2:
        sibsp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)
        parch = st.number_input("Parents/Children Aboard", 0, 10, 0)
        fare = st.number_input("Fare", 0.0, 600.0, 50.0)
        embarked = st.selectbox("Port of Embarkation", ["Southampton", "Cherbourg", "Queenstown"])

    # Encode inputs
    sex = 0 if sex == "Male" else 1
    embarked = {"Southampton": 0, "Cherbourg": 1, "Queenstown": 2}[embarked]
    input_data = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])

    if st.button("Predict Survival"):
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        st.subheader("Prediction Result")
        if prediction == 1:
            st.success(f"✅ This passenger would have survived! (Probability: {probability:.2%})")
        else:
            st.error(f"❌ Unfortunately, this passenger would not have survived. (Survival Probability: {probability:.2%})")

# ---------------- Insights Tab ----------------
with tab2:
    st.header("📊 Data Insights & Survival Patterns")

    st.subheader("Survival Rate by Sex")
    fig_sex = px.bar(df, x='Sex', color='Survived',
                     barmode='group', title="Survival by Gender")
    st.plotly_chart(fig_sex, use_container_width=True)

    st.subheader("Survival Rate by Passenger Class")
    fig_class = px.histogram(df, x='Pclass', color='Survived',
                             barmode='group', title="Survival by Class")
    st.plotly_chart(fig_class, use_container_width=True)

    st.subheader("Survival Rate by Embarkation Port")
    fig_embark = px.histogram(df, x='Embarked', color='Survived',
                              barmode='group', title="Survival by Embarkation")
    st.plotly_chart(fig_embark, use_container_width=True)

    st.subheader("Survival Distribution")
    fig_pie = px.pie(df, names='Survived', title='Overall Survival Distribution',
                     color_discrete_sequence=px.colors.sequential.RdBu)
    fig_pie.update_traces(textinfo='percent+label')
    st.plotly_chart(fig_pie, use_container_width=True)

# ---------------- Data Preview Tab ----------------
with tab3:
    st.header("📁 Explore the Titanic Dataset")

    st.markdown("Use the filters below to explore the dataset.")

    selected_class = st.multiselect("Filter by Class", options=[1, 2, 3], default=[1, 2, 3])
    selected_gender = st.multiselect("Filter by Sex", options=df["Sex"].unique(), default=df["Sex"].unique())

    filtered_df = df[(df["Pclass"].isin(selected_class)) & (df["Sex"].isin(selected_gender))]

    st.dataframe(filtered_df.head(50), use_container_width=True)

    with st.expander("Show Summary Stats"):
        st.write(filtered_df.describe())

    with st.expander("Missing Values Overview"):
        st.write(filtered_df.isnull().sum())

# ---------------- Footer ----------------
st.markdown("---")
st.markdown("<center>Made with ❤️ by Eben Villa | Powered by Streamlit</center>", unsafe_allow_html=True)
