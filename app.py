

import streamlit as st
import pandas as pd
import pickle
import numpy as np


st.set_page_config(
    page_title="Titanic Survival Prediction",
    layout="wide"
)

st.title("🚢 Titanic Survival Prediction System")
st.markdown("Built for **Budhhi – Innovation with Intention**")


@st.cache_resource
def load_model_scaler_features():
    with open("notebooks/rf_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("notebooks/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("notebooks/feature_columns.pkl", "rb") as f:
        feature_order = pickle.load(f)
    return model, scaler, feature_order

model, scaler, feature_order = load_model_scaler_features()


st.sidebar.header("🧍 Passenger Details")

pclass = st.sidebar.selectbox("Passenger Class", [1, 2, 3])
sex = st.sidebar.selectbox("Gender", ["male", "female"])
age = st.sidebar.slider("Age", 1, 80, 30)
fare = st.sidebar.number_input("Fare", 0.0, 600.0, 32.0)
sibsp = st.sidebar.number_input("Siblings / Spouse", 0, 8, 0)
parch = st.sidebar.number_input("Parents / Children", 0, 6, 0)
embarked = st.sidebar.selectbox("Embarked Port", ["C", "Q", "S"])
title = st.sidebar.selectbox("Title", ["Mr", "Miss", "Mrs", "Master", "Other"])


family_size = sibsp + parch + 1

input_data = {
    "Age": age,
    "Fare": fare,
    "SibSp": sibsp,
    "Parch": parch,
    "FamilySize": family_size,
    
    "Pclass_1": 1 if pclass == 1 else 0,
    "Pclass_2": 1 if pclass == 2 else 0,
    "Pclass_3": 1 if pclass == 3 else 0,
    "Sex_male": 1 if sex == "male" else 0,
    "Sex_female": 1 if sex == "female" else 0,
    "Embarked_C": 1 if embarked == "C" else 0,
    "Embarked_Q": 1 if embarked == "Q" else 0,
    "Embarked_S": 1 if embarked == "S" else 0,
    "Title_Mr": 1 if title == "Mr" else 0,
    "Title_Miss": 1 if title == "Miss" else 0,
    "Title_Mrs": 1 if title == "Mrs" else 0,
    "Title_Master": 1 if title == "Master" else 0,
    "Title_Other": 1 if title == "Other" else 0
}

input_df = pd.DataFrame([input_data])


for col in feature_order:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[feature_order]


input_scaled = scaler.transform(input_df)


st.subheader(" Survival Prediction")

if st.button("Predict Survival"):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.success(f" Passenger is **LIKELY TO SURVIVE** (Probability: {probability:.2f})")
    else:
        st.error(f" Passenger is **UNLIKELY TO SURVIVE** (Probability: {probability:.2f})")


st.divider()
st.subheader("🔍 Bulk Prediction via CSV Upload")

uploaded_file = st.file_uploader("Upload CSV for bulk predictions", type=["csv"])

if uploaded_file:
    bulk_df = pd.read_csv(uploaded_file)
    
    
    bulk_df["FamilySize"] = bulk_df["SibSp"] + bulk_df["Parch"] + 1
    
   
    bulk_df["Pclass_1"] = (bulk_df["Pclass"] == 1).astype(int)
    bulk_df["Pclass_2"] = (bulk_df["Pclass"] == 2).astype(int)
    bulk_df["Pclass_3"] = (bulk_df["Pclass"] == 3).astype(int)
    bulk_df["Sex_male"] = (bulk_df["Sex"] == "male").astype(int)
    bulk_df["Sex_female"] = (bulk_df["Sex"] == "female").astype(int)
    bulk_df["Embarked_C"] = (bulk_df["Embarked"] == "C").astype(int)
    bulk_df["Embarked_Q"] = (bulk_df["Embarked"] == "Q").astype(int)
    bulk_df["Embarked_S"] = (bulk_df["Embarked"] == "S").astype(int)
    
  
    for t in ["Title_Mr", "Title_Miss", "Title_Mrs", "Title_Master", "Title_Other"]:
        bulk_df[t] = 0
    
    def map_title(name):
        if "Mr." in name:
            return "Title_Mr"
        elif "Miss." in name:
            return "Title_Miss"
        elif "Mrs." in name:
            return "Title_Mrs"
        elif "Master." in name:
            return "Title_Master"
        else:
            return "Title_Other"
    
    if "Name" in bulk_df.columns:
        bulk_df["Title_col"] = bulk_df["Name"].apply(map_title)
        for t in ["Title_Mr", "Title_Miss", "Title_Mrs", "Title_Master", "Title_Other"]:
            bulk_df[t] = (bulk_df["Title_col"] == t).astype(int)
    

    for col in feature_order:
        if col not in bulk_df.columns:
            bulk_df[col] = 0
    X_bulk = bulk_df[feature_order]
    
    
    X_bulk_scaled = scaler.transform(X_bulk)
    bulk_df["Survival_Prediction"] = model.predict(X_bulk_scaled)
    bulk_df["Survival_Probability"] = model.predict_proba(X_bulk_scaled)[:, 1]
    
    st.dataframe(
        bulk_df[["PassengerId", "Pclass", "Sex", "Age", "Fare", "FamilySize",
                 "Survival_Prediction", "Survival_Probability"]],
        use_container_width=True
    )
