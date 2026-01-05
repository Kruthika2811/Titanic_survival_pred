
# # import streamlit as st
# # import pandas as pd
# # import pickle

# # # ----------------------------------------
# # # Page Configuration
# # # ----------------------------------------
# # st.set_page_config(
# #     page_title="Titanic Survival Prediction",
# #     layout="wide"
# # )

# # st.title("🚢 Titanic Survival Prediction System")
# # st.markdown("Built for **Budhhi – Innovation with Intention**")

# # # ----------------------------------------
# # # Load Trained Model
# # # ----------------------------------------
# # @st.cache_resource
# # def load_model():
# #     with open("notebooks/titanic_model.pkl", "rb") as f:
# #         return pickle.load(f)

# # model = load_model()

# # st.subheader("🔍 Model Feature Debug Info")
# # st.write("Model expects these features:")
# # st.write(model.feature_names_in_)


# # # ----------------------------------------
# # # Load Cleaned Dataset (RAW)
# # # ----------------------------------------
# # @st.cache_data
# # def load_data():
# #     return pd.read_csv("data/cleaned_titanic.csv")

# # df = load_data()

# # # ----------------------------------------
# # # Sidebar – Passenger Input
# # # ----------------------------------------
# # st.sidebar.header("🧍 Passenger Details")

# # pclass = st.sidebar.selectbox("Passenger Class", [1, 2, 3])
# # sex = st.sidebar.selectbox("Gender", ["male", "female"])
# # age = st.sidebar.slider("Age", 1, 80, 30)
# # fare = st.sidebar.number_input("Fare", 0.0, 600.0, 32.0)
# # sibsp = st.sidebar.number_input("Siblings / Spouse", 0, 8, 0)
# # parch = st.sidebar.number_input("Parents / Children", 0, 6, 0)
# # embarked = st.sidebar.selectbox("Embarked Port", ["C", "Q", "S"])

# # # ----------------------------------------
# # # Feature Engineering for Single Prediction
# # # ----------------------------------------
# # family_size = sibsp + parch + 1

# # input_df = pd.DataFrame([{
# #     "Pclass": pclass,
# #     "Age": age,
# #     "Fare": fare,
# #     "FamilySize": family_size,
# #     "Sex_male": 1 if sex == "male" else 0,
# #     "Sex_female": 1 if sex == "female" else 0,
# #     "Embarked_C": 1 if embarked == "C" else 0,
# #     "Embarked_Q": 1 if embarked == "Q" else 0,
# #     "Embarked_S": 1 if embarked == "S" else 0
# # }])

# # # ----------------------------------------
# # # Prediction Section
# # # ----------------------------------------
# # st.subheader("🎯 Survival Prediction")

# # if st.button("Predict Survival"):
# #     prediction = model.predict(input_df)[0]
# #     probability = model.predict_proba(input_df)[0][1]

# #     if prediction == 1:
# #         st.success(f"🎉 Passenger is **LIKELY TO SURVIVE** (Probability: {probability:.2f})")
# #     else:
# #         st.error(f"❌ Passenger is **UNLIKELY TO SURVIVE** (Probability: {probability:.2f})")

# # # ----------------------------------------
# # # Filtering & Sorting Section
# # # ----------------------------------------
# # st.divider()
# # st.subheader("🔍 Filter & Sort Passenger Predictions")

# # col1, col2, col3 = st.columns(3)

# # with col1:
# #     class_filter = st.multiselect("Passenger Class", [1, 2, 3], [1, 2, 3])

# # with col2:
# #     gender_filter = st.multiselect("Gender", ["male", "female"], ["male", "female"])

# # with col3:
# #     age_range = st.slider("Age Range", 0, 80, (0, 80))

# # filtered_df = df[
# #     (df["Pclass"].isin(class_filter)) &
# #     (df["Sex"].isin(gender_filter)) &
# #     (df["Age"].between(age_range[0], age_range[1]))
# # ].copy()

# # # ----------------------------------------
# # # Feature Engineering for Bulk Prediction
# # # ----------------------------------------
# # filtered_df["FamilySize"] = filtered_df["SibSp"] + filtered_df["Parch"] + 1

# # filtered_df["Sex_male"] = (filtered_df["Sex"] == "male").astype(int)
# # filtered_df["Sex_female"] = (filtered_df["Sex"] == "female").astype(int)

# # filtered_df["Embarked_C"] = (filtered_df["Embarked"] == "C").astype(int)
# # filtered_df["Embarked_Q"] = (filtered_df["Embarked"] == "Q").astype(int)
# # filtered_df["Embarked_S"] = (filtered_df["Embarked"] == "S").astype(int)

# # features = [
# #     "Pclass", "Age", "Fare", "FamilySize",
# #     "Sex_male", "Sex_female",
# #     "Embarked_C", "Embarked_Q", "Embarked_S"
# # ]

# # X_filtered = filtered_df[features]

# # filtered_df["Survival_Prediction"] = model.predict(X_filtered)
# # filtered_df["Survival_Probability"] = model.predict_proba(X_filtered)[:, 1]

# # # ----------------------------------------
# # # Sorting
# # # ----------------------------------------
# # sort_option = st.selectbox(
# #     "Sort By",
# #     ["Survival_Probability", "Age", "Fare"]
# # )

# # filtered_df = filtered_df.sort_values(by=sort_option, ascending=False)

# # # ----------------------------------------
# # # Display Output
# # # ----------------------------------------
# # st.dataframe(
# #     filtered_df[[
# #         "PassengerId", "Pclass", "Sex", "Age",
# #         "Fare", "FamilySize",
# #         "Survival_Prediction", "Survival_Probability"
# #     ]],
# #     use_container_width=True
# # )



# import streamlit as st
# import pandas as pd
# import pickle

# # ----------------------------------------
# # Page Config
# # ----------------------------------------
# st.set_page_config(
#     page_title="Titanic Survival Prediction",
#     layout="wide"
# )

# st.title("🚢 Titanic Survival Prediction")
# st.markdown("Built for **Budhhi – Innovation with Intention**")

# # ----------------------------------------
# # Load Model
# # ----------------------------------------
# @st.cache_resource
# def load_model():
#     with open("notebooks/titanic_model.pkl", "rb") as f:
#         return pickle.load(f)

# model = load_model()

# # ----------------------------------------
# # Load Cleaned Dataset
# # ----------------------------------------
# @st.cache_data
# def load_data():
#     return pd.read_csv("data/cleaned_titanic.csv")

# df = load_data()

# # ----------------------------------------
# # Define Model Feature Order (19 features)
# # ----------------------------------------
# # 🔹 REPLACE THIS LIST WITH YOUR TRAINED MODEL FEATURES
# feature_order = [
#     "Pclass_1", "Pclass_2", "Pclass_3",
#     "Sex_female", "Sex_male",
#     "Embarked_C", "Embarked_Q", "Embarked_S",
#     "Age", "Fare", "FamilySize",
#     "SibSp", "Parch",
#     "Title_Mr", "Title_Miss", "Title_Mrs", "Title_Master", "Title_Other"
# ]

# # ----------------------------------------
# # Sidebar – Single Passenger Input
# # ----------------------------------------
# st.sidebar.header("🧍 Passenger Details")

# pclass = st.sidebar.selectbox("Passenger Class", [1, 2, 3])
# sex = st.sidebar.selectbox("Gender", ["male", "female"])
# age = st.sidebar.slider("Age", 1, 80, 30)
# fare = st.sidebar.number_input("Fare", 0.0, 600.0, 32.0)
# sibsp = st.sidebar.number_input("Siblings / Spouse", 0, 8, 0)
# parch = st.sidebar.number_input("Parents / Children", 0, 6, 0)
# embarked = st.sidebar.selectbox("Embarked Port", ["C", "Q", "S"])
# title = st.sidebar.selectbox("Title", ["Mr", "Miss", "Mrs", "Master", "Other"])

# # ----------------------------------------
# # Single Passenger Feature Engineering
# # ----------------------------------------
# family_size = sibsp + parch + 1

# input_data = {
#     "Age": age,
#     "Fare": fare,
#     "SibSp": sibsp,
#     "Parch": parch,
#     "FamilySize": family_size,
#     # One-hot encoding
#     "Pclass_1": 1 if pclass == 1 else 0,
#     "Pclass_2": 1 if pclass == 2 else 0,
#     "Pclass_3": 1 if pclass == 3 else 0,
#     "Sex_male": 1 if sex == "male" else 0,
#     "Sex_female": 1 if sex == "female" else 0,
#     "Embarked_C": 1 if embarked == "C" else 0,
#     "Embarked_Q": 1 if embarked == "Q" else 0,
#     "Embarked_S": 1 if embarked == "S" else 0,
#     "Title_Mr": 1 if title == "Mr" else 0,
#     "Title_Miss": 1 if title == "Miss" else 0,
#     "Title_Mrs": 1 if title == "Mrs" else 0,
#     "Title_Master": 1 if title == "Master" else 0,
#     "Title_Other": 1 if title == "Other" else 0
# }

# input_df = pd.DataFrame([input_data])

# # Ensure all features exist
# for col in feature_order:
#     if col not in input_df.columns:
#         input_df[col] = 0

# input_df = input_df[feature_order]

# # ----------------------------------------
# # Prediction for Single Passenger
# # ----------------------------------------
# st.subheader("🎯 Survival Prediction")
# if st.button("Predict Survival"):
#     prediction = model.predict(input_df)[0]
#     probability = model.predict_proba(input_df)[0][1]

#     if prediction == 1:
#         st.success(f"🎉 Passenger is **LIKELY TO SURVIVE** (Probability: {probability:.2f})")
#     else:
#         st.error(f"❌ Passenger is **UNLIKELY TO SURVIVE** (Probability: {probability:.2f})")

# # ----------------------------------------
# # Filtering & Sorting Section (Bulk Prediction)
# # ----------------------------------------
# st.divider()
# st.subheader("🔍 Filter & Sort Passenger Predictions")

# col1, col2, col3 = st.columns(3)
# with col1:
#     class_filter = st.multiselect("Passenger Class", [1, 2, 3], [1, 2, 3])
# with col2:
#     gender_filter = st.multiselect("Gender", ["male", "female"], ["male", "female"])
# with col3:
#     age_range = st.slider("Age Range", 0, 80, (0, 80))

# filtered_df = df[
#     (df["Pclass"].isin(class_filter)) &
#     (df["Sex"].isin(gender_filter)) &
#     (df["Age"].between(age_range[0], age_range[1]))
# ].copy()

# # Bulk Feature Engineering
# filtered_df["FamilySize"] = filtered_df["SibSp"] + filtered_df["Parch"] + 1

# # One-hot encoding for bulk data
# filtered_df["Pclass_1"] = (filtered_df["Pclass"] == 1).astype(int)
# filtered_df["Pclass_2"] = (filtered_df["Pclass"] == 2).astype(int)
# filtered_df["Pclass_3"] = (filtered_df["Pclass"] == 3).astype(int)

# filtered_df["Sex_male"] = (filtered_df["Sex"] == "male").astype(int)
# filtered_df["Sex_female"] = (filtered_df["Sex"] == "female").astype(int)

# filtered_df["Embarked_C"] = (filtered_df["Embarked"] == "C").astype(int)
# filtered_df["Embarked_Q"] = (filtered_df["Embarked"] == "Q").astype(int)
# filtered_df["Embarked_S"] = (filtered_df["Embarked"] == "S").astype(int)

# # Title column – map unknown titles to Other
# for t in ["Title_Mr", "Title_Miss", "Title_Mrs", "Title_Master", "Title_Other"]:
#     filtered_df[t] = 0

# # Example: Extract Title from Name if available in df
# def map_title(name):
#     if "Mr." in name:
#         return "Title_Mr"
#     elif "Miss." in name:
#         return "Title_Miss"
#     elif "Mrs." in name:
#         return "Title_Mrs"
#     elif "Master." in name:
#         return "Title_Master"
#     else:
#         return "Title_Other"

# if "Name" in filtered_df.columns:
#     filtered_df["Title_col"] = filtered_df["Name"].apply(map_title)
#     for t in ["Title_Mr", "Title_Miss", "Title_Mrs", "Title_Master", "Title_Other"]:
#         filtered_df[t] = (filtered_df["Title_col"] == t).astype(int)

# # Align columns to model
# for col in feature_order:
#     if col not in filtered_df.columns:
#         filtered_df[col] = 0

# X_filtered = filtered_df[feature_order]

# # Predict
# filtered_df["Survival_Prediction"] = model.predict(X_filtered)
# filtered_df["Survival_Probability"] = model.predict_proba(X_filtered)[:, 1]

# # Sorting
# sort_option = st.selectbox("Sort By", ["Survival_Probability", "Age", "Fare"])
# filtered_df = filtered_df.sort_values(by=sort_option, ascending=False)

# # Display
# st.dataframe(
#     filtered_df[[
#         "PassengerId", "Pclass", "Sex", "Age", "Fare", "FamilySize",
#         "Survival_Prediction", "Survival_Probability"
#     ]],
#     use_container_width=True
# )





# ----------------------------------------
# Imports
# ----------------------------------------
import streamlit as st
import pandas as pd
import pickle
import numpy as np

# ----------------------------------------
# Page Configuration
# ----------------------------------------
st.set_page_config(
    page_title="Titanic Survival Prediction",
    layout="wide"
)

st.title("🚢 Titanic Survival Prediction System")
st.markdown("Built for **Budhhi – Innovation with Intention**")

# ----------------------------------------
# Load Model, Scaler, and Feature Columns
# ----------------------------------------
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

# ----------------------------------------
# Sidebar – Single Passenger Input
# ----------------------------------------
st.sidebar.header("🧍 Passenger Details")

pclass = st.sidebar.selectbox("Passenger Class", [1, 2, 3])
sex = st.sidebar.selectbox("Gender", ["male", "female"])
age = st.sidebar.slider("Age", 1, 80, 30)
fare = st.sidebar.number_input("Fare", 0.0, 600.0, 32.0)
sibsp = st.sidebar.number_input("Siblings / Spouse", 0, 8, 0)
parch = st.sidebar.number_input("Parents / Children", 0, 6, 0)
embarked = st.sidebar.selectbox("Embarked Port", ["C", "Q", "S"])
title = st.sidebar.selectbox("Title", ["Mr", "Miss", "Mrs", "Master", "Other"])

# ----------------------------------------
# Feature Engineering for Single Passenger
# ----------------------------------------
family_size = sibsp + parch + 1

input_data = {
    "Age": age,
    "Fare": fare,
    "SibSp": sibsp,
    "Parch": parch,
    "FamilySize": family_size,
    # One-hot encoding
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

# Ensure all features exist in the same order as model training
for col in feature_order:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[feature_order]

# ----------------------------------------
# Scale Features
# ----------------------------------------
input_scaled = scaler.transform(input_df)

# ----------------------------------------
# Predict for Single Passenger
# ----------------------------------------
st.subheader(" Survival Prediction")

if st.button("Predict Survival"):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.success(f" Passenger is **LIKELY TO SURVIVE** (Probability: {probability:.2f})")
    else:
        st.error(f" Passenger is **UNLIKELY TO SURVIVE** (Probability: {probability:.2f})")

# ----------------------------------------
# Bulk Prediction Section (Optional)
# ----------------------------------------
st.divider()
st.subheader("🔍 Bulk Prediction via CSV Upload")

uploaded_file = st.file_uploader("Upload CSV for bulk predictions", type=["csv"])

if uploaded_file:
    bulk_df = pd.read_csv(uploaded_file)
    
    # Feature Engineering for bulk data
    bulk_df["FamilySize"] = bulk_df["SibSp"] + bulk_df["Parch"] + 1
    
    # One-hot encoding
    bulk_df["Pclass_1"] = (bulk_df["Pclass"] == 1).astype(int)
    bulk_df["Pclass_2"] = (bulk_df["Pclass"] == 2).astype(int)
    bulk_df["Pclass_3"] = (bulk_df["Pclass"] == 3).astype(int)
    bulk_df["Sex_male"] = (bulk_df["Sex"] == "male").astype(int)
    bulk_df["Sex_female"] = (bulk_df["Sex"] == "female").astype(int)
    bulk_df["Embarked_C"] = (bulk_df["Embarked"] == "C").astype(int)
    bulk_df["Embarked_Q"] = (bulk_df["Embarked"] == "Q").astype(int)
    bulk_df["Embarked_S"] = (bulk_df["Embarked"] == "S").astype(int)
    
    # Titles
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
    
    # Align features
    for col in feature_order:
        if col not in bulk_df.columns:
            bulk_df[col] = 0
    X_bulk = bulk_df[feature_order]
    
    # Scale and predict
    X_bulk_scaled = scaler.transform(X_bulk)
    bulk_df["Survival_Prediction"] = model.predict(X_bulk_scaled)
    bulk_df["Survival_Probability"] = model.predict_proba(X_bulk_scaled)[:, 1]
    
    st.dataframe(
        bulk_df[["PassengerId", "Pclass", "Sex", "Age", "Fare", "FamilySize",
                 "Survival_Prediction", "Survival_Probability"]],
        use_container_width=True
    )
