# # from kafka import KafkaConsumer
# # import json
# # import pickle
# # import pandas as pd

# # # Load trained ML model
# # with open("notebooks/titanic_model.pkl", "rb") as f:
# #     model = pickle.load(f)

# # # Kafka Consumer
# # consumer = KafkaConsumer(
# #     'titanic-input',
# #     bootstrap_servers='localhost:9092',
# #     auto_offset_reset='earliest',
# #     value_deserializer=lambda x: json.loads(x.decode('utf-8'))
# # )

# # print("🚀 Consumer started... Waiting for messages")

# # for message in consumer:
# #     data = message.value
# #     print("📥 Received:", data)

# #     # Convert to DataFrame
# #     df = pd.DataFrame([data])

# #     # Prediction
# #     prediction = model.predict(df)[0]

# #     print("🎯 Survival Prediction:", "Survived" if prediction == 1 else "Not Survived")
# #     print("-" * 50)


# from kafka import KafkaConsumer
# import json
# import pickle
# import pandas as pd

# # Load trained model
# with open("notebooks/titanic_model.pkl", "rb") as f:
#     model = pickle.load(f)

# consumer = KafkaConsumer(
#     'titanic-input',
#     bootstrap_servers='localhost:9092',
#     auto_offset_reset='earliest',
#     value_deserializer=lambda x: json.loads(x.decode('utf-8'))
# )

# print("🚀 Consumer started... Waiting for messages")

# for message in consumer:
#     data = message.value
#     print("📥 Received:", data)

#     # ---------- MANUAL PREPROCESSING ----------
#     processed_data = {
#         "Pclass": data["Pclass"],
#         "Age": data["Age"],
#         "Fare": data["Fare"],
#         "Sex_male": 1 if data["Sex"] == "male" else 0,
#         "Sex_female": 1 if data["Sex"] == "female" else 0,
#         "Embarked_S": 1 if data["Embarked"] == "S" else 0,
#         "Embarked_C": 1 if data["Embarked"] == "C" else 0,
#         "Embarked_Q": 1 if data["Embarked"] == "Q" else 0,
#     }

#     df = pd.DataFrame([processed_data])

#     prediction = model.predict(df)[0]

#     print("🎯 Prediction:", "Survived" if prediction == 1 else "Not Survived")
#     print("-" * 50)


from kafka import KafkaConsumer
import json
import pickle
import pandas as pd

# --------------------------------------------------
# Load trained Titanic model
# --------------------------------------------------
with open("notebooks/titanic_model.pkl", "rb") as f:
    model = pickle.load(f)

print("✅ Model loaded successfully")
print("ℹ️ Model expects", model.n_features_in_, "features")

# --------------------------------------------------
# Kafka Consumer Configuration
# --------------------------------------------------
consumer = KafkaConsumer(
    'titanic-input',
    bootstrap_servers='localhost:9092',
    auto_offset_reset='earliest',
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

print("🚀 Kafka Consumer started... Waiting for passenger data")

# --------------------------------------------------
# Expected feature list (19 features)
# MUST match training-time features
# --------------------------------------------------
ALL_FEATURES = [
    'Pclass', 'Age', 'Fare',
    'Sex_female', 'Sex_male',
    'Embarked_C', 'Embarked_Q', 'Embarked_S',
    'SibSp', 'Parch', 'FamilySize',
    'Title_Mr', 'Title_Miss', 'Title_Mrs',
    'Title_Master', 'Title_Dr', 'Title_Rev',
    'Title_Col', 'Title_Other'
]

# --------------------------------------------------
# Consume messages
# --------------------------------------------------
for message in consumer:
    data = message.value
    print("\n📥 Raw Kafka Message:", data)

    # Initialize all features with 0
    processed = dict.fromkeys(ALL_FEATURES, 0)

    # Fill numeric features
    processed['Pclass'] = data.get('Pclass', 3)
    processed['Age'] = data.get('Age', 30)
    processed['Fare'] = data.get('Fare', 10.0)

    # Encode Sex
    if data.get('Sex') == 'male':
        processed['Sex_male'] = 1
        processed['Title_Mr'] = 1
    else:
        processed['Sex_female'] = 1
        processed['Title_Miss'] = 1

    # Encode Embarked
    emb = data.get('Embarked', 'S')
    if emb == 'S':
        processed['Embarked_S'] = 1
    elif emb == 'C':
        processed['Embarked_C'] = 1
    elif emb == 'Q':
        processed['Embarked_Q'] = 1

    # Default family features (not streamed yet)
    processed['SibSp'] = 0
    processed['Parch'] = 0
    processed['FamilySize'] = 1

    # Create DataFrame in correct order
    df = pd.DataFrame([processed])[ALL_FEATURES]

    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------
    prediction = model.predict(df)[0]

    result = "🟢 Survived" if prediction == 1 else "🔴 Not Survived"
    print("🎯 Prediction:", result)
    print("-" * 60)
