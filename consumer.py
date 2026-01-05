


from kafka import KafkaConsumer
import json
import pickle
import pandas as pd


with open("notebooks/titanic_model.pkl", "rb") as f:
    model = pickle.load(f)

print("✅ Model loaded successfully")
print("ℹ️ Model expects", model.n_features_in_, "features")


consumer = KafkaConsumer(
    'titanic-input',
    bootstrap_servers='localhost:9092',
    auto_offset_reset='earliest',
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

print("🚀 Kafka Consumer started... Waiting for passenger data")


ALL_FEATURES = [
    'Pclass', 'Age', 'Fare',
    'Sex_female', 'Sex_male',
    'Embarked_C', 'Embarked_Q', 'Embarked_S',
    'SibSp', 'Parch', 'FamilySize',
    'Title_Mr', 'Title_Miss', 'Title_Mrs',
    'Title_Master', 'Title_Dr', 'Title_Rev',
    'Title_Col', 'Title_Other'
]


for message in consumer:
    data = message.value
    print("\n📥 Raw Kafka Message:", data)

   
    processed = dict.fromkeys(ALL_FEATURES, 0)

   
    processed['Pclass'] = data.get('Pclass', 3)
    processed['Age'] = data.get('Age', 30)
    processed['Fare'] = data.get('Fare', 10.0)

    
    if data.get('Sex') == 'male':
        processed['Sex_male'] = 1
        processed['Title_Mr'] = 1
    else:
        processed['Sex_female'] = 1
        processed['Title_Miss'] = 1

    
    emb = data.get('Embarked', 'S')
    if emb == 'S':
        processed['Embarked_S'] = 1
    elif emb == 'C':
        processed['Embarked_C'] = 1
    elif emb == 'Q':
        processed['Embarked_Q'] = 1

    processed['SibSp'] = 0
    processed['Parch'] = 0
    processed['FamilySize'] = 1


    df = pd.DataFrame([processed])[ALL_FEATURES]


    prediction = model.predict(df)[0]

    result = "🟢 Survived" if prediction == 1 else "🔴 Not Survived"
    print("🎯 Prediction:", result)
    print("-" * 60)
