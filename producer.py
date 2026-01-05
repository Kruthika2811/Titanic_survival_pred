from kafka import KafkaProducer
import json
import time

# Connect to Kafka
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Sample Titanic passenger data
passenger = {
    "Pclass": 3,
    "Sex": "male",
    "Age": 22,
    "Fare": 7.25,
    "Embarked": "S"
}

# Send message
producer.send("titanic-input", passenger)
producer.flush()

print("✅ Message sent to Kafka:", passenger)
