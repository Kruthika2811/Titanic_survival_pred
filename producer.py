from kafka import KafkaProducer
import json
import time


producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)


passenger = {
    "Pclass": 3,
    "Sex": "male",
    "Age": 22,
    "Fare": 7.25,
    "Embarked": "S"
}


producer.send("titanic-input", passenger)
producer.flush()

print("✅ Message sent to Kafka:", passenger)
