"""
Kafka Producer for Traffic Data Streaming

Sends real-time traffic sensor data to Kafka topics
"""
import json
import time
from datetime import datetime
from kafka import KafkaProducer
from kafka.errors import KafkaError
import random


class TrafficDataProducer:
    """
    Kafka producer for traffic data streaming
    """

    def __init__(self, bootstrap_servers='localhost:9092'):
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8') if k else None,
        )
        self.topic_traffic = 'traffic-data'
        self.topic_anomalies = 'traffic-anomalies'
        self.topic_predictions = 'traffic-predictions'

    def send_traffic_reading(self, sensor_id, reading_data):
        """
        Send traffic sensor reading to Kafka

        Args:
            sensor_id: Sensor identifier
            reading_data: Dictionary with traffic metrics
        """
        message = {
            'sensor_id': sensor_id,
            'timestamp': datetime.utcnow().isoformat(),
            **reading_data
        }

        try:
            future = self.producer.send(
                self.topic_traffic,
                key=str(sensor_id),
                value=message
            )
            record_metadata = future.get(timeout=10)
            print(f"Sent to {record_metadata.topic} partition {record_metadata.partition}")
        except KafkaError as e:
            print(f"Failed to send message: {e}")

    def send_anomaly_detection(self, anomaly_data):
        """Send anomaly detection event"""
        message = {
            'timestamp': datetime.utcnow().isoformat(),
            **anomaly_data
        }

        try:
            self.producer.send(self.topic_anomalies, value=message)
            print(f"Anomaly sent: {anomaly_data['type']}")
        except KafkaError as e:
            print(f"Failed to send anomaly: {e}")

    def send_prediction(self, prediction_data):
        """Send traffic prediction"""
        message = {
            'timestamp': datetime.utcnow().isoformat(),
            **prediction_data
        }

        try:
            self.producer.send(self.topic_predictions, value=message)
        except KafkaError as e:
            print(f"Failed to send prediction: {e}")

    def simulate_traffic_data(self, num_sensors=10, duration_minutes=60):
        """
        Simulate traffic data from multiple sensors

        Args:
            num_sensors: Number of sensors to simulate
            duration_minutes: How long to run simulation
        """
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)

        print(f"Simulating traffic data from {num_sensors} sensors for {duration_minutes} minutes...")

        while time.time() < end_time:
            for sensor_id in range(1, num_sensors + 1):
                # Generate random traffic data
                reading = {
                    'vehicle_count': random.randint(5, 50),
                    'average_speed': random.uniform(20, 80),
                    'occupancy': random.uniform(0.1, 0.9),
                    'density': random.uniform(10, 100),
                }

                self.send_traffic_reading(f"sensor_{sensor_id}", reading)

                # Randomly send anomaly
                if random.random() < 0.05:  # 5% chance
                    anomaly = {
                        'sensor_id': f"sensor_{sensor_id}",
                        'type': random.choice(['congestion', 'accident', 'road_closure']),
                        'severity': random.choice(['low', 'medium', 'high']),
                        'confidence': random.uniform(0.7, 0.99),
                    }
                    self.send_anomaly_detection(anomaly)

            # Sleep for 5 seconds between batches
            time.sleep(5)
            print(f"Sent batch at {datetime.utcnow()}")

        print("Simulation complete")

    def close(self):
        """Close producer"""
        self.producer.close()


if __name__ == "__main__":
    # Run simulation
    producer = TrafficDataProducer(bootstrap_servers='localhost:9092')

    try:
        producer.simulate_traffic_data(num_sensors=10, duration_minutes=60)
    except KeyboardInterrupt:
        print("\nStopping producer...")
    finally:
        producer.close()
