"""
Traffic data models for storing sensor readings, predictions, and anomalies
"""
from sqlalchemy import Column, Integer, Float, String, DateTime, Boolean, JSON, ForeignKey, Index
from sqlalchemy.orm import relationship
from geoalchemy2 import Geometry
from datetime import datetime
from app.core.database import Base


class TrafficSensor(Base):
    """Traffic sensor locations and metadata"""
    __tablename__ = "traffic_sensors"

    id = Column(Integer, primary_key=True, index=True)
    sensor_id = Column(String(50), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    sensor_type = Column(String(50), nullable=False)  # CCTV, GPS, Induction Loop
    location = Column(Geometry('POINT', srid=4326), nullable=False)
    road_segment_id = Column(Integer, ForeignKey("road_segments.id"), nullable=True)
    status = Column(String(20), default="active")  # active, inactive, maintenance
    metadata = Column(JSON, default={})
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    readings = relationship("TrafficReading", back_populates="sensor")
    road_segment = relationship("RoadSegment", back_populates="sensors")

    __table_args__ = (
        Index('idx_sensor_location', 'location', postgresql_using='gist'),
    )


class TrafficReading(Base):
    """Real-time traffic sensor readings"""
    __tablename__ = "traffic_readings"

    id = Column(Integer, primary_key=True, index=True)
    sensor_id = Column(Integer, ForeignKey("traffic_sensors.id"), nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    vehicle_count = Column(Integer, nullable=False)
    average_speed = Column(Float, nullable=True)  # km/h
    occupancy = Column(Float, nullable=True)  # percentage
    density = Column(Float, nullable=True)  # vehicles per km
    flow_rate = Column(Float, nullable=True)  # vehicles per hour
    weather_condition = Column(String(50), nullable=True)
    temperature = Column(Float, nullable=True)
    metadata = Column(JSON, default={})
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    sensor = relationship("TrafficSensor", back_populates="readings")

    __table_args__ = (
        Index('idx_reading_sensor_timestamp', 'sensor_id', 'timestamp'),
    )


class TrafficPrediction(Base):
    """ML-generated traffic predictions"""
    __tablename__ = "traffic_predictions"

    id = Column(Integer, primary_key=True, index=True)
    road_segment_id = Column(Integer, ForeignKey("road_segments.id"), nullable=False)
    prediction_time = Column(DateTime, nullable=False)  # When prediction was made
    target_time = Column(DateTime, nullable=False, index=True)  # Future time being predicted
    predicted_density = Column(Float, nullable=False)  # vehicles per km
    predicted_speed = Column(Float, nullable=False)  # km/h
    congestion_level = Column(String(20), nullable=False)  # low, medium, high, severe
    confidence_score = Column(Float, nullable=False)  # 0.0 to 1.0
    model_version = Column(String(50), nullable=False)
    metadata = Column(JSON, default={})
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    road_segment = relationship("RoadSegment", back_populates="predictions")

    __table_args__ = (
        Index('idx_prediction_segment_target', 'road_segment_id', 'target_time'),
    )


class TrafficAnomaly(Base):
    """Detected traffic anomalies and incidents"""
    __tablename__ = "traffic_anomalies"

    id = Column(Integer, primary_key=True, index=True)
    anomaly_type = Column(String(50), nullable=False)  # accident, congestion, road_closure
    severity = Column(String(20), nullable=False)  # low, medium, high, critical
    location = Column(Geometry('POINT', srid=4326), nullable=False)
    road_segment_id = Column(Integer, ForeignKey("road_segments.id"), nullable=True)
    detected_at = Column(DateTime, nullable=False, index=True)
    resolved_at = Column(DateTime, nullable=True)
    status = Column(String(20), default="active")  # active, resolved, false_positive
    description = Column(String(500), nullable=True)
    confidence_score = Column(Float, nullable=False)
    source = Column(String(50), nullable=False)  # ML_model, manual, sensor
    metadata = Column(JSON, default={})
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    road_segment = relationship("RoadSegment", back_populates="anomalies")

    __table_args__ = (
        Index('idx_anomaly_location', 'location', postgresql_using='gist'),
        Index('idx_anomaly_detected', 'detected_at'),
    )


class VehicleDetection(Base):
    """Vehicle detections from CCTV/ML models"""
    __tablename__ = "vehicle_detections"

    id = Column(Integer, primary_key=True, index=True)
    sensor_id = Column(Integer, ForeignKey("traffic_sensors.id"), nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    vehicle_type = Column(String(50), nullable=False)  # car, truck, bus, motorcycle, bicycle
    confidence = Column(Float, nullable=False)
    bounding_box = Column(JSON, nullable=True)  # {x, y, width, height}
    speed = Column(Float, nullable=True)  # km/h
    direction = Column(Float, nullable=True)  # degrees
    metadata = Column(JSON, default={})
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    sensor = relationship("TrafficSensor")

    __table_args__ = (
        Index('idx_detection_sensor_timestamp', 'sensor_id', 'timestamp'),
    )


class EmergencyVehicle(Base):
    """Emergency vehicle tracking and routing"""
    __tablename__ = "emergency_vehicles"

    id = Column(Integer, primary_key=True, index=True)
    vehicle_id = Column(String(50), unique=True, nullable=False, index=True)
    vehicle_type = Column(String(50), nullable=False)  # ambulance, fire_truck, police
    status = Column(String(20), nullable=False)  # idle, en_route, arrived
    current_location = Column(Geometry('POINT', srid=4326), nullable=True)
    destination = Column(Geometry('POINT', srid=4326), nullable=True)
    priority = Column(Integer, default=5)  # 1-10, 10 being highest
    route_path = Column(Geometry('LINESTRING', srid=4326), nullable=True)
    estimated_arrival = Column(DateTime, nullable=True)
    activated_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    metadata = Column(JSON, default={})
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index('idx_emergency_location', 'current_location', postgresql_using='gist'),
        Index('idx_emergency_status', 'status'),
    )
