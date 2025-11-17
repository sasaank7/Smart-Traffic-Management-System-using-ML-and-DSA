"""
Database models package
"""
from app.models.traffic import (
    TrafficSensor,
    TrafficReading,
    TrafficPrediction,
    TrafficAnomaly,
    VehicleDetection,
    EmergencyVehicle,
)
from app.models.road_network import (
    Intersection,
    RoadSegment,
    TrafficSignal,
    SignalSchedule,
    Route,
)
from app.models.user import User, RefreshToken, UserRole

__all__ = [
    # Traffic models
    "TrafficSensor",
    "TrafficReading",
    "TrafficPrediction",
    "TrafficAnomaly",
    "VehicleDetection",
    "EmergencyVehicle",
    # Road network models
    "Intersection",
    "RoadSegment",
    "TrafficSignal",
    "SignalSchedule",
    "Route",
    # User models
    "User",
    "RefreshToken",
    "UserRole",
]
