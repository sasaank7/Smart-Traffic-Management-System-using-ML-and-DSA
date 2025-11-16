"""
Road network models for graph-based routing algorithms
"""
from sqlalchemy import Column, Integer, Float, String, DateTime, Boolean, JSON, ForeignKey, Index
from sqlalchemy.orm import relationship
from geoalchemy2 import Geometry
from datetime import datetime
from app.core.database import Base


class Intersection(Base):
    """Traffic intersections (graph nodes)"""
    __tablename__ = "intersections"

    id = Column(Integer, primary_key=True, index=True)
    node_id = Column(String(50), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=True)
    location = Column(Geometry('POINT', srid=4326), nullable=False)
    has_traffic_signal = Column(Boolean, default=False)
    signal_id = Column(Integer, ForeignKey("traffic_signals.id"), nullable=True)
    intersection_type = Column(String(50), nullable=True)  # 4-way, T-junction, roundabout
    metadata = Column(JSON, default={})
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    outgoing_segments = relationship("RoadSegment", foreign_keys="RoadSegment.start_intersection_id", back_populates="start_intersection")
    incoming_segments = relationship("RoadSegment", foreign_keys="RoadSegment.end_intersection_id", back_populates="end_intersection")
    traffic_signal = relationship("TrafficSignal", back_populates="intersection", uselist=False)

    __table_args__ = (
        Index('idx_intersection_location', 'location', postgresql_using='gist'),
    )


class RoadSegment(Base):
    """Road segments between intersections (graph edges)"""
    __tablename__ = "road_segments"

    id = Column(Integer, primary_key=True, index=True)
    segment_id = Column(String(50), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=True)
    start_intersection_id = Column(Integer, ForeignKey("intersections.id"), nullable=False)
    end_intersection_id = Column(Integer, ForeignKey("intersections.id"), nullable=False)
    geometry = Column(Geometry('LINESTRING', srid=4326), nullable=False)
    length = Column(Float, nullable=False)  # meters
    speed_limit = Column(Float, nullable=False)  # km/h
    lanes = Column(Integer, default=1)
    road_type = Column(String(50), nullable=False)  # highway, arterial, local
    one_way = Column(Boolean, default=False)
    base_travel_time = Column(Float, nullable=False)  # seconds
    current_congestion_factor = Column(Float, default=1.0)  # 1.0 = no congestion
    metadata = Column(JSON, default={})
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    start_intersection = relationship("Intersection", foreign_keys=[start_intersection_id], back_populates="outgoing_segments")
    end_intersection = relationship("Intersection", foreign_keys=[end_intersection_id], back_populates="incoming_segments")
    sensors = relationship("TrafficSensor", back_populates="road_segment")
    predictions = relationship("TrafficPrediction", back_populates="road_segment")
    anomalies = relationship("TrafficAnomaly", back_populates="road_segment")

    __table_args__ = (
        Index('idx_segment_geometry', 'geometry', postgresql_using='gist'),
        Index('idx_segment_intersections', 'start_intersection_id', 'end_intersection_id'),
    )


class TrafficSignal(Base):
    """Traffic signal control and timing"""
    __tablename__ = "traffic_signals"

    id = Column(Integer, primary_key=True, index=True)
    signal_id = Column(String(50), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=True)
    current_state = Column(String(20), nullable=False)  # red, yellow, green
    cycle_length = Column(Integer, default=90)  # seconds
    green_duration = Column(Integer, default=30)  # seconds
    yellow_duration = Column(Integer, default=5)  # seconds
    red_duration = Column(Integer, default=55)  # seconds
    last_state_change = Column(DateTime, nullable=True)
    adaptive_mode = Column(Boolean, default=True)
    priority_direction = Column(String(50), nullable=True)  # north, south, east, west
    metadata = Column(JSON, default={})
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    intersection = relationship("Intersection", back_populates="traffic_signal", uselist=False)
    schedule = relationship("SignalSchedule", back_populates="signal")


class SignalSchedule(Base):
    """Traffic signal scheduling queue"""
    __tablename__ = "signal_schedules"

    id = Column(Integer, primary_key=True, index=True)
    signal_id = Column(Integer, ForeignKey("traffic_signals.id"), nullable=False)
    scheduled_time = Column(DateTime, nullable=False, index=True)
    direction = Column(String(50), nullable=False)
    duration = Column(Integer, nullable=False)  # seconds
    priority = Column(Integer, default=0)  # higher = more urgent
    reason = Column(String(100), nullable=True)  # emergency, congestion, schedule
    status = Column(String(20), default="pending")  # pending, active, completed
    metadata = Column(JSON, default={})
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    signal = relationship("TrafficSignal", back_populates="schedule")

    __table_args__ = (
        Index('idx_schedule_signal_time', 'signal_id', 'scheduled_time'),
    )


class Route(Base):
    """Computed routes for vehicles"""
    __tablename__ = "routes"

    id = Column(Integer, primary_key=True, index=True)
    route_id = Column(String(50), unique=True, nullable=False, index=True)
    origin = Column(Geometry('POINT', srid=4326), nullable=False)
    destination = Column(Geometry('POINT', srid=4326), nullable=False)
    path = Column(Geometry('LINESTRING', srid=4326), nullable=False)
    segment_ids = Column(JSON, nullable=False)  # List of road segment IDs
    total_distance = Column(Float, nullable=False)  # meters
    estimated_duration = Column(Integer, nullable=False)  # seconds
    algorithm_used = Column(String(50), nullable=False)  # dijkstra, a_star
    is_emergency = Column(Boolean, default=False)
    congestion_aware = Column(Boolean, default=True)
    metadata = Column(JSON, default={})
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)

    __table_args__ = (
        Index('idx_route_path', 'path', postgresql_using='gist'),
        Index('idx_route_created', 'created_at'),
    )
