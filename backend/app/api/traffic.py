"""
Traffic data and prediction API endpoints
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import List, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel

from app.core.database import get_db
from app.models.traffic import TrafficReading, TrafficPrediction, TrafficAnomaly, TrafficSensor

router = APIRouter()


# Pydantic models
class TrafficReadingCreate(BaseModel):
    sensor_id: int
    vehicle_count: int
    average_speed: Optional[float] = None
    occupancy: Optional[float] = None
    density: Optional[float] = None
    weather_condition: Optional[str] = None
    temperature: Optional[float] = None


class TrafficReadingResponse(BaseModel):
    id: int
    sensor_id: int
    timestamp: datetime
    vehicle_count: int
    average_speed: Optional[float]
    density: Optional[float]

    class Config:
        from_attributes = True


class PredictionRequest(BaseModel):
    road_segment_id: int
    target_time: datetime


class PredictionResponse(BaseModel):
    road_segment_id: int
    predicted_density: float
    predicted_speed: float
    congestion_level: str
    confidence_score: float
    target_time: datetime


@router.post("/readings", response_model=TrafficReadingResponse)
async def create_traffic_reading(
    reading: TrafficReadingCreate,
    db: AsyncSession = Depends(get_db),
):
    """
    Create new traffic sensor reading

    This endpoint receives real-time traffic data from sensors
    """
    # Create reading
    db_reading = TrafficReading(
        sensor_id=reading.sensor_id,
        timestamp=datetime.utcnow(),
        vehicle_count=reading.vehicle_count,
        average_speed=reading.average_speed,
        occupancy=reading.occupancy,
        density=reading.density,
        weather_condition=reading.weather_condition,
        temperature=reading.temperature,
    )

    db.add(db_reading)
    await db.commit()
    await db.refresh(db_reading)

    return db_reading


@router.get("/readings", response_model=List[TrafficReadingResponse])
async def get_traffic_readings(
    sensor_id: Optional[int] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: int = Query(100, le=1000),
    db: AsyncSession = Depends(get_db),
):
    """
    Get traffic readings with optional filters

    Args:
        sensor_id: Filter by sensor
        start_time: Start of time range
        end_time: End of time range
        limit: Maximum number of records
    """
    query = select(TrafficReading)

    # Apply filters
    if sensor_id:
        query = query.where(TrafficReading.sensor_id == sensor_id)

    if start_time:
        query = query.where(TrafficReading.timestamp >= start_time)

    if end_time:
        query = query.where(TrafficReading.timestamp <= end_time)

    # Order and limit
    query = query.order_by(TrafficReading.timestamp.desc()).limit(limit)

    result = await db.execute(query)
    readings = result.scalars().all()

    return readings


@router.get("/predictions/{road_segment_id}", response_model=List[PredictionResponse])
async def get_predictions(
    road_segment_id: int,
    hours_ahead: int = Query(1, ge=1, le=24),
    db: AsyncSession = Depends(get_db),
):
    """
    Get traffic predictions for a road segment

    Args:
        road_segment_id: Road segment ID
        hours_ahead: Number of hours to predict ahead
    """
    # Get latest predictions
    target_time = datetime.utcnow() + timedelta(hours=hours_ahead)

    query = select(TrafficPrediction).where(
        TrafficPrediction.road_segment_id == road_segment_id,
        TrafficPrediction.target_time >= datetime.utcnow(),
        TrafficPrediction.target_time <= target_time,
    ).order_by(TrafficPrediction.target_time)

    result = await db.execute(query)
    predictions = result.scalars().all()

    return predictions


@router.post("/predict")
async def request_prediction(
    request: PredictionRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Request traffic prediction for specific road segment and time

    This triggers the ML service to generate a prediction
    """
    # TODO: Call ML service for prediction
    # For now, return mock prediction

    prediction = TrafficPrediction(
        road_segment_id=request.road_segment_id,
        prediction_time=datetime.utcnow(),
        target_time=request.target_time,
        predicted_density=35.5,
        predicted_speed=45.0,
        congestion_level="medium",
        confidence_score=0.87,
        model_version="v1.0",
    )

    db.add(prediction)
    await db.commit()
    await db.refresh(prediction)

    return {
        "success": True,
        "prediction": {
            "road_segment_id": prediction.road_segment_id,
            "predicted_density": prediction.predicted_density,
            "predicted_speed": prediction.predicted_speed,
            "congestion_level": prediction.congestion_level,
            "confidence_score": prediction.confidence_score,
            "target_time": prediction.target_time.isoformat(),
        },
    }


@router.get("/anomalies")
async def get_anomalies(
    status: str = Query("active"),
    limit: int = Query(50, le=500),
    db: AsyncSession = Depends(get_db),
):
    """
    Get traffic anomalies

    Args:
        status: Filter by status (active, resolved, false_positive)
        limit: Maximum number of records
    """
    query = select(TrafficAnomaly).where(
        TrafficAnomaly.status == status
    ).order_by(TrafficAnomaly.detected_at.desc()).limit(limit)

    result = await db.execute(query)
    anomalies = result.scalars().all()

    return {
        "count": len(anomalies),
        "anomalies": [
            {
                "id": a.id,
                "type": a.anomaly_type,
                "severity": a.severity,
                "detected_at": a.detected_at.isoformat(),
                "description": a.description,
                "confidence_score": a.confidence_score,
            }
            for a in anomalies
        ],
    }


@router.get("/sensors")
async def get_sensors(
    status: str = Query("active"),
    db: AsyncSession = Depends(get_db),
):
    """Get all traffic sensors"""
    query = select(TrafficSensor).where(TrafficSensor.status == status)

    result = await db.execute(query)
    sensors = result.scalars().all()

    return {
        "count": len(sensors),
        "sensors": [
            {
                "id": s.id,
                "sensor_id": s.sensor_id,
                "name": s.name,
                "type": s.sensor_type,
                "status": s.status,
            }
            for s in sensors
        ],
    }


@router.get("/stats/current")
async def get_current_stats(db: AsyncSession = Depends(get_db)):
    """
    Get current traffic statistics

    Returns aggregate statistics across all sensors
    """
    # Get readings from last 5 minutes
    five_min_ago = datetime.utcnow() - timedelta(minutes=5)

    query = select(
        func.count(TrafficReading.id).label("total_readings"),
        func.avg(TrafficReading.vehicle_count).label("avg_vehicle_count"),
        func.avg(TrafficReading.average_speed).label("avg_speed"),
        func.max(TrafficReading.vehicle_count).label("max_vehicle_count"),
    ).where(TrafficReading.timestamp >= five_min_ago)

    result = await db.execute(query)
    stats = result.first()

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "total_readings": stats.total_readings or 0,
        "avg_vehicle_count": round(stats.avg_vehicle_count or 0, 2),
        "avg_speed": round(stats.avg_speed or 0, 2),
        "max_vehicle_count": stats.max_vehicle_count or 0,
    }
