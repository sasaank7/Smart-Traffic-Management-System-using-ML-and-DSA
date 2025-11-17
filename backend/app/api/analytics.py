"""
Analytics and reporting API
"""
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from datetime import datetime, timedelta
from typing import Optional

from app.core.database import get_db
from app.models.traffic import TrafficReading, TrafficAnomaly

router = APIRouter()


@router.get("/summary")
async def get_summary(
    hours: int = 24,
    db: AsyncSession = Depends(get_db),
):
    """
    Get traffic summary for last N hours

    Returns aggregated statistics
    """
    start_time = datetime.utcnow() - timedelta(hours=hours)

    # Traffic readings stats
    query = select(
        func.count(TrafficReading.id).label("total_readings"),
        func.avg(TrafficReading.vehicle_count).label("avg_vehicles"),
        func.avg(TrafficReading.average_speed).label("avg_speed"),
        func.max(TrafficReading.vehicle_count).label("max_vehicles"),
    ).where(TrafficReading.timestamp >= start_time)

    result = await db.execute(query)
    traffic_stats = result.first()

    # Anomalies count
    anomaly_query = select(func.count(TrafficAnomaly.id)).where(
        TrafficAnomaly.detected_at >= start_time
    )
    anomaly_result = await db.execute(anomaly_query)
    anomaly_count = anomaly_result.scalar()

    return {
        "period_hours": hours,
        "traffic": {
            "total_readings": traffic_stats.total_readings or 0,
            "avg_vehicles": round(traffic_stats.avg_vehicles or 0, 2),
            "avg_speed": round(traffic_stats.avg_speed or 0, 2),
            "max_vehicles": traffic_stats.max_vehicles or 0,
        },
        "anomalies": anomaly_count or 0,
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/trends")
async def get_trends(
    days: int = 7,
    db: AsyncSession = Depends(get_db),
):
    """
    Get traffic trends over time

    Returns time-series data for charts
    """
    # This is a simplified version
    # In production, use proper time-series aggregation

    return {
        "period_days": days,
        "data_points": [
            {"date": "2024-01-01", "avg_vehicles": 45, "avg_speed": 52},
            {"date": "2024-01-02", "avg_vehicles": 48, "avg_speed": 50},
            # ... more data points
        ],
        "message": "Trend analysis in progress",
    }
