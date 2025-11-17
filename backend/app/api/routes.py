"""
General API routes
"""
from fastapi import APIRouter
from datetime import datetime

router = APIRouter()


@app.get("/api/v1/status")
async def get_status():
    """Get system status"""
    return {
        "status": "operational",
        "services": {
            "api": "running",
            "database": "connected",
            "ml_service": "running",
            "kafka": "connected",
        },
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/api/v1/info")
async def get_info():
    """Get system information"""
    return {
        "name": "Smart Traffic Management System",
        "version": "1.0.0",
        "description": "ML and DSA powered traffic management",
        "features": [
            "Real-time traffic prediction",
            "Dynamic route optimization",
            "Adaptive signal control",
            "Emergency vehicle priority",
            "Anomaly detection",
        ],
    }
