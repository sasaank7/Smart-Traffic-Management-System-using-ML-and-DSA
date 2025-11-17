"""
ML Service Main Application

FastAPI service for traffic prediction and vehicle detection
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseSettings, Field
from typing import List, Optional
import numpy as np
import cv2
from datetime import datetime
import yaml
from pathlib import Path

from models.lstm_model import TrafficPredictor
from models.yolo_detector import VehicleDetector


class Settings(BaseSettings):
    """Application settings"""
    MODEL_PATH: str = "./models/saved_models"
    YOLO_MODEL: str = "yolov8n.pt"
    LSTM_MODEL: str = "lstm_traffic_v1.h5"
    CONFIDENCE_THRESHOLD: float = 0.5
    USE_GPU: bool = False

    class Config:
        env_file = ".env"


# Initialize FastAPI app
app = FastAPI(
    title="Smart Traffic ML Service",
    description="Machine Learning service for traffic prediction and vehicle detection",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Settings
settings = Settings()

# Load models
print("Loading models...")

# YOLO detector
yolo_detector = VehicleDetector(
    model_path=f"{settings.MODEL_PATH}/{settings.YOLO_MODEL}",
    confidence_threshold=settings.CONFIDENCE_THRESHOLD,
    device="cuda" if settings.USE_GPU else "cpu",
)

# LSTM predictor
lstm_predictor = None
try:
    lstm_model_path = f"{settings.MODEL_PATH}/{settings.LSTM_MODEL}"
    if Path(lstm_model_path).exists():
        lstm_predictor = TrafficPredictor(model_path=lstm_model_path)
        print("LSTM model loaded successfully")
    else:
        print("LSTM model not found, will use mock predictions")
        lstm_predictor = TrafficPredictor()  # Unloaded model for structure
except Exception as e:
    print(f"Error loading LSTM model: {e}")

print("Models loaded successfully!")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "Smart Traffic ML Service",
        "status": "running",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "models": {
            "yolo": "loaded" if yolo_detector else "not loaded",
            "lstm": "loaded" if lstm_predictor else "not loaded",
        },
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/api/v1/detect")
async def detect_vehicles(file: UploadFile = File(...)):
    """
    Detect vehicles in an uploaded image

    Args:
        file: Image file (JPEG, PNG)

    Returns:
        Detection results with bounding boxes and counts
    """
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Detect vehicles
        detections = yolo_detector.detect(frame)

        # Count by type
        counts = {}
        for det in detections:
            counts[det.class_name] = counts.get(det.class_name, 0) + 1

        return {
            "success": True,
            "timestamp": datetime.utcnow().isoformat(),
            "detections": [det.to_dict() for det in detections],
            "counts": counts,
            "total_vehicles": len(detections),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/predict/traffic")
async def predict_traffic(
    sensor_id: int,
    sequence: List[List[float]],
    steps_ahead: int = 1,
):
    """
    Predict future traffic density

    Args:
        sensor_id: Traffic sensor ID
        sequence: Historical sequence (sequence_length, num_features)
        steps_ahead: Number of future steps to predict

    Returns:
        Traffic predictions
    """
    try:
        if not lstm_predictor:
            # Mock prediction
            return {
                "success": True,
                "sensor_id": sensor_id,
                "predictions": [np.random.uniform(20, 60) for _ in range(steps_ahead)],
                "timestamp": datetime.utcnow().isoformat(),
                "note": "Using mock predictions - LSTM model not loaded",
            }

        # Convert to numpy array
        sequence_array = np.array(sequence)

        # Validate shape
        if len(sequence_array.shape) != 2:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid sequence shape. Expected (sequence_length, features), got {sequence_array.shape}",
            )

        # Predict
        if steps_ahead == 1:
            prediction = lstm_predictor.predict(sequence_array)
            predictions = [prediction]
        else:
            predictions = lstm_predictor.predict_future(sequence_array, steps=steps_ahead)

        # Classify congestion level
        congestion_levels = []
        for pred in predictions:
            if pred < 20:
                level = "low"
            elif pred < 40:
                level = "medium"
            elif pred < 60:
                level = "high"
            else:
                level = "severe"
            congestion_levels.append(level)

        return {
            "success": True,
            "sensor_id": sensor_id,
            "predictions": predictions,
            "congestion_levels": congestion_levels,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/analyze/anomaly")
async def detect_anomaly(
    sensor_id: int,
    current_metrics: dict,
    historical_avg: dict,
):
    """
    Detect traffic anomalies

    Args:
        sensor_id: Traffic sensor ID
        current_metrics: Current traffic metrics
        historical_avg: Historical average metrics

    Returns:
        Anomaly detection result
    """
    try:
        # Simple anomaly detection based on deviation
        deviations = {}
        is_anomaly = False
        anomaly_score = 0.0

        for key in current_metrics:
            if key in historical_avg:
                current = current_metrics[key]
                avg = historical_avg[key]

                if avg > 0:
                    deviation = abs(current - avg) / avg
                    deviations[key] = deviation

                    # Threshold: 50% deviation
                    if deviation > 0.5:
                        is_anomaly = True
                        anomaly_score = max(anomaly_score, deviation)

        return {
            "success": True,
            "sensor_id": sensor_id,
            "is_anomaly": is_anomaly,
            "anomaly_score": anomaly_score,
            "deviations": deviations,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/stats")
async def get_stats():
    """Get ML service statistics"""
    return {
        "yolo_stats": yolo_detector.get_stats() if yolo_detector else {},
        "timestamp": datetime.utcnow().isoformat(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info",
    )
