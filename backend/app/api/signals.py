"""
Traffic Signal Control API
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict
from pydantic import BaseModel
from datetime import datetime

from app.core.database import get_db
from app.services.dsa.signal_optimizer import SignalOptimizer, Direction
from app.models.road_network import TrafficSignal

router = APIRouter()

# Global signal optimizer instance
signal_optimizer = SignalOptimizer(
    default_green_duration=30,
    min_green_duration=10,
    max_green_duration=120,
)


class SignalRequest(BaseModel):
    signal_id: int
    direction: str
    vehicle_count: int
    is_emergency: bool = False


class SignalTimingUpdate(BaseModel):
    signal_id: int
    traffic_data: Dict[str, int]  # direction -> vehicle count


@router.post("/request")
async def request_signal(
    request: SignalRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Request green light for a direction

    This endpoint is called by:
    - Traffic monitoring system with vehicle counts
    - Emergency dispatch system for priority
    - Adaptive control algorithms

    Args:
        request: Signal request details

    Returns:
        Request acknowledgment with priority
    """
    try:
        # Convert direction string to enum
        direction = Direction(request.direction.lower())

        # Submit request to optimizer
        signal_optimizer.request_green_light(
            signal_id=request.signal_id,
            direction=direction,
            vehicle_count=request.vehicle_count,
            is_emergency=request.is_emergency,
        )

        return {
            "success": True,
            "signal_id": request.signal_id,
            "direction": request.direction,
            "queued": True,
            "is_emergency": request.is_emergency,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/process")
async def process_next_signal(db: AsyncSession = Depends(get_db)):
    """
    Process next signal request from priority queue

    This is called by the signal control system periodically

    Returns:
        Next signal change to execute
    """
    next_request = signal_optimizer.process_next_request()

    if not next_request:
        return {
            "success": True,
            "has_request": False,
            "message": "No pending requests",
        }

    return {
        "success": True,
        "has_request": True,
        "signal_change": {
            "signal_id": next_request.signal_id,
            "direction": next_request.direction.value,
            "duration": next_request.duration,
            "reason": next_request.reason,
            "priority": -next_request.priority,  # Un-negate
        },
    }


@router.post("/optimize")
async def optimize_timing(
    update: SignalTimingUpdate,
    db: AsyncSession = Depends(get_db),
):
    """
    Optimize signal timing based on current traffic

    Uses greedy algorithm to maximize throughput

    Args:
        update: Current traffic data by direction

    Returns:
        Optimized timing plan
    """
    try:
        # Convert direction strings to enums
        traffic_data = {
            Direction(dir_str.lower()): count
            for dir_str, count in update.traffic_data.items()
        }

        # Get optimized timing
        timing = signal_optimizer.optimize_signal_timing(
            signal_id=update.signal_id,
            traffic_data=traffic_data,
        )

        # Convert back to strings
        timing_response = {
            direction.value: duration
            for direction, duration in timing.items()
        }

        return {
            "success": True,
            "signal_id": update.signal_id,
            "optimized_timing": timing_response,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{signal_id}/state")
async def get_signal_state(
    signal_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    Get current state of a traffic signal

    Returns:
        Current signal state and timing
    """
    state = signal_optimizer.get_current_state(signal_id)

    if not state:
        raise HTTPException(status_code=404, detail="Signal not found")

    return {
        "signal_id": signal_id,
        "current_state": state.current_state.value,
        "current_direction": state.current_direction.value,
        "green_duration": state.green_duration,
        "yellow_duration": state.yellow_duration,
        "red_duration": state.red_duration,
        "adaptive_mode": state.adaptive_mode,
        "last_state_change": state.last_state_change.isoformat() if state.last_state_change else None,
    }


@router.get("/queue/stats")
async def get_queue_stats():
    """Get signal request queue statistics"""
    stats = signal_optimizer.get_queue_stats()

    return {
        "stats": stats,
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.post("/emergency/preempt")
async def emergency_preempt(
    signal_ids: list[int],
    directions: list[str],
    db: AsyncSession = Depends(get_db),
):
    """
    Preempt signals for emergency vehicle

    This gives immediate green light along emergency route

    Args:
        signal_ids: List of signals to preempt
        directions: Corresponding directions

    Returns:
        Preemption status
    """
    if len(signal_ids) != len(directions):
        raise HTTPException(
            status_code=400,
            detail="signal_ids and directions must have same length",
        )

    # Request high-priority green lights
    for signal_id, direction_str in zip(signal_ids, directions):
        try:
            direction = Direction(direction_str.lower())
            signal_optimizer.request_green_light(
                signal_id=signal_id,
                direction=direction,
                vehicle_count=1,
                is_emergency=True,
                reason="emergency_preemption",
            )
        except ValueError:
            continue

    return {
        "success": True,
        "preempted_signals": signal_ids,
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.post("/register")
async def register_signal(
    signal_id: int,
    initial_direction: str = "north_south",
    db: AsyncSession = Depends(get_db),
):
    """Register a new traffic signal in the optimizer"""
    try:
        direction = Direction(initial_direction.lower())
        signal_optimizer.add_signal(signal_id, direction)

        return {
            "success": True,
            "signal_id": signal_id,
            "message": "Signal registered successfully",
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
