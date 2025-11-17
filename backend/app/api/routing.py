"""
Routing API - DSA-based route optimization
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime

from app.core.database import get_db
from app.services.dsa.routing import RoutingService, RouteAlgorithm
from app.services.dsa.graph import TrafficGraph, Node, Edge
from app.models.road_network import Route as RouteModel

router = APIRouter()


class RouteRequest(BaseModel):
    origin_lat: float
    origin_lon: float
    dest_lat: float
    dest_lon: float
    algorithm: str = "a_star"  # dijkstra or a_star
    is_emergency: bool = False
    num_alternatives: int = 1


class RouteResponse(BaseModel):
    route_id: str
    path: List[int]
    segments: List[int]
    total_distance: float
    total_time: float
    algorithm: str
    is_emergency: bool


# Global graph instance (in production, load from database on startup)
traffic_graph = TrafficGraph()

# TODO: Load graph from database
# This is a placeholder - in production, populate from database
def init_sample_graph():
    """Initialize sample graph for testing"""
    # Add sample nodes (intersections)
    for i in range(1, 11):
        node = Node(
            id=i,
            name=f"Intersection_{i}",
            latitude=17.385 + i * 0.001,
            longitude=78.486 + i * 0.001,
        )
        traffic_graph.add_node(node)

    # Add sample edges (road segments)
    edges = [
        (1, 2, 1000, 50),
        (2, 3, 1200, 60),
        (3, 4, 800, 40),
        (4, 5, 1500, 55),
        (1, 6, 900, 45),
        (6, 7, 1100, 50),
        (7, 8, 1000, 60),
        (8, 5, 700, 40),
        (2, 7, 1300, 50),
        (3, 8, 950, 45),
    ]

    for idx, (src, tgt, length, speed_limit) in enumerate(edges, start=1):
        edge = Edge(
            source=src,
            target=tgt,
            segment_id=idx,
            length=length,
            speed_limit=speed_limit,
            base_travel_time=length / (speed_limit / 3.6),  # Convert km/h to m/s
        )
        traffic_graph.add_edge(edge)


# Initialize on startup
init_sample_graph()

# Routing service
routing_service = RoutingService(traffic_graph)


@router.post("/compute", response_model=RouteResponse)
async def compute_route(
    request: RouteRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Compute optimal route between two points

    Uses A* or Dijkstra algorithm with real-time traffic data

    Args:
        request: Route request with origin, destination, and preferences

    Returns:
        Optimal route with path, distance, and time
    """
    try:
        # Validate algorithm
        try:
            algorithm = RouteAlgorithm(request.algorithm)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid algorithm. Use 'dijkstra' or 'a_star'",
            )

        # Compute route
        result = routing_service.compute_route(
            origin_lat=request.origin_lat,
            origin_lon=request.origin_lon,
            dest_lat=request.dest_lat,
            dest_lon=request.dest_lon,
            algorithm=algorithm,
            is_emergency=request.is_emergency,
        )

        if not result:
            raise HTTPException(
                status_code=404,
                detail="No route found between origin and destination",
            )

        # Save route to database
        route_id = f"route_{datetime.utcnow().timestamp()}"

        # TODO: Save to database with proper geometry
        # route_model = RouteModel(...)

        return RouteResponse(
            route_id=route_id,
            path=result.path,
            segments=result.segments,
            total_distance=result.total_distance,
            total_time=result.total_time,
            algorithm=result.algorithm,
            is_emergency=result.is_emergency,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alternatives")
async def get_alternative_routes(
    request: RouteRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Get alternative routes

    Returns multiple route options sorted by total time

    Args:
        request: Route request with num_alternatives

    Returns:
        List of alternative routes
    """
    try:
        routes = routing_service.get_alternative_routes(
            origin_lat=request.origin_lat,
            origin_lon=request.origin_lon,
            dest_lat=request.dest_lat,
            dest_lon=request.dest_lon,
            num_alternatives=min(request.num_alternatives, 5),
        )

        return {
            "count": len(routes),
            "routes": [
                {
                    "route_id": f"alt_route_{i}",
                    "path": r.path,
                    "segments": r.segments,
                    "total_distance": r.total_distance,
                    "total_time": r.total_time,
                    "algorithm": r.algorithm,
                }
                for i, r in enumerate(routes, 1)
            ],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/update-congestion")
async def update_congestion(
    segment_id: int,
    congestion_factor: float,
    db: AsyncSession = Depends(get_db),
):
    """
    Update congestion factor for a road segment

    This is called periodically by the ML service with predictions

    Args:
        segment_id: Road segment ID
        congestion_factor: Congestion multiplier (1.0 = normal, 2.0 = double travel time)

    Returns:
        Success status
    """
    try:
        traffic_graph.update_congestion(segment_id, congestion_factor)

        return {
            "success": True,
            "segment_id": segment_id,
            "congestion_factor": congestion_factor,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/graph/stats")
async def get_graph_stats():
    """Get traffic graph statistics"""
    stats = traffic_graph.get_stats()

    return {
        "graph": stats,
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.post("/emergency/route")
async def compute_emergency_route(
    origin_lat: float,
    origin_lon: float,
    dest_lat: float,
    dest_lon: float,
    vehicle_type: str = "ambulance",
    db: AsyncSession = Depends(get_db),
):
    """
    Compute priority route for emergency vehicles

    Emergency routes:
    - Use A* algorithm for speed
    - Reduce edge weights significantly
    - Trigger traffic signal preemption

    Args:
        origin_lat, origin_lon: Current location
        dest_lat, dest_lon: Destination
        vehicle_type: Type of emergency vehicle

    Returns:
        Emergency route with signal preemption schedule
    """
    result = routing_service.compute_route(
        origin_lat=origin_lat,
        origin_lon=origin_lon,
        dest_lat=dest_lat,
        dest_lon=dest_lon,
        algorithm=RouteAlgorithm.A_STAR,
        is_emergency=True,
    )

    if not result:
        raise HTTPException(status_code=404, detail="No emergency route found")

    # TODO: Trigger signal preemption along route

    return {
        "route": {
            "path": result.path,
            "segments": result.segments,
            "total_distance": result.total_distance,
            "total_time": result.total_time,
            "eta": result.total_time,
        },
        "vehicle_type": vehicle_type,
        "priority": "HIGH",
        "signals_preempted": len(result.path),
    }
