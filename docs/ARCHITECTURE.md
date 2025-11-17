# System Architecture

## Overview

The Smart Traffic Management System is built using a microservices architecture with the following key components:

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Client Layer                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Web App    │  │  Mobile App  │  │   Admin      │          │
│  │   (React)    │  │              │  │   Portal     │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────┼────────────────────────────────────┐
│                   API Gateway / Load Balancer                    │
│                         (Nginx / AWS ALB)                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
┌───────▼────────┐  ┌───────▼────────┐  ┌───────▼────────┐
│   Backend API  │  │   ML Service   │  │   WebSocket    │
│   (FastAPI)    │  │   (FastAPI)    │  │   Service      │
└───────┬────────┘  └───────┬────────┘  └───────┬────────┘
        │                    │                    │
        └────────────────────┼────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
┌───────▼────────┐  ┌───────▼────────┐  ┌───────▼────────┐
│   PostgreSQL   │  │     Redis      │  │     Kafka      │
│   + PostGIS    │  │    (Cache)     │  │   (Streaming)  │
└────────────────┘  └────────────────┘  └────────────────┘
```

## Components

### 1. Backend API (FastAPI)

**Responsibilities:**
- RESTful API endpoints
- Business logic orchestration
- Database operations (CRUD)
- Authentication & authorization
- Request validation
- DSA algorithms (routing, signal optimization)

**Technology Stack:**
- FastAPI (async web framework)
- SQLAlchemy (ORM)
- Alembic (migrations)
- Pydantic (validation)
- Python 3.11+

**Key Modules:**
- `app/api/` - API route handlers
- `app/models/` - Database models
- `app/services/dsa/` - DSA algorithms
- `app/core/` - Configuration and database

### 2. ML Service

**Responsibilities:**
- Traffic prediction (LSTM)
- Vehicle detection (YOLOv8)
- Anomaly detection
- Model inference
- Model training (offline)

**Technology Stack:**
- PyTorch (LSTM models)
- Ultralytics YOLOv8 (object detection)
- OpenCV (image processing)
- FastAPI (API server)
- NumPy, Pandas (data processing)

**Key Modules:**
- `models/lstm_model.py` - LSTM predictor
- `models/yolo_detector.py` - YOLO detector
- `training/` - Training scripts
- `inference/` - Inference optimization

### 3. Frontend Dashboard (React)

**Responsibilities:**
- User interface
- Real-time map visualization
- Traffic analytics charts
- Signal control interface
- User management

**Technology Stack:**
- React 18
- Material-UI (components)
- Mapbox GL JS (maps)
- Recharts (charts)
- React Query (data fetching)
- WebSocket (real-time updates)

### 4. PostgreSQL + PostGIS

**Responsibilities:**
- Primary data storage
- Geospatial data
- Traffic readings history
- Predictions storage
- User data

**Schema:**
- Traffic sensors & readings
- Road network (nodes & edges)
- Predictions & anomalies
- Traffic signals
- Routes
- Users

### 5. Redis

**Responsibilities:**
- Session caching
- API response caching
- Rate limiting
- Real-time data cache
- Message broker (pub/sub)

**Usage:**
- Cache TTL: 5 minutes (configurable)
- Session store
- Temporary route cache

### 6. Kafka

**Responsibilities:**
- Real-time data streaming
- Event-driven architecture
- Traffic data pipeline
- Prediction distribution
- Anomaly alerts

**Topics:**
- `traffic-data` - Sensor readings
- `traffic-predictions` - ML predictions
- `traffic-anomalies` - Detected anomalies
- `emergency-vehicles` - Emergency routing

## Data Flow

### 1. Traffic Data Collection

```
Sensors → Kafka Producer → Kafka Topic → Kafka Consumer → Database
                                            ↓
                                       ML Service (Prediction)
                                            ↓
                                     Update Graph Weights
```

### 2. Route Computation

```
User Request → API → Load Graph from Cache/DB
                          ↓
                    DSA Algorithm (A*/Dijkstra)
                          ↓
                    Apply Congestion Factors
                          ↓
                    Return Optimized Route
                          ↓
                    Cache Result (Redis)
```

### 3. Traffic Prediction

```
Historical Data → LSTM Model → Prediction
                                    ↓
                            Store in Database
                                    ↓
                            Update Graph Weights
                                    ↓
                            Publish to Kafka
                                    ↓
                            Notify Frontend (WebSocket)
```

### 4. Signal Optimization

```
Traffic Queue Data → Signal Optimizer (Priority Queue)
                            ↓
                    Calculate Optimal Timing
                            ↓
                    Apply Signal Change
                            ↓
                    Log to Database
```

## DSA Algorithms Implementation

### 1. Routing Algorithms

**A* Algorithm:**
```python
function a_star(start, goal):
    open_set = {start}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set not empty:
        current = node in open_set with lowest f_score

        if current == goal:
            return reconstruct_path()

        for neighbor in neighbors(current):
            tentative_g = g_score[current] + distance(current, neighbor)

            if tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal)

    return failure
```

**Heuristic:** Haversine distance / max speed

**Time Complexity:** O((V + E) log V)
**Space Complexity:** O(V)

### 2. Signal Scheduling

**Priority Queue:**
```python
class SignalRequest:
    priority: int  # Negated for max-heap
    signal_id: int
    direction: str
    duration: int

heap = []  # Min-heap with negated priorities
heappush(heap, SignalRequest(-priority, ...))
next_request = heappop(heap)  # O(log N)
```

**Adaptive Timing (Webster's Method):**
```
optimal_cycle = (1.5 * lost_time + 5) / (1 - sum(flow_ratios))
green_time = (flow_ratio / sum(flow_ratios)) * cycle_length
```

## ML Model Architecture

### LSTM Traffic Predictor

```
Input: (batch_size, sequence_length=12, features=7)
       Features: vehicle_count, speed, occupancy, temp, hour, day, weather

Layer 1: LSTM(input_size=7, hidden_size=128, num_layers=2)
Layer 2: Dropout(p=0.2)
Layer 3: Fully Connected(128 → 1)

Output: Predicted traffic density (vehicles/km)
```

**Training:**
- Loss: MSE
- Optimizer: Adam (lr=0.001)
- Batch size: 32
- Early stopping: patience=10

### YOLOv8 Vehicle Detector

```
Input: Image (640x640)
Backbone: CSPDarknet
Neck: PANet
Head: YOLO Detection Head

Output: Bounding boxes + class probabilities
Classes: car, truck, bus, motorcycle, bicycle
```

**Inference:**
- Confidence threshold: 0.5
- NMS IoU threshold: 0.45
- FPS: ~30 on GPU, ~5 on CPU

## Scalability

### Horizontal Scaling

1. **Stateless Services:**
   - Backend API (multiple replicas)
   - ML Service (load balanced)

2. **Database Scaling:**
   - Read replicas for PostgreSQL
   - Redis cluster mode
   - Kafka partitions

3. **Auto-scaling:**
   - Kubernetes HPA based on CPU/memory
   - Target: 70% CPU utilization

### Performance Optimizations

1. **Caching Strategy:**
   - Route cache: 5 min TTL
   - Prediction cache: 10 min TTL
   - Graph cache: 30 min TTL

2. **Database Optimization:**
   - Spatial indexes (GIST)
   - Compound indexes on frequently queried fields
   - Connection pooling (size=20)

3. **Async Processing:**
   - Async database queries (asyncpg)
   - Background tasks (FastAPI BackgroundTasks)
   - Kafka async producers

## Security Architecture

1. **Authentication:**
   - JWT tokens (30 min expiry)
   - Refresh tokens (7 day expiry)
   - Password hashing (bcrypt)

2. **Authorization:**
   - Role-based access control (RBAC)
   - Roles: admin, traffic_operator, analyst, viewer

3. **API Security:**
   - Rate limiting (60 req/min per user)
   - CORS configuration
   - SQL injection prevention (ORM)
   - XSS protection (input validation)

4. **Network Security:**
   - HTTPS/TLS encryption
   - Private network for services
   - Firewall rules

## Monitoring & Observability

1. **Metrics (Prometheus):**
   - Request rate
   - Response time
   - Error rate
   - CPU/Memory usage
   - Database connections

2. **Logging:**
   - Structured logging (JSON)
   - Log levels: DEBUG, INFO, WARNING, ERROR
   - Centralized logging (ELK stack)

3. **Tracing:**
   - Distributed tracing (Jaeger)
   - Request tracking across services

4. **Dashboards (Grafana):**
   - System health
   - Traffic metrics
   - ML model performance
   - API analytics

## Disaster Recovery

1. **Backup Strategy:**
   - Database: Daily automated backups
   - Retention: 30 days
   - Off-site backup storage

2. **High Availability:**
   - Multi-region deployment
   - Database replication
   - Load balancer health checks

3. **Recovery Time:**
   - RTO (Recovery Time Objective): 1 hour
   - RPO (Recovery Point Objective): 15 minutes
