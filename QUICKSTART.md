# Quick Start Guide

Get the Smart Traffic Management System running in 10 minutes!

## Prerequisites

- Docker Desktop or Docker Engine + Docker Compose
- 8GB RAM minimum
- 20GB free disk space

## Step 1: Clone and Setup (2 minutes)

```bash
# Clone repository
git clone <repository-url>
cd Smart-Traffic-Management-System-using-ML-and-DSA

# Create environment file
cp .env.example .env

# Optional: Edit .env for customization
nano .env
```

## Step 2: Download Real Traffic Data (5 minutes)

### Option A: Use Real METR-LA Dataset

```bash
# Install dataset downloader dependencies
pip install -r scripts/requirements.txt

# Setup Kaggle API (required for dataset download)
# 1. Create Kaggle account at https://www.kaggle.com
# 2. Go to Account Settings -> API -> Create New API Token
# 3. Place kaggle.json in ~/.kaggle/

# Download and preprocess dataset
python scripts/data_collection/download_datasets.py
# Select option 5: "Download and preprocess all"
```

**Dataset Details:**
- **METR-LA**: 207 traffic sensors in Los Angeles
- **34,272 timesteps** (120 days of data)
- **5-minute intervals**
- **Features**: vehicle speed, flow, occupancy

### Option B: Use Synthetic Data (for testing)

```bash
# Generate synthetic traffic data
python scripts/data_collection/download_datasets.py
# Select option 4: "Create synthetic data"
```

## Step 3: Train ML Models (Optional - 30 minutes)

```bash
# Train LSTM traffic prediction model
cd ml-service
python training/train_lstm.py

# This will:
# - Load METR-LA dataset
# - Train LSTM model (128 hidden units, 2 layers)
# - Achieve ~5-10 km/h RMSE on test set
# - Save model to models/saved_models/lstm_traffic_v1.pt
```

**Skip this step** if you want to use the pre-configured model (will use mock predictions).

## Step 4: Start All Services (2 minutes)

```bash
# Build and start all containers
make build
make up

# Or manually:
docker-compose up --build -d
```

**Services starting:**
- PostgreSQL with PostGIS
- Redis cache
- Kafka streaming
- Backend API (FastAPI)
- ML Service
- Frontend (React)
- Prometheus monitoring
- Grafana dashboards

## Step 5: Initialize Database (1 minute)

```bash
# Run database migrations
make db-migrate

# Or manually:
docker-compose exec backend alembic upgrade head
```

## Step 6: Access the System

### Web Interfaces

| Service | URL | Credentials |
|---------|-----|-------------|
| **Frontend Dashboard** | http://localhost:3000 | - |
| **Backend API Docs** | http://localhost:8000/docs | - |
| **ML Service Docs** | http://localhost:8001/docs | - |
| **Grafana Monitoring** | http://localhost:3001 | admin/admin |
| **Prometheus** | http://localhost:9090 | - |

### API Examples

#### 1. Check System Status

```bash
curl http://localhost:8000/api/v1/status
```

#### 2. Get Traffic Predictions

```bash
curl -X POST http://localhost:8000/api/v1/traffic/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sensor_id": 1,
    "sequence": [[45, 50, 0.6, 20, 8, 1, 0]],
    "steps_ahead": 1
  }'
```

#### 3. Compute Optimal Route

```bash
curl -X POST http://localhost:8000/api/v1/routing/compute \
  -H "Content-Type: application/json" \
  -d '{
    "origin_lat": 17.385,
    "origin_lon": 78.486,
    "dest_lat": 17.395,
    "dest_lon": 78.496,
    "algorithm": "a_star"
  }'
```

#### 4. Vehicle Detection from Image

```bash
curl -X POST http://localhost:8001/api/v1/detect \
  -F "file=@path/to/traffic_image.jpg"
```

## Step 7: Run Tests

```bash
# Run comprehensive test suite
bash scripts/local_testing/run_tests.sh

# Or run specific tests
cd backend
pytest tests/ -v

cd ../ml-service
pytest tests/ -v
```

## Common Tasks

### View Logs

```bash
# All services
make logs

# Specific service
docker-compose logs -f backend
docker-compose logs -f ml-service
```

### Stop Services

```bash
make down

# Or:
docker-compose down
```

### Restart Services

```bash
make down
make up
```

### Clean Everything

```bash
# Remove all containers, volumes, and data
make clean
```

## Next Steps

1. **Configure Grafana Dashboards**
   - Visit http://localhost:3001
   - Import dashboards from `monitoring/grafana/dashboards/`

2. **Setup Real Data Collection**
   - Integrate with city traffic sensors
   - See `scripts/city_integration/traffic_sensor_api.py`

3. **Train Custom Models**
   - Use your own traffic data
   - Modify `ml-service/training/train_lstm.py`

4. **Deploy to Production**
   - See `docs/DEPLOYMENT.md`
   - Configure Kubernetes: `kubectl apply -f k8s/`

## Troubleshooting

### Services Not Starting

```bash
# Check Docker status
docker ps

# Check service logs
docker-compose logs backend
docker-compose logs postgres
```

### Database Connection Issues

```bash
# Verify PostgreSQL is running
docker-compose exec postgres pg_isready -U traffic_admin

# Check migrations
docker-compose exec backend alembic current
```

### Out of Memory

```bash
# Increase Docker Desktop memory allocation
# Docker Desktop -> Preferences -> Resources -> Memory -> 8GB+
```

### Port Already in Use

```bash
# Change ports in .env file
# Example: API_PORT=8001 instead of 8000
```

## Architecture Overview

```
┌─────────────────────────────────────────┐
│  Frontend (React + Mapbox)              │
│  http://localhost:3000                  │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│  Backend API (FastAPI)                  │
│  http://localhost:8000                  │
│  - Traffic data management              │
│  - DSA routing (A*, Dijkstra)          │
│  - Signal optimization                  │
└──────┬────────────────┬─────────────────┘
       │                │
┌──────▼───────┐ ┌─────▼──────────────────┐
│ ML Service   │ │ PostgreSQL + PostGIS   │
│ (LSTM,YOLO)  │ │ Traffic data storage   │
└──────────────┘ └────────────────────────┘
```

## Support

- **Documentation**: `docs/` directory
- **GitHub Issues**: [Create Issue](https://github.com/your-repo/issues)
- **Architecture**: See `docs/ARCHITECTURE.md`
- **Deployment**: See `docs/DEPLOYMENT.md`

---

**You're all set! 🚀**

The system is now running with real traffic data, ML predictions, and route optimization. Explore the dashboards and APIs to see it in action!
