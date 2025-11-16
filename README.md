# Smart Traffic Management System using ML and DSA

A production-ready intelligent traffic management system that combines Machine Learning for traffic prediction and Data Structures & Algorithms for route optimization and signal control.

## 🚀 Features

- **Real-time Traffic Prediction** using LSTM networks
- **Vehicle Detection & Anomaly Detection** using YOLOv8
- **Dynamic Route Optimization** using A* and Dijkstra algorithms
- **Adaptive Traffic Signal Control** with priority queue scheduling
- **Emergency Vehicle Priority Routing**
- **Real-time Dashboard** with interactive maps and analytics
- **Scalable Microservices Architecture**
- **Kafka-based Real-time Data Streaming**
- **PostgreSQL + PostGIS** for geospatial data
- **Production-ready Deployment** with Docker & Kubernetes

## 📋 Prerequisites

- Docker & Docker Compose
- Python 3.9+
- Node.js 16+
- PostgreSQL 14+ with PostGIS extension
- NVIDIA GPU (optional, for ML inference acceleration)

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Data Sources Layer                       │
│  CCTV Cameras │ GPS Sensors │ Traffic Sensors │ APIs        │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                   Kafka Streaming Layer                      │
│        Real-time Data Ingestion & Stream Processing         │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────┴────────────────┐
        │                                 │
┌───────▼────────┐              ┌────────▼──────────┐
│  ML Service    │              │  DSA Service      │
│  - LSTM        │              │  - Dijkstra/A*    │
│  - YOLOv8      │              │  - Signal Opt     │
│  - Prediction  │              │  - Emergency Path │
└───────┬────────┘              └────────┬──────────┘
        │                                 │
        └────────────────┬────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                   API Gateway (FastAPI)                      │
│         REST APIs │ WebSocket │ Authentication              │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                    Frontend Dashboard                        │
│      React │ Mapbox │ Real-time Charts │ Analytics          │
└─────────────────────────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                PostgreSQL + PostGIS + Redis                  │
│         Traffic Data │ Geospatial │ Cache                   │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Using Docker Compose (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd Smart-Traffic-Management-System-using-ML-and-DSA

# Create environment file
cp .env.example .env

# Edit .env with your configuration
nano .env

# Build and start all services
docker-compose up --build

# Access the services:
# - Frontend Dashboard: http://localhost:3000
# - API Documentation: http://localhost:8000/docs
# - Grafana Monitoring: http://localhost:3001
```

### Manual Setup

#### 1. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run database migrations
alembic upgrade head

# Start the API server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

#### 2. ML Service Setup

```bash
cd ml-service

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download pre-trained models (if available)
python scripts/download_models.py

# Start ML inference server
python main.py
```

#### 3. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm start

# Build for production
npm run build
```

## 📊 Data Pipeline

### 1. Data Collection

```bash
# Start data collection from sensors
python scripts/data_collection/collect_traffic_data.py

# Process CCTV feeds
python scripts/data_collection/process_cctv_feeds.py
```

### 2. Model Training

```bash
# Train LSTM traffic prediction model
python ml-service/training/train_lstm.py --config configs/lstm_config.yaml

# Train YOLOv8 vehicle detection model
python ml-service/training/train_yolo.py --config configs/yolo_config.yaml
```

### 3. Kafka Streaming Setup

```bash
# Create Kafka topics
python scripts/kafka/create_topics.py

# Start Kafka consumers
python scripts/kafka/consume_traffic_data.py
```

## 🧪 Testing

```bash
# Run backend tests
cd backend
pytest tests/ -v --cov=app

# Run ML service tests
cd ml-service
pytest tests/ -v

# Run frontend tests
cd frontend
npm test
```

## 📦 Deployment

### Kubernetes Deployment

```bash
# Create namespace
kubectl create namespace smart-traffic

# Deploy PostgreSQL
kubectl apply -f k8s/postgres/

# Deploy Redis
kubectl apply -f k8s/redis/

# Deploy Kafka
kubectl apply -f k8s/kafka/

# Deploy backend services
kubectl apply -f k8s/backend/

# Deploy ML services
kubectl apply -f k8s/ml-service/

# Deploy frontend
kubectl apply -f k8s/frontend/

# Check deployment status
kubectl get pods -n smart-traffic
```

### AWS/GCP Deployment

See [DEPLOYMENT.md](docs/DEPLOYMENT.md) for detailed cloud deployment instructions.

## 📚 API Documentation

Once the backend is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 🔧 Configuration

Key configuration files:
- `.env` - Environment variables
- `backend/app/core/config.py` - Backend configuration
- `ml-service/config/model_config.yaml` - ML model configurations
- `docker-compose.yml` - Docker services configuration
- `k8s/` - Kubernetes deployment configurations

## 📈 Monitoring & Observability

- **Prometheus**: Metrics collection (http://localhost:9090)
- **Grafana**: Dashboards and visualization (http://localhost:3001)
- **ELK Stack**: Centralized logging
- **Jaeger**: Distributed tracing

## 🛡️ Security

- JWT-based authentication
- Role-based access control (RBAC)
- API rate limiting
- HTTPS/TLS encryption
- SQL injection prevention
- XSS protection
- CORS configuration

## 📖 Documentation

- [Architecture Guide](docs/ARCHITECTURE.md)
- [API Reference](docs/API.md)
- [ML Models Documentation](docs/ML_MODELS.md)
- [DSA Algorithms Documentation](docs/DSA_ALGORITHMS.md)
- [Deployment Guide](docs/DEPLOYMENT.md)
- [Contributing Guide](CONTRIBUTING.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Team

- **ML Team**: Traffic prediction, anomaly detection
- **DSA Team**: Routing algorithms, signal optimization
- **Backend Team**: API development, data pipeline
- **Frontend Team**: Dashboard, user interface
- **DevOps Team**: Infrastructure, deployment, monitoring

## 📞 Support

For issues and questions:
- GitHub Issues: [Create an issue](https://github.com/your-repo/issues)
- Email: support@smarttraffic.com
- Documentation: [Wiki](https://github.com/your-repo/wiki)

## 🙏 Acknowledgments

- YOLOv8 by Ultralytics
- OpenStreetMap for map data
- Mapbox for visualization
- TensorFlow/PyTorch communities
- Open-source contributors

---

**Built with ❤️ for smarter cities**
