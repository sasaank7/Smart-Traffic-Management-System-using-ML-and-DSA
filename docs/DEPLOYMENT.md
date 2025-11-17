# Deployment Guide

Complete guide for deploying Smart Traffic Management System to production.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Development](#local-development)
3. [Docker Deployment](#docker-deployment)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Cloud Deployment](#cloud-deployment)
6. [Configuration](#configuration)
7. [Monitoring](#monitoring)
8. [Troubleshooting](#troubleshooting)

## Prerequisites

### Required Software

- Docker 20.10+ and Docker Compose 2.0+
- Python 3.9+ (for local development)
- Node.js 16+ (for frontend development)
- PostgreSQL 14+ with PostGIS extension
- Kubernetes 1.24+ (for production)
- kubectl CLI
- Helm 3+ (optional)

### Hardware Requirements

**Minimum (Development)**:
- 4 CPU cores
- 8 GB RAM
- 50 GB storage

**Recommended (Production)**:
- 16 CPU cores
- 32 GB RAM
- 500 GB SSD storage
- GPU (optional, for ML inference acceleration)

## Local Development

### 1. Clone Repository

```bash
git clone <repository-url>
cd Smart-Traffic-Management-System-using-ML-and-DSA
```

### 2. Setup Environment

```bash
cp .env.example .env
# Edit .env with your configuration
nano .env
```

### 3. Run with Docker Compose

```bash
# Build and start all services
make build
make up

# Or manually
docker-compose up --build -d
```

### 4. Access Services

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000/docs
- ML Service: http://localhost:8001/docs
- Grafana: http://localhost:3001

## Docker Deployment

### Production Build

```bash
# Build optimized images
docker-compose -f docker-compose.yml -f docker-compose.prod.yml build

# Start services
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### Database Migrations

```bash
# Run migrations
docker-compose exec backend alembic upgrade head

# Seed initial data
docker-compose exec backend python scripts/seed_data.py
```

## Kubernetes Deployment

### 1. Create Namespace

```bash
kubectl create namespace smart-traffic
```

### 2. Deploy PostgreSQL

```bash
kubectl apply -f k8s/postgres/
```

### 3. Deploy Redis and Kafka

```bash
kubectl apply -f k8s/redis/
kubectl apply -f k8s/kafka/
```

### 4. Deploy Application Services

```bash
# Backend
kubectl apply -f k8s/backend/

# ML Service
kubectl apply -f k8s/ml-service/

# Frontend
kubectl apply -f k8s/frontend/
```

### 5. Verify Deployment

```bash
kubectl get pods -n smart-traffic
kubectl get services -n smart-traffic
```

### 6. Access Application

```bash
# Get service URLs
kubectl get svc -n smart-traffic

# Port forward for testing
kubectl port-forward svc/frontend 3000:80 -n smart-traffic
```

## Cloud Deployment

### AWS Deployment

#### Using EKS (Elastic Kubernetes Service)

```bash
# Create EKS cluster
eksctl create cluster \
  --name smart-traffic-cluster \
  --region us-east-1 \
  --nodegroup-name standard-workers \
  --node-type t3.xlarge \
  --nodes 3 \
  --nodes-min 1 \
  --nodes-max 5

# Deploy application
kubectl apply -f k8s/

# Create load balancer
kubectl apply -f k8s/ingress/aws-alb.yaml
```

#### Using RDS for PostgreSQL

```bash
# Create RDS instance
aws rds create-db-instance \
  --db-instance-identifier smart-traffic-db \
  --db-instance-class db.r5.large \
  --engine postgres \
  --engine-version 14.7 \
  --allocated-storage 100 \
  --master-username admin \
  --master-user-password <password>

# Update .env with RDS endpoint
POSTGRES_HOST=<rds-endpoint>
```

### GCP Deployment

#### Using GKE (Google Kubernetes Engine)

```bash
# Create GKE cluster
gcloud container clusters create smart-traffic-cluster \
  --region us-central1 \
  --num-nodes 3 \
  --machine-type n1-standard-4 \
  --enable-autoscaling \
  --min-nodes 1 \
  --max-nodes 5

# Deploy application
kubectl apply -f k8s/

# Create ingress
kubectl apply -f k8s/ingress/gcp-ingress.yaml
```

### Azure Deployment

```bash
# Create AKS cluster
az aks create \
  --resource-group smart-traffic-rg \
  --name smart-traffic-cluster \
  --node-count 3 \
  --node-vm-size Standard_D4s_v3 \
  --enable-addons monitoring

# Deploy application
kubectl apply -f k8s/
```

## Configuration

### Environment Variables

Key configuration in `.env`:

```bash
# Application
ENVIRONMENT=production
DEBUG=False
SECRET_KEY=<generate-strong-secret>
JWT_SECRET_KEY=<generate-jwt-secret>

# Database
POSTGRES_HOST=postgres
POSTGRES_DB=smart_traffic_db
POSTGRES_USER=traffic_admin
POSTGRES_PASSWORD=<strong-password>

# Redis
REDIS_HOST=redis
REDIS_PASSWORD=<redis-password>

# Kafka
KAFKA_BOOTSTRAP_SERVERS=kafka:9092

# External Services
GOOGLE_MAPS_API_KEY=<your-api-key>
MAPBOX_TOKEN=<your-mapbox-token>
```

### Secrets Management

#### Kubernetes Secrets

```bash
# Create secrets
kubectl create secret generic app-secrets \
  --from-literal=postgres-password=<password> \
  --from-literal=jwt-secret=<secret> \
  -n smart-traffic

# Or from file
kubectl create secret generic app-secrets \
  --from-env-file=.env.production \
  -n smart-traffic
```

#### AWS Secrets Manager

```bash
# Store secrets
aws secretsmanager create-secret \
  --name smart-traffic/postgres \
  --secret-string '{"password":"<password>"}'

# Update deployment to use secrets
# See k8s/backend/deployment-aws-secrets.yaml
```

## Monitoring

### Prometheus & Grafana

```bash
# Access Grafana
kubectl port-forward svc/grafana 3001:80 -n smart-traffic

# Default credentials: admin/admin
# Import dashboards from monitoring/grafana/dashboards/
```

### Logging

```bash
# View logs
kubectl logs -f deployment/backend -n smart-traffic
kubectl logs -f deployment/ml-service -n smart-traffic

# Aggregate logs with ELK stack (if deployed)
kubectl port-forward svc/kibana 5601:5601 -n logging
```

### Alerts

Configure alerts in `monitoring/prometheus/alerts.yml`:

```yaml
groups:
  - name: smart-traffic-alerts
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status="500"}[5m]) > 0.05
        for: 5m
        annotations:
          summary: "High error rate detected"
```

## SSL/TLS Configuration

### Using cert-manager (Kubernetes)

```bash
# Install cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Create issuer
kubectl apply -f k8s/cert-manager/issuer.yaml

# Certificate will be auto-generated via ingress annotation
```

### Using Let's Encrypt (Docker)

```bash
# Install certbot
certbot --nginx -d yourdomain.com

# Update nginx.conf with SSL certificates
```

## Backup & Recovery

### Database Backup

```bash
# Automated backup
kubectl create cronjob db-backup \
  --image=postgres:15 \
  --schedule="0 2 * * *" \
  -- pg_dump -U admin smart_traffic_db > /backup/db.sql

# Manual backup
kubectl exec -it postgres-0 -- pg_dump -U admin smart_traffic_db > backup.sql
```

### Restore Database

```bash
kubectl exec -i postgres-0 -- psql -U admin smart_traffic_db < backup.sql
```

## Scaling

### Horizontal Pod Autoscaling

```bash
# Enable autoscaling
kubectl autoscale deployment backend \
  --cpu-percent=70 \
  --min=3 \
  --max=10 \
  -n smart-traffic
```

### Database Scaling

```bash
# PostgreSQL read replicas
kubectl apply -f k8s/postgres/read-replica.yaml

# Redis cluster mode
kubectl apply -f k8s/redis/cluster.yaml
```

## Troubleshooting

### Common Issues

**1. Database Connection Failed**

```bash
# Check postgres pod
kubectl get pods -n smart-traffic
kubectl logs postgres-0 -n smart-traffic

# Test connection
kubectl exec -it backend-xxx -- python -c "from app.core.database import engine; print(engine)"
```

**2. ML Service Out of Memory**

```bash
# Increase memory limits
kubectl edit deployment ml-service -n smart-traffic

# Add memory limits
resources:
  limits:
    memory: "4Gi"
  requests:
    memory: "2Gi"
```

**3. Kafka Consumer Lag**

```bash
# Check consumer group
kubectl exec -it kafka-0 -- kafka-consumer-groups.sh \
  --bootstrap-server localhost:9092 \
  --describe --group traffic-management-group
```

### Performance Tuning

1. **Database Optimization**
   - Add indexes
   - Connection pooling
   - Query optimization

2. **Redis Caching**
   - Cache frequently accessed data
   - Set appropriate TTL

3. **Load Balancing**
   - Use nginx/HAProxy
   - Configure health checks

## Security Checklist

- [ ] Change default passwords
- [ ] Enable SSL/TLS
- [ ] Configure firewall rules
- [ ] Set up authentication
- [ ] Enable audit logging
- [ ] Regular security updates
- [ ] Backup encryption
- [ ] API rate limiting
- [ ] CORS configuration
- [ ] Environment variable encryption

## Production Checklist

- [ ] All tests passing
- [ ] Documentation updated
- [ ] Monitoring configured
- [ ] Backup strategy in place
- [ ] SSL certificates configured
- [ ] Resource limits set
- [ ] Auto-scaling configured
- [ ] Disaster recovery plan
- [ ] Security audit completed
- [ ] Load testing performed

## Support

For deployment issues:
- GitHub Issues: [Create Issue](https://github.com/your-repo/issues)
- Email: devops@smarttraffic.com
- Documentation: [Wiki](https://github.com/your-repo/wiki)
