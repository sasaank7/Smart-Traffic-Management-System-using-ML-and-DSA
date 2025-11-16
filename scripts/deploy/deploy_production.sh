#!/bin/bash

# Production Deployment Script for Smart Traffic Management System

set -e

echo "=========================================="
echo "Smart Traffic Management System"
echo "Production Deployment Script"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Configuration
NAMESPACE="smart-traffic"
ENVIRONMENT=${1:-production}

echo -e "${YELLOW}Deploying to: $ENVIRONMENT${NC}"
echo ""

# Pre-deployment checks
echo "1. Pre-deployment checks..."

# Check kubectl
if ! command -v kubectl &> /dev/null; then
    echo -e "${RED}kubectl not found. Please install kubectl.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ kubectl installed${NC}"

# Check cluster connection
if ! kubectl cluster-info &> /dev/null; then
    echo -e "${RED}Cannot connect to Kubernetes cluster${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Connected to Kubernetes cluster${NC}"

# Check context
CURRENT_CONTEXT=$(kubectl config current-context)
echo -e "${YELLOW}Current context: $CURRENT_CONTEXT${NC}"
read -p "Continue with this context? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
fi

echo ""
echo "2. Creating namespace..."
kubectl apply -f k8s/namespace.yaml
echo -e "${GREEN}✓ Namespace created${NC}"

echo ""
echo "3. Creating secrets..."
echo -e "${YELLOW}⚠ Make sure to update k8s/secrets/secrets-template.yaml with real values${NC}"
read -p "Have you updated the secrets? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${RED}Please update secrets first${NC}"
    exit 1
fi

kubectl apply -f k8s/secrets/secrets-template.yaml
echo -e "${GREEN}✓ Secrets created${NC}"

echo ""
echo "4. Deploying PostgreSQL..."
kubectl apply -f k8s/postgres/
kubectl wait --for=condition=ready pod -l app=postgres -n $NAMESPACE --timeout=300s
echo -e "${GREEN}✓ PostgreSQL deployed${NC}"

echo ""
echo "5. Deploying Redis..."
kubectl apply -f k8s/redis/
kubectl wait --for=condition=ready pod -l app=redis -n $NAMESPACE --timeout=120s
echo -e "${GREEN}✓ Redis deployed${NC}"

echo ""
echo "6. Deploying Kafka..."
kubectl apply -f k8s/kafka/
echo -e "${GREEN}✓ Kafka deployed${NC}"
sleep 30  # Wait for Kafka to be ready

echo ""
echo "7. Running database migrations..."
# Wait for backend pod
kubectl wait --for=condition=ready pod -l app=backend -n $NAMESPACE --timeout=300s

# Run migrations
BACKEND_POD=$(kubectl get pods -n $NAMESPACE -l app=backend -o jsonpath='{.items[0].metadata.name}')
kubectl exec -n $NAMESPACE $BACKEND_POD -- alembic upgrade head
echo -e "${GREEN}✓ Database migrations completed${NC}"

echo ""
echo "8. Deploying backend services..."
kubectl apply -f k8s/backend/
kubectl rollout status deployment/backend -n $NAMESPACE
echo -e "${GREEN}✓ Backend deployed${NC}"

echo ""
echo "9. Deploying ML service..."
kubectl apply -f k8s/ml-service/
kubectl rollout status deployment/ml-service -n $NAMESPACE
echo -e "${GREEN}✓ ML service deployed${NC}"

echo ""
echo "10. Deploying frontend..."
kubectl apply -f k8s/frontend/
kubectl rollout status deployment/frontend -n $NAMESPACE
echo -e "${GREEN}✓ Frontend deployed${NC}"

echo ""
echo "11. Deploying ingress..."
kubectl apply -f k8s/ingress/
echo -e "${GREEN}✓ Ingress deployed${NC}"

echo ""
echo "12. Deploying monitoring..."
kubectl apply -f k8s/monitoring/
echo -e "${GREEN}✓ Monitoring deployed${NC}"

echo ""
echo "=========================================="
echo -e "${GREEN}✓ Deployment completed successfully!${NC}"
echo "=========================================="
echo ""

# Get service URLs
echo "Service endpoints:"
echo ""

FRONTEND_IP=$(kubectl get svc frontend -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
if [ -z "$FRONTEND_IP" ]; then
    FRONTEND_IP=$(kubectl get svc frontend -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
fi

if [ ! -z "$FRONTEND_IP" ]; then
    echo "Frontend: http://$FRONTEND_IP"
else
    echo "Frontend: Waiting for LoadBalancer IP..."
fi

echo ""
echo "Check status with:"
echo "  kubectl get pods -n $NAMESPACE"
echo "  kubectl get svc -n $NAMESPACE"
echo ""
echo "View logs with:"
echo "  kubectl logs -f deployment/backend -n $NAMESPACE"
echo "  kubectl logs -f deployment/ml-service -n $NAMESPACE"
echo ""
echo "Monitor with:"
echo "  kubectl port-forward svc/grafana 3001:80 -n $NAMESPACE"
echo ""
