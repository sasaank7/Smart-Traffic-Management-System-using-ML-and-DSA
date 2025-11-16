# Smart Traffic Management System - Makefile

.PHONY: help build up down logs clean test

help:
	@echo "Smart Traffic Management System - Make Commands"
	@echo "================================================"
	@echo "make build          - Build all Docker containers"
	@echo "make up             - Start all services"
	@echo "make down           - Stop all services"
	@echo "make logs           - View logs from all services"
	@echo "make clean          - Remove all containers, volumes, and images"
	@echo "make test           - Run tests"
	@echo "make db-migrate     - Run database migrations"
	@echo "make db-seed        - Seed database with sample data"
	@echo "make lint           - Run code linters"
	@echo "make format         - Format code"

build:
	@echo "Building Docker containers..."
	docker-compose build

up:
	@echo "Starting all services..."
	docker-compose up -d
	@echo "Services started. Access:"
	@echo "  - Frontend: http://localhost:3000"
	@echo "  - Backend API: http://localhost:8000/docs"
	@echo "  - ML Service: http://localhost:8001/docs"
	@echo "  - Grafana: http://localhost:3001"

down:
	@echo "Stopping all services..."
	docker-compose down

logs:
	docker-compose logs -f

clean:
	@echo "Cleaning up..."
	docker-compose down -v --rmi all
	@echo "Cleanup complete"

test:
	@echo "Running backend tests..."
	cd backend && pytest tests/ -v
	@echo "Running ML service tests..."
	cd ml-service && pytest tests/ -v

db-migrate:
	@echo "Running database migrations..."
	docker-compose exec backend alembic upgrade head

db-seed:
	@echo "Seeding database..."
	docker-compose exec backend python scripts/seed_data.py

lint:
	@echo "Running linters..."
	cd backend && flake8 app/
	cd ml-service && flake8 .

format:
	@echo "Formatting code..."
	cd backend && black app/
	cd ml-service && black .

# Development commands
dev-backend:
	cd backend && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

dev-ml:
	cd ml-service && python main.py

dev-frontend:
	cd frontend && npm start

# Install dependencies
install-backend:
	cd backend && pip install -r requirements.txt

install-ml:
	cd ml-service && pip install -r requirements.txt

install-frontend:
	cd frontend && npm install

# Production deployment
deploy-k8s:
	@echo "Deploying to Kubernetes..."
	kubectl apply -f k8s/

# Monitor
monitor:
	@echo "Opening Grafana dashboard..."
	open http://localhost:3001
