#!/bin/bash

# Local Testing Script for Smart Traffic Management System

set -e  # Exit on error

echo "=================================="
echo "Smart Traffic Management System"
echo "Local Testing Script"
echo "=================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Test results
TESTS_PASSED=0
TESTS_FAILED=0

run_test() {
    TEST_NAME=$1
    TEST_COMMAND=$2

    echo -n "Running: $TEST_NAME... "

    if eval $TEST_COMMAND > /tmp/test_output.log 2>&1; then
        echo -e "${GREEN}✓ PASS${NC}"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        echo -e "${RED}✗ FAIL${NC}"
        cat /tmp/test_output.log
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
}

echo "1. Checking Docker..."
run_test "Docker installed" "docker --version"
run_test "Docker Compose installed" "docker-compose --version"

echo ""
echo "2. Starting services..."
echo "Building containers..."
docker-compose build

echo "Starting all services..."
docker-compose up -d

echo "Waiting for services to be ready..."
sleep 30

echo ""
echo "3. Health checks..."
run_test "Backend API health" "curl -f http://localhost:8000/health"
run_test "ML Service health" "curl -f http://localhost:8001/health"
run_test "Frontend accessible" "curl -f http://localhost:3000"
run_test "PostgreSQL connection" "docker-compose exec -T postgres pg_isready -U traffic_admin"
run_test "Redis connection" "docker-compose exec -T redis redis-cli ping"

echo ""
echo "4. API endpoint tests..."
run_test "Backend status endpoint" "curl -f http://localhost:8000/api/v1/status"
run_test "ML service info" "curl -f http://localhost:8001/"

echo ""
echo "5. Backend unit tests..."
cd backend
if [ -d "venv" ]; then
    source venv/bin/activate
    run_test "Backend pytest" "pytest tests/ -v --tb=short"
    deactivate
else
    echo "Virtual environment not found, skipping Python tests"
fi
cd ..

echo ""
echo "6. Database tests..."
run_test "Database migrations" "docker-compose exec -T backend alembic upgrade head"
run_test "PostGIS extension" "docker-compose exec -T postgres psql -U traffic_admin -d smart_traffic_db -c 'SELECT PostGIS_version();'"

echo ""
echo "7. Integration tests..."
run_test "Create traffic reading" "curl -f -X POST http://localhost:8000/api/v1/traffic/readings \
  -H 'Content-Type: application/json' \
  -d '{\"sensor_id\": 1, \"vehicle_count\": 50, \"average_speed\": 45.5}'"

echo ""
echo "8. Performance tests..."
run_test "API response time < 200ms" "curl -o /dev/null -s -w '%{time_total}' http://localhost:8000/health | awk '{if (\$1 < 0.2) exit 0; else exit 1}'"

echo ""
echo "9. Monitoring stack..."
run_test "Prometheus accessible" "curl -f http://localhost:9090/-/healthy"
run_test "Grafana accessible" "curl -f http://localhost:3001/api/health"

echo ""
echo "=================================="
echo "Test Results"
echo "=================================="
echo -e "${GREEN}Passed: $TESTS_PASSED${NC}"
echo -e "${RED}Failed: $TESTS_FAILED${NC}"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
    echo ""
    echo "Services are running:"
    echo "  - Frontend: http://localhost:3000"
    echo "  - Backend API: http://localhost:8000/docs"
    echo "  - ML Service: http://localhost:8001/docs"
    echo "  - Grafana: http://localhost:3001 (admin/admin)"
    echo ""
    echo "To view logs: make logs"
    echo "To stop services: make down"
    exit 0
else
    echo -e "${RED}✗ Some tests failed${NC}"
    echo "Check logs with: docker-compose logs"
    exit 1
fi
