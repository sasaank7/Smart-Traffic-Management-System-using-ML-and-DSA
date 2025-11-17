# Production Deployment Checklist

Complete this checklist before deploying to production.

## Pre-Deployment

### Security
- [ ] Change all default passwords in `.env`
- [ ] Generate strong SECRET_KEY and JWT_SECRET_KEY
- [ ] Update Kubernetes secrets with real values
- [ ] Enable SSL/TLS certificates (Let's Encrypt)
- [ ] Configure firewall rules
- [ ] Enable audit logging
- [ ] Set up API rate limiting
- [ ] Review CORS configuration
- [ ] Encrypt environment variables
- [ ] Set up VPN for database access

### Data & Models
- [ ] Download real traffic datasets (METR-LA, PEMS-BAY)
- [ ] Train LSTM model with production data
- [ ] Download YOLOv8 weights
- [ ] Test model accuracy (RMSE < 10 km/h)
- [ ] Set up model versioning
- [ ] Configure model auto-retraining schedule

### Infrastructure
- [ ] Provision Kubernetes cluster (min 3 nodes)
- [ ] Set up PostgreSQL with replication
- [ ] Configure Redis cluster mode
- [ ] Set up Kafka cluster (3 brokers)
- [ ] Configure load balancer
- [ ] Set up CDN for frontend assets
- [ ] Configure DNS records
- [ ] Set up backup storage (S3/GCS)

### Monitoring
- [ ] Configure Prometheus scraping
- [ ] Import Grafana dashboards
- [ ] Set up alert rules
- [ ] Configure notification channels (email, Slack)
- [ ] Enable distributed tracing
- [ ] Set up log aggregation (ELK/Loki)
- [ ] Configure uptime monitoring

### Backup & Recovery
- [ ] Set up automated database backups (daily)
- [ ] Test database restore procedure
- [ ] Document disaster recovery plan
- [ ] Set up off-site backup storage
- [ ] Define RTO (1 hour) and RPO (15 minutes)

### Testing
- [ ] All unit tests passing
- [ ] Integration tests passing
- [ ] Load testing completed (1000+ req/sec)
- [ ] Security audit completed
- [ ] Penetration testing completed
- [ ] Performance benchmarking completed

### Documentation
- [ ] API documentation updated
- [ ] Deployment guide reviewed
- [ ] Runbook created
- [ ] Architecture diagrams updated
- [ ] User guide completed

## Deployment Steps

### 1. Environment Setup
```bash
# Create production namespace
kubectl create namespace smart-traffic

# Create secrets
kubectl create secret generic postgres-secret \
  --from-literal=password='<STRONG_PASSWORD>' \
  -n smart-traffic

kubectl create secret generic app-secrets \
  --from-literal=secret-key='<SECRET_KEY>' \
  --from-literal=jwt-secret='<JWT_SECRET>' \
  -n smart-traffic
```

### 2. Database Setup
```bash
# Deploy PostgreSQL with PostGIS
kubectl apply -f k8s/postgres/

# Wait for ready
kubectl wait --for=condition=ready pod -l app=postgres \
  -n smart-traffic --timeout=300s

# Run migrations
kubectl exec -it postgres-0 -n smart-traffic -- \
  psql -U traffic_admin -d smart_traffic_db
```

### 3. Deploy Services
```bash
# Run deployment script
bash scripts/deploy/deploy_production.sh production

# Or deploy manually
kubectl apply -f k8s/backend/
kubectl apply -f k8s/ml-service/
kubectl apply -f k8s/frontend/
kubectl apply -f k8s/ingress/
```

### 4. Verify Deployment
```bash
# Check all pods running
kubectl get pods -n smart-traffic

# Check services
kubectl get svc -n smart-traffic

# Check ingress
kubectl get ingress -n smart-traffic

# Test endpoints
curl https://api.smarttraffic.com/health
```

### 5. Configure Monitoring
```bash
# Import Grafana dashboards
kubectl port-forward svc/grafana 3001:80 -n smart-traffic

# Access Grafana at http://localhost:3001
# Import dashboards from monitoring/grafana/dashboards/
```

## Post-Deployment

### Verification
- [ ] All pods in Running state
- [ ] Health checks passing
- [ ] API endpoints responding
- [ ] Frontend accessible
- [ ] Database migrations completed
- [ ] Metrics being collected
- [ ] Logs being aggregated
- [ ] Alerts configured

### Performance Tuning
- [ ] Enable autoscaling (HPA)
- [ ] Optimize database queries
- [ ] Configure Redis caching
- [ ] Enable CDN caching
- [ ] Tune connection pools
- [ ] Optimize Docker images

### Data Migration
- [ ] Import historical traffic data
- [ ] Load trained ML models
- [ ] Configure traffic sensors
- [ ] Set up data collection pipelines
- [ ] Initialize traffic graph

### User Access
- [ ] Create admin users
- [ ] Set up RBAC roles
- [ ] Configure SSO/LDAP (if needed)
- [ ] Create API keys for integrations
- [ ] Document user onboarding

## Ongoing Maintenance

### Daily
- [ ] Monitor system health
- [ ] Check error logs
- [ ] Review alerts
- [ ] Verify backups

### Weekly
- [ ] Review metrics and trends
- [ ] Update documentation
- [ ] Security updates
- [ ] Performance optimization

### Monthly
- [ ] Backup verification and restore test
- [ ] Security audit
- [ ] Capacity planning
- [ ] Model retraining
- [ ] Cost optimization review

## Rollback Plan

If deployment fails:

```bash
# Rollback backend
kubectl rollout undo deployment/backend -n smart-traffic

# Rollback ML service
kubectl rollout undo deployment/ml-service -n smart-traffic

# Rollback frontend
kubectl rollout undo deployment/frontend -n smart-traffic

# Restore database from backup
kubectl exec -it postgres-0 -n smart-traffic -- \
  psql -U traffic_admin -d smart_traffic_db < backup.sql
```

## Support Contacts

- **DevOps Lead**: devops@smarttraffic.com
- **ML Team**: ml@smarttraffic.com
- **Backend Team**: backend@smarttraffic.com
- **On-call**: +1-xxx-xxx-xxxx

## Emergency Procedures

### Service Down
1. Check pod status: `kubectl get pods -n smart-traffic`
2. View logs: `kubectl logs <pod-name> -n smart-traffic`
3. Restart if needed: `kubectl rollout restart deployment/<name> -n smart-traffic`

### Database Issues
1. Check PostgreSQL status
2. Review slow query log
3. Check connection pool
4. Scale replicas if needed

### High Traffic
1. Check autoscaler status
2. Manually scale if needed: `kubectl scale deployment/backend --replicas=10`
3. Enable caching
4. Review rate limiting

---

**Sign-off:**

- [ ] Deployment Lead: _________________ Date: _______
- [ ] Security Review: _________________ Date: _______
- [ ] ML Team Lead: ___________________ Date: _______
- [ ] Operations: _____________________ Date: _______

**Production Ready: YES / NO**
