# scripts/deploy.sh
#!/bin/bash

# Build Docker images
docker-compose build

# Deploy to Kubernetes
kubectl apply -f kubernetes/api/
kubectl apply -f kubernetes/model_server/

# Wait for deployments
kubectl rollout status deployment/api-deployment
kubectl rollout status deployment/model-server

echo "Deployment complete!"