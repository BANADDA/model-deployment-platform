#!/bin/bash

# Create config directory and settings
mkdir -p config
cat > config/settings.json << EOF
{
 "api": {
   "port": 8000,
   "log_level": "info"
 },
 "models": {
   "base_path": "/app/models"
 }
}
EOF

# Build and run containers
docker-compose -f docker/docker-compose.yml build
docker-compose -f docker/docker-compose.yml up -d