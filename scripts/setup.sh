# scripts/setup.sh
#!/bin/bash

# Install dependencies
pip install -r requirements.txt

# Setup development environment
python -m venv venv
source venv/bin/activate

# Create necessary directories
mkdir -p models
mkdir -p logs

# Setup Docker network
docker network create model-network

echo "Setup complete!"