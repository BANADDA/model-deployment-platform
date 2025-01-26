# Model Deployment Platform
A scalable platform for deploying and managing ML models locally.

## Prerequisites
- Python 3.9+
- Docker & Docker Compose
- NVIDIA GPU with CUDA support

## Installation
```bash
# Clone repository
git clone https://github.com/BANADDA/model-deployment-platform
cd model-deployment-platform

# Setup environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or .\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your configurations:
# HF_ACCESS_TOKEN=your_huggingface_token

# Fix permissions & start services
sudo chmod 666 /var/run/docker.sock
sudo docker-compose -f docker/docker-compose.yml up --build -d
```

## Usage Examples

### Python Client
```python
from model_platform import ModelClient

# Initialize client
client = ModelClient(api_key="your_key")

# Deploy model
deployment = client.deploy(
    model="llama2-70b",
    machine={
        "gpu": "A100",
        "vram": "160GB"
    }
)

print(deployment)
# Output:
# {
#     "deployment_id": "dep_abc123",
#     "endpoint": "http://localhost:8000/v1/deployments/dep_abc123/predict",
#     "status": "running"
# }

# Make inference request
response = client.generate(
    deployment_id=deployment.id,
    prompt="What is quantum computing?"
)

print(response)
# Output:
# {
#     "deployment_id": "dep_abc123",
#     "model": "llama2-70b",
#     "prompt": "What is quantum computing?",
#     "response": "Quantum computing is a type of computation that...",
#     "tokens": 50
# }
```

### API Endpoints
```bash
# List models
curl http://localhost:8000/v1/models

# Response:
{
    "llama2-70b": {
        "status": "available",
        "hardware_requirements": {
            "gpu": "A100",
            "vram": "160GB"
        }
    },
    "llama2-7b": {
        "status": "available",
        "hardware_requirements": {
            "gpu": "A100",
            "vram": "40GB"
        }
    }
}

# Deploy model
curl -X POST http://localhost:8000/v1/models/deploy \
-H "Content-Type: application/json" \
-d '{
    "model_id": "llama2-70b",
    "machine_config": {
        "gpu": "A100",
        "vram": "160GB"
    }
}'

# Generate prediction
curl -X POST http://localhost:8000/v1/deployments/dep_abc123/predict \
-H "Content-Type: application/json" \
-d '{
    "prompt": "What is quantum computing?",
    "max_tokens": 50
}'
```

## Monitoring

View metrics:
```bash
# Health check
curl http://localhost:8000/health/
# Response: {"status": "ok"}

# System metrics
curl http://localhost:8000/health/metrics
# Response:
{
    "cpu_percent": 12.5,
    "memory": {
        "percent": 45.2,
        "used": 7.3,
        "total": 16.0
    },
    "disk": {
        "percent": 68.3,
        "used": 89.5,
        "total": 256.0
    }
}

# Container logs
sudo docker logs docker-api-1
sudo docker logs docker-model_server-1
```

## Supported Models & Requirements

| Model | Min GPU | VRAM |
|-------|---------|------|
| LLaMa2 70B | A100 | 160GB |
| LLaMa2 13B | A100 | 80GB |
| LLaMa2 7B | A100 | 40GB |
| DeepSeek 3B | RTX 3080 | 16GB |
| DeepSeek 1B | RTX 2080 | 8GB |
| GPT-2 | RTX 3080 | 16GB |