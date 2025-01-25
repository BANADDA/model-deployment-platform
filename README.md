# Model Deployment Platform

A scalable platform for deploying and managing ML models on polaris network.

## Prerequisites
- Python 3.9+
- Docker & Docker Compose
- NVIDIA GPU with CUDA support
- Kubernetes cluster access

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
# API_KEY=your_api_key
# KUBERNETES_CONTEXT=your-context
# MODEL_BASE_PATH=/path/to/models

# Deploy platform
chmod +x scripts/deploy.sh
./scripts/deploy.sh
```

## Usage Examples

### Python Client

```python
from model_platform import ModelClient

# Initialize client
client = ModelClient(api_key="your_key")

# Deploy model
deployment = client.deploy(
    model="llama-3.1-70b-instruct",
    machine={
        "gpu": "RTX 4090",
        "region": "us-east-1",
        "memory_gb": 24
    }
)

print(deployment)
# Output:
# {
#     "deployment_id": "dep_abc123",
#     "endpoint": "https://api.yourservice.com/v1/dep_abc123",
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
#     "text": "Quantum computing is...",
#     "tokens_generated": 156,
#     "inference_time": 0.8
# }
```

### API Endpoints

```bash
# List models
curl http://localhost:8000/v1/models

# Create deployment
curl -X POST http://localhost:8000/v1/deployments \
  -H "Authorization: Bearer your_api_key" \
  -d '{
    "model": "llama-3.1-70b-instruct",
    "machine": {
      "gpu": "RTX 4090",
      "region": "us-east-1"
    }
  }'

# Generate text
curl -X POST http://localhost:8000/v1/completions \
  -H "Authorization: Bearer your_api_key" \
  -d '{
    "deployment_id": "dep_abc123",
    "prompt": "What is quantum computing?"
  }'
```

## Monitoring

View deployment metrics:
```bash
kubectl get deployments
kubectl logs deployment/model-server
```

GPU utilization:
```bash
nvidia-smi
```

## Supported Models
- LLaMA 3.1 (70B, 7B)
- GPT-2 (Base, Large)
- DeepSek (33B)

## Resource Requirements
| Model | Min GPU | VRAM | Storage |
|-------|---------|------|---------|
| LLaMA 70B | A100 | 80GB | 150GB |
| LLaMA 7B | RTX 4090 | 16GB | 15GB |
| GPT-2 Large | RTX 3080 | 16GB | 6GB |
```