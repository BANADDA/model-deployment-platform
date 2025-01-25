# api/config.py

from pydantic_settings import BaseSettings
from typing import Dict, Any

class Settings(BaseSettings):
    # API Configuration
    API_VERSION: str = "v1"
    DEBUG: bool = False
    
    # Docker Configuration
    DOCKER_HOST: str = "unix://var/run/docker.sock"
    DOCKER_API_VERSION: str = "1.41"
    
    # Kubernetes Configuration
    KUBERNETES_CONTEXT: str = "default"
    
    # Model Configuration
    MODEL_BASE_PATH: str = "/models"
    DEFAULT_MODEL_CONFIG: Dict[str, Any] = {
        "max_length": 2048,
        "temperature": 0.7
    }
    
    class Config:
        env_file = ".env"