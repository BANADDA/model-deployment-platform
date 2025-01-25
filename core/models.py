# core/models.py

from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from enum import Enum

class ModelType(str, Enum):
    CHAT = "chat"
    COMPLETION = "completion"
    EMBEDDING = "embedding"

class MachineConfig(BaseModel):
    gpu_type: str
    memory_gb: int
    region: str
    cpu_count: Optional[int] = None

class DeploymentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    FAILED = "failed"
    TERMINATED = "terminated"

class Deployment(BaseModel):
    id: str
    user_id: str
    model_name: str
    model_type: ModelType
    machine_config: MachineConfig
    status: DeploymentStatus
    endpoint: str
    created_at: str
    metrics: Optional[Dict[str, Any]] = None