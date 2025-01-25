# core/models.py

from pydantic import BaseModel
from typing import Dict, Any, Optional
from enum import Enum
from datetime import datetime

class ModelType(str, Enum):
   CHAT = "chat"
   COMPLETION = "completion"
   EMBEDDING = "embedding"

class DeploymentStatus(str, Enum):
   PENDING = "pending"
   RUNNING = "running"
   FAILED = "failed"
   STOPPED = "stopped"

class DeploymentMetrics(BaseModel):
   requests_processed: int = 0
   avg_latency_ms: float = 0
   uptime_seconds: int = 0
   last_request: Optional[datetime] = None

class Deployment(BaseModel):
   id: str
   model_id: str
   status: DeploymentStatus
   created_at: datetime
   endpoint: str
   metrics: DeploymentMetrics
   config: Dict[str, Any]