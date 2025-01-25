# model_server/plugins/shared/metrics.py

from dataclasses import dataclass
import time
from typing import Dict, Any
import psutil
import torch

@dataclass
class ModelMetrics:
   inference_time: float
   tokens_generated: int
   memory_used: float
   gpu_utilization: float

class MetricsCollector:
   @staticmethod
   def collect_inference_metrics(start_time: float, tokens: int) -> ModelMetrics:
       return ModelMetrics(
           inference_time=time.time() - start_time,
           tokens_generated=tokens,
           memory_used=psutil.Process().memory_info().rss / 1024 / 1024,
           gpu_utilization=torch.cuda.utilization() if torch.cuda.is_available() else 0
       )

   @staticmethod
   def log_metrics(metrics: ModelMetrics, model_name: str):
       # Implementation for logging metrics to monitoring system
       pass