# core/monitoring.py

import psutil
import GPUtil
from typing import Dict, Any

class ResourceMonitor:
    @staticmethod
    def get_system_metrics() -> Dict[str, Any]:
        metrics = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent
        }
        
        gpu_metrics = []
        for gpu in GPUtil.getGPUs():
            gpu_metrics.append({
                "id": gpu.id,
                "name": gpu.name,
                "load": gpu.load,
                "memory_used": gpu.memoryUsed,
                "memory_total": gpu.memoryTotal
            })
        metrics["gpus"] = gpu_metrics
        
        return metrics

    @staticmethod
    async def check_deployment_health(deployment_id: str) -> Dict[str, Any]:
        # Implement deployment-specific health checks
        return {
            "status": "healthy",
            "last_check": "2024-01-25T12:00:00Z",
            "metrics": ResourceMonitor.get_system_metrics()
        }