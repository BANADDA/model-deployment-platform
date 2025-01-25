# core/monitoring.py
import psutil
import GPUtil
from typing import Dict, Any
from datetime import datetime

class ResourceMonitor:
   @staticmethod
   def get_system_metrics() -> Dict[str, Any]:
       metrics = {
           "cpu_percent": psutil.cpu_percent(),
           "memory": {
               "percent": psutil.virtual_memory().percent,
               "used": psutil.virtual_memory().used / (1024 * 1024 * 1024),  # GB
               "total": psutil.virtual_memory().total / (1024 * 1024 * 1024)  # GB
           },
           "disk": {
               "percent": psutil.disk_usage('/').percent,
               "used": psutil.disk_usage('/').used / (1024 * 1024 * 1024),  # GB 
               "total": psutil.disk_usage('/').total / (1024 * 1024 * 1024)  # GB
           },
           "network": {
               "bytes_sent": psutil.net_io_counters().bytes_sent,
               "bytes_recv": psutil.net_io_counters().bytes_recv
           }
       }

       # Safely try to get GPU metrics
       try:
           gpu_metrics = []
           for gpu in GPUtil.getGPUs():
               gpu_metrics.append({
                   "id": gpu.id,
                   "name": gpu.name,
                   "load": gpu.load * 100,  # Convert to percentage
                   "memory": {
                       "used": gpu.memoryUsed,
                       "total": gpu.memoryTotal,
                       "percent": (gpu.memoryUsed / gpu.memoryTotal) * 100
                   },
                   "temperature": gpu.temperature
               })
           metrics["gpus"] = gpu_metrics
       except Exception:
           metrics["gpus"] = []

       return metrics

   @staticmethod
   def check_deployment_health(deployment_id: str) -> Dict[str, Any]:
       try:
           process = psutil.Process()
           return {
               "status": "healthy",
               "last_check": datetime.now().isoformat(),
               "metrics": {
                   "cpu_percent": process.cpu_percent(),
                   "memory_percent": process.memory_percent(),
                   "threads": process.num_threads(),
                   "open_files": len(process.open_files()),
                   "connections": len(process.connections())
               }
           }
       except Exception:
           return {
               "status": "unhealthy",
               "last_check": datetime.now().isoformat(),
               "error": "Process not found"
           }