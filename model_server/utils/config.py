# model_server/utils/config.py

from typing import Dict, Any
import json
import os

class ConfigManager:
   def __init__(self):
       self.config = self._load_config()

   def _load_config(self) -> Dict[str, Any]:
       config_path = os.getenv("CONFIG_PATH", "config.json")
       if os.path.exists(config_path):
           with open(config_path, "r") as f:
               return json.load(f)
       return self._default_config()

   def _default_config(self) -> Dict[str, Any]:
       return {
           "logging": {
               "level": "INFO",
               "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
           },
           "model_defaults": {
               "max_batch_size": 32,
               "timeout_seconds": 30,
               "max_tokens": 2048
           },
           "resource_limits": {
               "max_concurrent_requests": 100,
               "memory_limit_gb": 32,
               "gpu_memory_limit_gb": 24
           }
       }

   def get(self, key: str, default: Any = None) -> Any:
       return self.config.get(key, default)

   def update(self, key: str, value: Any):
       self.config[key] = value
       self._save_config()

   def _save_config(self):
       with open("config.json", "w") as f:
           json.dump(self.config, f, indent=2)