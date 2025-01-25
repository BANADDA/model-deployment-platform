# core/container.py

import docker
from typing import Dict, Any

class ContainerManager:
   def __init__(self):
       self.client = docker.from_env()

   async def create_container(self, model_name: str, config: Dict[str, Any]) -> str:
       container = self.client.containers.run(
           image=f"model-server:{model_name}",
           environment={
               "MODEL_NAME": model_name,
               "GPU_CONFIG": str(config.get("gpu", "")),
               "MEMORY_LIMIT": str(config.get("memory", "8g"))
           },
           runtime="nvidia",
           detach=True,
           network="model-network"
       )
       return container.id

   async def stop_container(self, container_id: str):
       container = self.client.containers.get(container_id)
       container.stop()
       container.remove()