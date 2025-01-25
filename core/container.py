# core/container.py

from typing import Dict, Any
import uuid

class ContainerManager:
   def __init__(self):
       pass

   async def create_container(self, model_name: str, config: Dict[str, Any]) -> str:
       return f"model-{uuid.uuid4()}"

   async def stop_container(self, container_id: str):
       pass