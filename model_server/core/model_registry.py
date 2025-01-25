# model_server/core/model_registry.py

from typing import Dict, Type
from .model_interface import ModelInterface

class ModelRegistry:
   """Registry for managing model plugins."""
   
   def __init__(self):
       self._models: Dict[str, Type[ModelInterface]] = {}
       self._instances: Dict[str, ModelInterface] = {}
   
   def register(self, name: str, model_class: Type[ModelInterface]) -> None:
       self._models[name] = model_class
   
   async def get_model(self, name: str, model_path: str = None) -> ModelInterface:
       if name not in self._instances:
           if name not in self._models:
               raise KeyError(f"Model {name} not registered")
           
           model = self._models[name]()
           if model_path:
               await model.initialize(model_path)
           self._instances[name] = model
           
       return self._instances[name]
   
   async def unload_model(self, name: str) -> None:
       if name in self._instances:
           await self._instances[name].unload()
           del self._instances[name]

   def list_models(self) -> Dict[str, Type[ModelInterface]]:
       return self._models.copy()