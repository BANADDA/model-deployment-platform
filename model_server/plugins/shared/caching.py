# model_server/plugins/shared/caching.py

from typing import Dict, Any
import torch
from functools import lru_cache
import json

class ModelCache:
   def __init__(self, max_size: int = 5):
       self.max_size = max_size
       self._cache: Dict[str, Any] = {}

   @lru_cache(maxsize=100)
   def get_embeddings(self, text: str) -> List[float]:
       cache_key = hash(text)
       return self._cache.get(cache_key)

   def cache_embeddings(self, text: str, embeddings: List[float]):
       cache_key = hash(text)
       if len(self._cache) >= self.max_size:
           self._cache.pop(next(iter(self._cache)))
       self._cache[cache_key] = embeddings

   def clear(self):
       self._cache.clear()
       self.get_embeddings.cache_clear()