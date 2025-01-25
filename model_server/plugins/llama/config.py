# model_server/plugins/llama/config.py

from pydantic import BaseModel
from typing import Dict, Any

class LlamaConfig(BaseModel):
   max_length: int = 2048
   temperature: float = 0.7
   top_p: float = 0.9
   top_k: int = 50
   repetition_penalty: float = 1.1
   
   model_configs: Dict[str, Any] = {
       "llama-3.1-70b-instruct": {
           "min_gpu_memory": "80GB",
           "recommended_gpu": "A100",
           "quantization": "8bit"
       },
       "llama-3.1-7b-instruct": {
           "min_gpu_memory": "16GB",
           "recommended_gpu": "RTX 4090",
           "quantization": "4bit"
       }
   }