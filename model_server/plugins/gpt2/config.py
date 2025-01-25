# model_server/plugins/gpt2/config.py

from pydantic import BaseModel

class GPT2Config(BaseModel):
    max_length: int = 1024
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 40
    repetition_penalty: float = 1.0
    
    model_configs = {
        "gpt2": {
            "min_gpu_memory": "8GB",
            "recommended_gpu": "RTX 3080",
            "quantization": None
        },
        "gpt2-large": {
            "min_gpu_memory": "16GB",
            "recommended_gpu": "RTX 4090",
            "quantization": None
        }
    }