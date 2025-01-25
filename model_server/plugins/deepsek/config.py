# model_server/plugins/deepsek/config.py

from pydantic import BaseModel

class DeepSekConfig(BaseModel):
    max_length: int = 4096
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50
    repetition_penalty: float = 1.1
    
    model_configs = {
        "deepsek-33b": {
            "min_gpu_memory": "48GB",
            "recommended_gpu": "A6000",
            "quantization": "8bit"
        }
    }