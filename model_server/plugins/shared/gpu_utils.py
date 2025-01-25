# model_server/plugins/shared/gpu_utils.py

import torch
import GPUtil
from typing import Dict

class GPUManager:
    @staticmethod
    def get_available_gpu() -> int:
        gpus = GPUtil.getAvailable(order='memory', limit=1)
        return gpus[0] if gpus else -1

    @staticmethod
    def get_gpu_memory_info() -> Dict[str, int]:
        gpu = GPUtil.getGPUs()[0]
        return {
            "total": gpu.memoryTotal,
            "used": gpu.memoryUsed,
            "free": gpu.memoryFree
        }

    @staticmethod
    def optimize_for_gpu(model: torch.nn.Module) -> torch.nn.Module:
        if torch.cuda.is_available():
            model = model.half()  # Convert to FP16
        return model