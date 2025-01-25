# model_server/plugins/shared/errors.py

class ModelError(Exception):
    def __init__(self, message: str, model_name: str, error_type: str):
        self.message = message
        self.model_name = model_name
        self.error_type = error_type
        super().__init__(self.message)

class ModelLoadError(ModelError):
    def __init__(self, model_name: str, details: str):
        super().__init__(f"Failed to load model: {details}", model_name, "load_error")

class InferenceError(ModelError):
    def __init__(self, model_name: str, details: str):
        super().__init__(f"Inference failed: {details}", model_name, "inference_error")