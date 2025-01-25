# model_server/plugins/shared/validation.py

from typing import Dict, Any, List
from pydantic import BaseModel

class ModelInput(BaseModel):
    prompt: str
    max_length: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50

class ValidationUtils:
    @staticmethod
    def validate_chat_messages(messages: List[Dict[str, str]]) -> bool:
        required_keys = {"role", "content"}
        valid_roles = {"user", "assistant", "system"}
        
        for message in messages:
            if not all(key in message for key in required_keys):
                return False
            if message["role"] not in valid_roles:
                return False
        return True