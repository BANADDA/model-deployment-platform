# model_server/core/model_interface.py

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any

class ModelInterface(ABC):
    """Base interface for all model implementations."""
    
    @abstractmethod
    async def initialize(self, model_path: str, **kwargs) -> None:
        """Initialize model with given path and parameters."""
        pass
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt."""
        pass
        
    @abstractmethod
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Handle chat-style interactions."""
        pass
    
    @abstractmethod
    async def embeddings(self, text: str) -> List[float]:
        """Generate embeddings for input text."""
        pass
    
    @property
    @abstractmethod
    def model_info(self) -> Dict[str, Any]:
        """Return model metadata."""
        pass
    
    @abstractmethod
    async def unload(self) -> None:
        """Cleanup and unload model from memory."""
        pass

class ModelException(Exception):
    """Base exception for model-related errors."""
    pass