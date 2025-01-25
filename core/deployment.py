# core/deployment.py

import uuid
import asyncio
import torch
from typing import Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, logging

# Suppress warnings from transformers library for cleaner output
logging.set_verbosity_error()


class ModelWrapper:
    def __init__(self, model_id: str, hf_token: Optional[str] = None):
        """
        Initialize model wrapper for different model types

        Args:
            model_id (str): Identifier for the model to load (e.g., "gpt2", "llama-3.1-70b", "deepseek-1b")
            hf_token (Optional[str]): Hugging Face access token (required for some models)
        """
        self.model_id = model_id
        self.hf_token = hf_token
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """
        Load the appropriate model based on model_id
        """
        try:
            if self.model_id.lower() == "gpt2":
                self.model = AutoModelForCausalLM.from_pretrained("gpt2")
                self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            elif self.model_id.lower().startswith("llama"):
                # Convert model_id to Hugging Face repository name
                model_repo = self._get_hf_repo_name(self.model_id)
                if self.hf_token is None:
                    raise ValueError("Hugging Face token is required for LLaMA models.")

                # Determine torch dtype based on model size
                torch_dtype = self._get_torch_dtype(model_repo)

                self.model = AutoModelForCausalLM.from_pretrained(
                    model_repo,
                    use_auth_token=self.hf_token,
                    torch_dtype=torch_dtype,
                    device_map="auto"  # Automatically map to available devices
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_repo,
                    use_auth_token=self.hf_token
                )
            elif self.model_id.lower().startswith("deepseek"):
                # Convert model_id to Hugging Face repository name
                model_repo = self._get_hf_repo_name(self.model_id)
                if self.hf_token is None:
                    raise ValueError("Hugging Face token is required for DeepSeek models.")

                # Determine torch dtype based on model size
                torch_dtype = self._get_torch_dtype(model_repo)

                self.model = AutoModelForCausalLM.from_pretrained(
                    model_repo,
                    use_auth_token=self.hf_token,
                    torch_dtype=torch_dtype,
                    device_map="auto"  # Automatically map to available devices
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_repo,
                    use_auth_token=self.hf_token
                )
            else:
                raise ValueError(f"Unsupported model: {self.model_id}")
        except Exception as e:
            print(f"Model loading error for '{self.model_id}': {e}")
            raise

    def _get_hf_repo_name(self, model_id: str) -> str:
        """
        Convert a model identifier to the corresponding Hugging Face repository name.

        Args:
            model_id (str): Model identifier (e.g., "llama-3.1-70b", "deepseek-1b")

        Returns:
            str: Hugging Face repository name
        """
        # Define a mapping from model identifiers to HF repo names
        model_repo_mapping = {
            "llama-3.1-70b": "meta-llama/Llama-3.1-70b-chat-hf",
            "llama2-7b": "meta-llama/Llama-2-7b-chat-hf",
            "llama2-13b": "meta-llama/Llama-2-13b-chat-hf",
            "llama2-70b": "meta-llama/Llama-2-70b-chat-hf",
            "deepseek-1b": "deepseek-ai/deepseek-1b",
            "deepseek-3b": "deepseek-ai/deepseek-3b",
        }

        repo_name = model_repo_mapping.get(model_id.lower())
        if not repo_name:
            raise ValueError(f"No Hugging Face repository mapping found for model_id '{model_id}'. Please update the mapping.")
        return repo_name

    def _get_torch_dtype(self, model_repo: str):
        """
        Determine the appropriate torch dtype based on the model size.

        Args:
            model_repo (str): Hugging Face repository name

        Returns:
            torch.dtype: The torch data type to use
        """
        # Example logic: Use float16 for larger models to save memory
        if "70b" in model_repo.lower():
            return torch.float16
        elif "13b" in model_repo.lower():
            return torch.float16
        elif "3b" in model_repo.lower():
            return torch.float16  # Use float16 for 3B models
        else:
            return torch.float32  # Use float32 for smaller models like 1B

    async def generate(self, prompt: str, max_tokens: int = 50) -> str:
        """
        Generate text using the loaded model

        Args:
            prompt (str): Input text prompt
            max_tokens (int): Maximum number of tokens to generate

        Returns:
            str: Generated text response
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not properly initialized")

        # Ensure generation happens in a separate thread to avoid blocking
        return await asyncio.to_thread(self._generate_sync, prompt, max_tokens)

    def _generate_sync(self, prompt: str, max_tokens: int) -> str:
        """
        Synchronous text generation method

        Args:
            prompt (str): Input text prompt
            max_tokens (int): Maximum number of tokens to generate

        Returns:
            str: Generated text response
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=inputs['input_ids'].shape[1] + max_tokens,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,  # Nucleus sampling
                eos_token_id=self.tokenizer.eos_token_id
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


class Deployment:
    def __init__(self, id: str, access_token: str, container_id: str, hf_token: Optional[str] = None):
        """
        Initialize a deployment instance

        Args:
            id (str): Unique deployment ID
            access_token (str): Access token for the deployment
            container_id (str): Model identifier (e.g., "gpt2", "llama-3.1-70b", "deepseek-1b")
            hf_token (Optional[str]): Hugging Face access token (required for some models)
        """
        self.id = id
        self.access_token = access_token
        self.container_id = container_id
        self.model_wrapper = ModelWrapper(container_id, hf_token=hf_token)
        self.status = "running"

    async def predict(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate prediction for the given request

        Args:
            request (Dict[str, Any]): Prediction request details

        Returns:
            Dict[str, Any]: Prediction response
        """
        # Update status to 'busy' before prediction
        self.status = "busy"

        prompt = request.get('prompt', '')
        max_tokens = request.get('max_tokens', 50)

        if not prompt:
            self.status = "running"
            return {
                "error": "Prompt is required",
                "deployment_id": self.id
            }

        try:
            response = await self.model_wrapper.generate(prompt, max_tokens)
            self.status = "running"
            return {
                "deployment_id": self.id,
                "model": self.container_id,
                "prompt": prompt,
                "response": response,
                "tokens": len(self.model_wrapper.tokenizer.encode(response))
            }
        except Exception as e:
            self.status = "failed"
            return {
                "error": str(e),
                "deployment_id": self.id
            }


class DeploymentManager:
    def __init__(self, hf_token: Optional[str] = None):
        """
        Initialize deployment manager

        Args:
            hf_token (Optional[str]): Hugging Face access token (required for some models)
        """
        self.deployments: Dict[str, Deployment] = {}
        self.hf_token = hf_token

    async def deploy(self, container_id: str, machine_config: Dict[str, Any], endpoint_type: str = "chat") -> Deployment:
        """
        Deploy a new model

        Args:
            container_id (str): Model identifier (e.g., "gpt2", "llama-3.1-70b", "deepseek-1b")
            machine_config (Dict[str, Any]): Machine configuration (unused in this example)
            endpoint_type (str): Type of endpoint (default: "chat")

        Returns:
            Deployment: Deployed model instance
        """
        deployment_id = str(uuid.uuid4())
        access_token = str(uuid.uuid4())

        deployment = Deployment(deployment_id, access_token, container_id, hf_token=self.hf_token)
        self.deployments[deployment_id] = deployment

        return deployment

    async def get_deployment(self, deployment_id: str) -> Optional[Deployment]:
        """
        Retrieve a specific deployment

        Args:
            deployment_id (str): ID of the deployment to retrieve

        Returns:
            Optional[Deployment]: Deployment instance or None
        """
        return self.deployments.get(deployment_id)

    async def get_status(self, deployment_id: str) -> str:
        """
        Get deployment status

        Args:
            deployment_id (str): ID of the deployment

        Returns:
            str: Deployment status
        """
        deployment = await self.get_deployment(deployment_id)
        return deployment.status if deployment else "not found"

    async def delete(self, deployment_id: str):
        """
        Delete a deployment

        Args:
            deployment_id (str): ID of the deployment to delete
        """
        if deployment_id in self.deployments:
            del self.deployments[deployment_id]