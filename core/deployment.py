# core/deployment.py

import uuid
import asyncio
import torch
from typing import Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, logging as transformers_logging
import logging

# Suppress warnings from transformers library for cleaner output
transformers_logging.set_verbosity_error()

# Initialize logger
logger = logging.getLogger("model_deployment_platform")


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
        logger.debug(f"Initializing ModelWrapper for model_id: {model_id}")
        self._load_model()

    def _load_model(self):
        """
        Load the appropriate model based on model_id
        """
        try:
            if self.model_id.lower() == "gpt2":
                logger.info("Loading GPT-2 model.")
                self.model = AutoModelForCausalLM.from_pretrained("gpt2")
                self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            elif self.model_id.lower().startswith("llama"):
                logger.info(f"Loading LLaMA model: {self.model_id}")
                model_repo = self._get_hf_repo_name(self.model_id)
                if self.hf_token is None:
                    logger.critical("Hugging Face token is required for LLaMA models but not provided.")
                    raise ValueError("Hugging Face token is required for LLaMA models.")
                
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
                logger.info(f"Loading DeepSeek model: {self.model_id}")
                model_repo = self._get_hf_repo_name(self.model_id)
                if self.hf_token is None:
                    logger.critical("Hugging Face token is required for DeepSeek models but not provided.")
                    raise ValueError("Hugging Face token is required for DeepSeek models.")
                
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
                logger.error(f"Unsupported model: {self.model_id}")
                raise ValueError(f"Unsupported model: {self.model_id}")
            logger.info(f"Model '{self.model_id}' loaded successfully.")
        except Exception as e:
            logger.exception(f"Model loading error for '{self.model_id}': {e}")
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
            "deepseek-67b-chat": "deepseek-ai/deepseek-llm-67b-chat",
            "deepseek-7b-chat": "deepseek-ai/deepseek-llm-7b-chat",
            "deepseek-coder-33b": "deepseek-ai/deepseek-coder-33b-instruct",
            "deepseek-coder-6.7b": "deepseek-ai/deepseek-coder-6.7b-instruct",
            "deepseek-coder-1.3b": "deepseek-ai/deepseek-coder-1.3b-instruct"
        }

        repo_name = model_repo_mapping.get(model_id.lower())
        if not repo_name:
            logger.error(f"No Hugging Face repository mapping found for model_id '{model_id}'.")
            raise ValueError(f"No Hugging Face repository mapping found for model_id '{model_id}'. Please update the mapping.")
        logger.debug(f"Model ID '{model_id}' mapped to repository '{repo_name}'.")
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
            dtype = torch.float16
        elif "13b" in model_repo.lower():
            dtype = torch.float16
        elif "3b" in model_repo.lower():
            dtype = torch.float16  # Use float16 for 3B models
        else:
            dtype = torch.float32  # Use float32 for smaller models like 1B
        logger.debug(f"Determined torch dtype '{dtype}' for repository '{model_repo}'.")
        return dtype

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
            logger.error("Model not properly initialized.")
            raise RuntimeError("Model not properly initialized.")

        logger.debug(f"Generating text with prompt: '{prompt}' and max_tokens: {max_tokens}")
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
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            logger.debug(f"Tokenized input: {inputs}")

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
            logger.debug(f"Model generated outputs: {outputs}")

            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.debug(f"Generated text: '{generated_text}'")
            return generated_text
        except Exception as e:
            logger.exception(f"Error during text generation: {e}")
            raise


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
        logger.debug(f"Initializing Deployment with ID: {id}, Model: {container_id}")
        try:
            self.model_wrapper = ModelWrapper(container_id, hf_token=hf_token)
            logger.info(f"Deployment '{self.id}' initialized and status set to 'running'.")
        except Exception as e:
            logger.exception(f"Failed to initialize Deployment '{self.id}': {e}")
            self.status = "failed"
            raise
        self.status = "running"

    async def predict(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate prediction for the given request

        Args:
            request (Dict[str, Any]): Prediction request details

        Returns:
            Dict[str, Any]: Prediction response
        """
        logger.info(f"Starting prediction for Deployment ID: {self.id}")
        # Update status to 'busy' before prediction
        self.status = "busy"
        logger.debug(f"Deployment '{self.id}' status updated to 'busy'.")

        prompt = request.get('prompt', '')
        max_tokens = request.get('max_tokens', 50)

        if not prompt:
            logger.warning(f"Empty prompt received for Deployment ID: {self.id}")
            self.status = "running"
            return {
                "error": "Prompt is required",
                "deployment_id": self.id
            }

        try:
            response = await self.model_wrapper.generate(prompt, max_tokens)
            tokens_used = len(self.model_wrapper.tokenizer.encode(response))
            logger.info(f"Prediction successful for Deployment ID: {self.id}. Tokens used: {tokens_used}")
            self.status = "running"
            return {
                "deployment_id": self.id,
                "model": self.container_id,
                "prompt": prompt,
                "response": response,
                "tokens": tokens_used
            }
        except Exception as e:
            logger.exception(f"Prediction failed for Deployment ID '{self.id}': {e}")
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
        logger.debug("DeploymentManager initialized.")

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

        logger.info(f"Deploying model '{container_id}' with Deployment ID: {deployment_id}")

        try:
            deployment = Deployment(deployment_id, access_token, container_id, hf_token=self.hf_token)
            self.deployments[deployment_id] = deployment
            logger.debug(f"Deployment '{deployment_id}' added to DeploymentManager.")
            return deployment
        except Exception as e:
            logger.exception(f"Failed to deploy model '{container_id}' with Deployment ID '{deployment_id}': {e}")
            raise

    async def get_deployment(self, deployment_id: str) -> Optional[Deployment]:
        """
        Retrieve a specific deployment

        Args:
            deployment_id (str): ID of the deployment to retrieve

        Returns:
            Optional[Deployment]: Deployment instance or None
        """
        logger.debug(f"Retrieving Deployment ID: {deployment_id}")
        deployment = self.deployments.get(deployment_id)
        if deployment:
            logger.debug(f"Deployment '{deployment_id}' found.")
        else:
            logger.warning(f"Deployment '{deployment_id}' not found.")
        return deployment

    async def get_status(self, deployment_id: str) -> str:
        """
        Get deployment status

        Args:
            deployment_id (str): ID of the deployment

        Returns:
            str: Deployment status
        """
        deployment = await self.get_deployment(deployment_id)
        status = deployment.status if deployment else "not found"
        logger.debug(f"Status for Deployment ID '{deployment_id}': {status}")
        return status

    async def delete(self, deployment_id: str):
        """
        Delete a deployment

        Args:
            deployment_id (str): ID of the deployment to delete
        """
        if deployment_id in self.deployments:
            logger.info(f"Deleting Deployment ID: {deployment_id}")
            del self.deployments[deployment_id]
            logger.debug(f"Deployment ID '{deployment_id}' deleted.")
        else:
            logger.warning(f"Tried to delete non-existent Deployment ID: {deployment_id}")
