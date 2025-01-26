# api/routers/models.py

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Dict, Any
from core.deployment import DeploymentManager
import os
from dotenv import load_dotenv

router = APIRouter(prefix="/v1")
load_dotenv()  # Load environment variables from .env file if present

# Retrieve the Hugging Face token from the environment variable
HUGGING_FACE_TOKEN = os.getenv("HF_ACCESS_TOKEN")

if not HUGGING_FACE_TOKEN:
    raise EnvironmentError("Hugging Face token not found. Please set the HF_ACCESS_TOKEN environment variable.")

# Initialize DeploymentManager with Hugging Face token
deployment_manager = DeploymentManager(hf_token=HUGGING_FACE_TOKEN)


class DeployModelRequest(BaseModel):
    model_id: str = Field(..., example="gpt2")
    machine_config: Dict[str, Any] = Field(default_factory=dict)
    endpoint_type: str = Field(default="chat", example="chat")


class PredictRequest(BaseModel):
    prompt: str = Field(..., example="Once upon a time in a land far, far away,")
    max_tokens: int = Field(default=50, example=50)


class PredictResponse(BaseModel):
    deployment_id: str
    model: str
    prompt: str
    response: str
    tokens: int


AVAILABLE_MODELS = {
   "llama2-70b": {
       "status": "available",
       "hardware_requirements": {"gpu": "A100", "vram": "160GB"}
   },
   "llama2-13b": {
       "status": "available", 
       "hardware_requirements": {"gpu": "A100", "vram": "80GB"}
   },
   "llama2-7b": {
       "status": "available",
       "hardware_requirements": {"gpu": "A100", "vram": "40GB"}
   },
   "deepseek-67b-chat": {
       "status": "available",
       "hardware_requirements": {"gpu": "A100", "vram": "80GB"}
   },
   "deepseek-7b-chat": {
       "status": "available",
       "hardware_requirements": {"gpu": "A100", "vram": "40GB"}  
   },
   "deepseek-coder-33b": {
       "status": "available",
       "hardware_requirements": {"gpu": "A100", "vram": "48GB"}
   },
   "deepseek-coder-6.7b": {
       "status": "available",
       "hardware_requirements": {"gpu": "RTX 4090", "vram": "24GB"}
   },
   "deepseek-coder-1.3b": {
       "status": "available",
       "hardware_requirements": {"gpu": "RTX 3080", "vram": "10GB"}
   },
   "gpt2": {
       "status": "available",
       "hardware_requirements": {"gpu": "RTX 2080", "vram": "8GB"}
   }
}


@router.get("/models", summary="List Available Models")
async def list_models():
    """
    Retrieve a list of all available models with their statuses and hardware requirements.
    """
    return AVAILABLE_MODELS


@router.post("/models/deploy", summary="Deploy a Model")
async def deploy_model(request: DeployModelRequest):
    """
    Deploy a specified model with optional machine configurations.

    Args:
        request (DeployModelRequest): The model deployment request containing model_id, machine_config, and endpoint_type.

    Returns:
        Dict[str, Any]: Details of the deployed model including deployment_id and endpoint URL.
    """
    if request.model_id not in AVAILABLE_MODELS:
        raise HTTPException(status_code=404, detail="Model not found")

    try:
        deployment = await deployment_manager.deploy(
            container_id=request.model_id,
            machine_config=request.machine_config,
            endpoint_type=request.endpoint_type
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "deployment_id": deployment.id,
        "endpoint": f"/v1/deployments/{deployment.id}/predict",
        "status": deployment.status
    }


@router.post("/deployments/{deployment_id}/predict", response_model=PredictResponse, summary="Generate Prediction")
async def predict(deployment_id: str, request: PredictRequest):
    """
    Generate a prediction using a specific deployment.

    Args:
        deployment_id (str): The unique ID of the deployment.
        request (PredictRequest): The prediction request containing prompt and max_tokens.

    Returns:
        PredictResponse: The prediction response including the generated text and token count.
    """
    deployment = await deployment_manager.get_deployment(deployment_id)
    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")

    response = await deployment.predict(request.dict())

    if "error" in response:
        raise HTTPException(status_code=500, detail=response["error"])

    return PredictResponse(**response)


@router.get("/models/{model_id}", summary="Get Model Details")
async def get_model(model_id: str):
    """
    Retrieve details of a specific model.

    Args:
        model_id (str): The identifier of the model.

    Returns:
        Dict[str, Any]: Details of the specified model.
    """
    if model_id not in AVAILABLE_MODELS:
        raise HTTPException(status_code=404, detail="Model not found")
    return AVAILABLE_MODELS[model_id]