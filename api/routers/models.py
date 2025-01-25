# api/routers/models.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from core.deployment import DeploymentManager

router = APIRouter(prefix="/v1")
deployment_manager = DeploymentManager()

class DeployModelRequest(BaseModel):
   model_id: str
   machine_config: Dict[str, Any]
   endpoint_type: str = "chat"

AVAILABLE_MODELS = {
   "llama-3.1-70b": {
       "status": "available",
       "hardware_requirements": {"gpu": "A100", "vram": "80GB"}
   },
   "gpt2": {
       "status": "available", 
       "hardware_requirements": {"gpu": "RTX 3080", "vram": "16GB"}
   }
}

@router.get("/models")
async def list_models():
   return AVAILABLE_MODELS

@router.post("/models/deploy")
async def deploy_model(request: DeployModelRequest):
   if request.model_id not in AVAILABLE_MODELS:
       raise HTTPException(status_code=404, detail="Model not found")
       
   deployment = await deployment_manager.deploy(
       container_id=request.model_id,
       machine_config=request.machine_config,
       endpoint_type=request.endpoint_type
   )
   
   return {
       "deployment_id": deployment.id,
       "endpoint": f"/v1/deployments/{deployment.id}/predict",
       "status": "running"
   }
   
@router.post("/deployments/{deployment_id}/predict")
async def predict(deployment_id: str, request: Dict[str, Any]):
    deployment = await deployment_manager.get_deployment(deployment_id)
    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")
    return await deployment.predict(request)

@router.get("/models/{model_id}")
async def get_model(model_id: str):
   if model_id not in AVAILABLE_MODELS:
       raise HTTPException(status_code=404, detail="Model not found")
   return AVAILABLE_MODELS[model_id]