# api/routers/deployments.py

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, Any
from core.deployment import DeploymentManager
from core.container import ContainerManager

router = APIRouter()
deployment_manager = DeploymentManager()
container_manager = ContainerManager()

class DeploymentRequest(BaseModel):
   model_name: str
   machine_config: Dict[str, Any]
   endpoint_type: str
   user_id: str

class DeploymentResponse(BaseModel):
   deployment_id: str
   endpoint: str
   token: str
   status: str

@router.post("/deploy", response_model=DeploymentResponse)
async def create_deployment(request: DeploymentRequest):
   try:
       # Create container for the model
       container_id = await container_manager.create_container(
           model_name=request.model_name,
           config=request.machine_config
       )
       
       # Deploy to specified machine
       deployment = await deployment_manager.deploy(
           container_id=container_id,
           machine_config=request.machine_config,
           endpoint_type=request.endpoint_type
       )
       
       return {
           "deployment_id": deployment.id,
           "endpoint": f"https://api.yourservice.com/v1/{deployment.id}",
           "token": deployment.access_token,
           "status": "running"
       }
   except Exception as e:
       raise HTTPException(status_code=500, detail=str(e))

@router.get("/{deployment_id}/status")
async def get_deployment_status(deployment_id: str):
   status = await deployment_manager.get_status(deployment_id)
   return {"status": status}

@router.delete("/{deployment_id}")
async def delete_deployment(deployment_id: str):
   await deployment_manager.delete(deployment_id)
   return {"status": "deleted"}