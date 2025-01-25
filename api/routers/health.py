# api/routers/health.py

from fastapi import APIRouter
from core.monitoring import ResourceMonitor

router = APIRouter()

@router.get("/")
async def health_check():
    return {"status": "ok"}

@router.get("/metrics")
async def get_metrics():
    return await ResourceMonitor.get_system_metrics()

@router.get("/deployment/{deployment_id}")
async def deployment_health(deployment_id: str):
    return await ResourceMonitor.check_deployment_health(deployment_id)