# api/main.py

from fastapi import FastAPI, HTTPException
from .routers import deployments, health, users
from .config import Settings

app = FastAPI(title="Model Deployment Platform")
settings = Settings()

app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(users.router, prefix="/users", tags=["users"])
app.include_router(deployments.router, prefix="/deployments", tags=["deployments"])

@app.on_event("startup")
async def startup_event():
    # Initialize services
    pass

@app.on_event("shutdown")
async def shutdown_event():
    # Cleanup services
    pass