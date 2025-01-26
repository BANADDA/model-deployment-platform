from fastapi import FastAPI
from .routers import deployments, health, users, models, logs  # Added logs router
from .config import Settings
from .logging_config import setup_logging
import logging

# Initialize logging
logger = setup_logging()
logger.info("Initializing Model Deployment Platform")

app = FastAPI(title="Model Deployment Platform")
settings = Settings()

# Include your routers
app.include_router(health.router)
app.include_router(users.router, prefix="/users", tags=["users"]) 
app.include_router(deployments.router, prefix="/deployments", tags=["deployments"])
app.include_router(models.router, tags=["models"])
app.include_router(logs.router)  # Include logs router

@app.on_event("startup")
async def startup_event():
    logger.info("FastAPI application startup")

@app.on_event("shutdown") 
async def shutdown_event():
    logger.info("FastAPI application shutdown")
