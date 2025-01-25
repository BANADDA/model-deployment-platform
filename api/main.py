from fastapi import FastAPI
from .routers import deployments, health, users, models
from .config import Settings

app = FastAPI(title="Model Deployment Platform")
settings = Settings()

app.include_router(health.router)
app.include_router(users.router, prefix="/users", tags=["users"]) 
app.include_router(deployments.router, prefix="/deployments", tags=["deployments"])
app.include_router(models.router, tags=["models"])

@app.on_event("startup")
async def startup_event():
   pass

@app.on_event("shutdown") 
async def shutdown_event():
   pass