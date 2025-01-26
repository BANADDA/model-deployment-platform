# api/routers/logs.py

from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import logging
from pathlib import Path
from typing import List
import os

router = APIRouter(prefix="/v1", tags=["Logs"])

# Initialize logger
logger = logging.getLogger("model_deployment_platform")

# Security: Basic Auth (for demonstration purposes)
security = HTTPBasic()

# Retrieve credentials from environment variables for security
USERNAME = os.getenv("LOGS_API_USERNAME", "admin")
PASSWORD = os.getenv("LOGS_API_PASSWORD", "password123")


def get_current_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = credentials.username == USERNAME
    correct_password = credentials.password == PASSWORD
    if not (correct_username and correct_password):
        logger.warning("Unauthorized access attempt to logs.")
        raise HTTPException(status_code=401, detail="Unauthorized")
    return credentials.username


@router.get("/logs", summary="Retrieve Application Logs", response_model=List[str])
async def get_logs(username: str = Depends(get_current_credentials)):
    """
    Retrieve the latest application logs.

    Args:
        username (str): Authenticated username.

    Returns:
        List[str]: List of log entries.
    """
    log_file = Path("app.log")
    if not log_file.exists():
        logger.error("Log file not found.")
        raise HTTPException(status_code=404, detail="Log file not found.")

    try:
        with log_file.open("r") as f:
            # Fetch the last 100 lines
            lines = f.readlines()[-100:]
            logger.info(f"Logs retrieved by user: {username}")
            return [line.strip() for line in lines]
    except Exception as e:
        logger.exception(f"Failed to read log file: {e}")
        raise HTTPException(status_code=500, detail="Failed to read log file.")
