# api/routers/users.py

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
import uuid

router = APIRouter()

class User(BaseModel):
   id: str
   api_key: str
   active_deployments: list[str] = []

class UserCreate(BaseModel):
   email: str

users_db = {}

@router.post("/create")
async def create_user(user: UserCreate):
   user_id = str(uuid.uuid4())
   api_key = str(uuid.uuid4())
   users_db[user_id] = User(
       id=user_id,
       api_key=api_key,
       active_deployments=[]
   )
   return {"user_id": user_id, "api_key": api_key}

@router.get("/{user_id}/deployments")
async def get_user_deployments(user_id: str):
   if user_id not in users_db:
       raise HTTPException(status_code=404, detail="User not found")
   return users_db[user_id].active_deployments