# schema/user.py
from pydantic import BaseModel
from typing import Optional, List

class UserCreate(BaseModel):
    email: str
    name: str
    password: str
    age: Optional[int] = None
    gender: Optional[str] = None
    height_ft: Optional[int] = None
    height_in: Optional[int] = None
    weight_kg: Optional[int] = None
    weight_gm: Optional[int] = None
    medical_conditions: Optional[List[int]] = None  # Now list of disease IDs
    preference: Optional[str] = None
    goal_id: Optional[List[int]] = None             # Now list of goal IDs

class UserUpdate(BaseModel):
    name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    height_ft: Optional[int] = None
    height_in: Optional[int] = None
    weight_kg: Optional[int] = None
    weight_gm: Optional[int] = None
    preference: Optional[str] = None
    medical_conditions: Optional[List[int]] = None
    goal_id: Optional[List[int]] = None

class UserResponse(BaseModel):
    id: int
    email: str
    name: str
    age: Optional[int] = None
    gender: Optional[str] = None
    height_ft: Optional[int] = None
    height_in: Optional[int] = None
    weight_kg: Optional[int] = None
    weight_gm: Optional[int] = None
    preference: Optional[str] = None

    diseases: Optional[List["DiseasesResponse"]] = None  # <-- corrected to match model
    goals: Optional[List["GoalResponse"]] = None          # <-- corrected to match model

    class Config:
        orm_mode = True

# Avoid circular import
from schema.diseases import DiseasesResponse
from schema.goal import GoalResponse
