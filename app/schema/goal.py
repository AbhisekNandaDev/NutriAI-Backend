from pydantic import BaseModel
from typing import Optional

class GoalCreate(BaseModel):
    goal: str
    desc: str
    yoga_needed: Optional[str] = None
    good_to_have_foods: Optional[str] = None
    avoid_foods: Optional[str] = None

class GoalResponse(BaseModel):
    id: int
    goal: str
    desc: str
    yoga_needed: Optional[str] = None
    good_to_have_foods: Optional[str] = None
    avoid_foods: Optional[str] = None

    class Config:
        orm_mode = True