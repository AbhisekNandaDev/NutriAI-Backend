# At the bottom of models.py (or in a separate schema file)
from pydantic import BaseModel
from typing import Optional
from schema.user import UserResponse


class FoodCreate(BaseModel):
    food_name: str
    food_desc: str
    food_carbohydrate: float
    food_protein: float
    food_fat: float
    food_calories: float



class FoodLogCreate(BaseModel):
    food_name: str
    food_desc: str
    food_carbohydrate: float
    food_protein: float
    food_fat: float
    food_calories: float
    food_image: Optional[str] = None

class FoodLogResponse(BaseModel):
    id: int
    user_id: int
    food_name: str
    food_desc: str
    food_carbohydrate: float
    food_protein: float
    food_fat: float
    food_calories: float
    food_image: Optional[str]
    date: str

    class Config:
        orm_mode = True


class FoodResponse(BaseModel):
    id: int
    food_name: str
    food_desc: str
    food_carbohydrate: float
    food_protein: float
    food_fat: float
    food_calories: float

    class Config:
        orm_mode = True
        from_attributes = True  # Add this line to enable from_orm


class UserPreferanceBase(BaseModel):
    food_id: int
    user_id: int

    class Config:
        orm_mode = True

class UserPreferanceFoodResponse(BaseModel):
    id: int
    food: Optional[FoodResponse] = None
    user: Optional[UserResponse] = None

    class Config:
        orm_mode = True
        from_attributes = True  # Add this line to enable from_orm
        

# Import after class definitions to avoid circular imports
from schema.user import UserResponse
from schema.food import FoodResponse

# Update the references
FoodLogResponse.update_forward_refs()