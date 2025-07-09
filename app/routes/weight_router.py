from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from core.database import get_db
from models.models import User, Weight  # Import the Weight model
from core.security import get_current_user
from pydantic import BaseModel
from typing import List
from datetime import datetime

weight_router = APIRouter()

# Pydantic model for creating weight entries
class WeightCreate(BaseModel):
    weight: float
    date: str  # Consider using a date format like "YYYY-MM-DD"

# Pydantic model for representing weight entries in responses
class WeightResponse(BaseModel):
    weight: float
    date: str

    class Config:
        orm_mode = True  # Enable ORM mode for automatic conversion from database objects

@weight_router.post("/weight/", response_model=WeightResponse, summary="Add a new weight entry")
def create_weight_entry(
    weight_data: WeightCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Adds a new weight entry for the current user.
    """
    try:
        #date_obj = datetime.strptime(weight_data.date, "%Y-%m-%d") # commented out as the model uses string
        db_weight = Weight(
            user_id=current_user.id,
            weight=weight_data.weight,
            date=datetime.now().strftime("%Y-%m-%d"),
        )
        db.add(db_weight)
        db.commit()
        db.refresh(db_weight)
        return db_weight
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Could not create weight entry: {e}")

@weight_router.get("/weight/", response_model=List[WeightResponse], summary="Get all weight entries for the current user")
def get_weight_entries(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Retrieves all weight entries for the currently authenticated user.
    """
    weights = db.query(Weight).filter(Weight.user_id == current_user.id).all()
    return weights

@weight_router.get("/weight/{weight_id}", response_model=WeightResponse, summary="Get a specific weight entry by ID")
def get_weight_entry(
    weight_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Retrieves a specific weight entry by its ID, ensuring that it belongs to the current user.
    """
    weight = db.query(Weight).filter(Weight.id == weight_id, Weight.user_id == current_user.id).first()
    if not weight:
        raise HTTPException(status_code=404, detail="Weight entry not found")
    return weight
