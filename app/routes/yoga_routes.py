# yoga_routes.py
from typing import List
from fastapi import APIRouter, Depends, HTTPException,Request
from sqlalchemy.orm import Session
from core.database import get_db  # Assuming this function gets the database session
from schema.yoga import YogaDataSchema, YogaPoseLogSchema, YogaDataCreate, YogaPoseLogCreate
from models.models import YogaData as YogaDataModel # Import your SQLAlchemy models
from models.models import YogaPoseLog as YogaPoseLogModel
from models.models import User #Import User model
from core.security import get_current_user  # Assuming this function gets the current user



yoga_router = APIRouter()

# Yoga Data Endpoints
@yoga_router.get("/data/", response_model=List[YogaDataSchema])
def get_yoga_data(db: Session = Depends(get_db)):
    """
    Retrieve all yoga data.
    """
    yoga_data = db.query(YogaDataModel).all()
    return yoga_data

@yoga_router.post("/data/", response_model=List[YogaDataSchema])
def create_yoga_data(yoga_data_list: List[YogaDataCreate], db: Session = Depends(get_db)):
    """
    Create multiple new yoga data entries.
    """
    # Check for duplicates *before* adding anything to the database
    yoga_names = [data.yoga_name for data in yoga_data_list]
    existing_names = db.query(YogaDataModel.yoga_name).filter(YogaDataModel.yoga_name.in_(yoga_names)).all()
    existing_names_set = {name[0] for name in existing_names} #faster lookup

    if existing_names_set:
        duplicates = [name for name in yoga_names if name in existing_names_set]
        raise HTTPException(status_code=400, detail=f"Yoga data with these names already exists: {duplicates}")

    new_yoga_data_objects = [YogaDataModel(**data.dict()) for data in yoga_data_list]
    db.add_all(new_yoga_data_objects)
    db.commit()

    # Refresh the objects so they contain any server-generated attributes like the primary key
    for obj in new_yoga_data_objects:
        db.refresh(obj)
    
    return new_yoga_data_objects

@yoga_router.get("/data/{yoga_id}", response_model=YogaDataSchema)
def get_yoga_data_by_id(yoga_id: int, db: Session = Depends(get_db)):
    """
    Retrieve yoga data by its ID.
    """
    yoga_data = db.query(YogaDataModel).filter(YogaDataModel.id == yoga_id).first()
    if not yoga_data:
        raise HTTPException(status_code=404, detail="Yoga data not found")
    return yoga_data

# Yoga Pose Log Endpoints
@yoga_router.get("/log/", response_model=List[YogaPoseLogSchema])
def get_yoga_pose_logs(db: Session = Depends(get_db),current_user: User = Depends(get_current_user)):
    """
    Retrieve all yoga pose logs.
    """
    yoga_pose_logs = db.query(YogaPoseLogModel).all()
    return yoga_pose_logs

@yoga_router.post("/log/", response_model=YogaPoseLogSchema)
def create_yoga_pose_log(yoga_pose_log: YogaPoseLogCreate, db: Session = Depends(get_db),current_user: User = Depends(get_current_user)):
    """
    Create a new yoga pose log.
    """
    #  basic validation (check IDs exist)
    user_id=current_user.id #changed User Model
    user_exists = db.query(User).filter(User.id == user_id).first() #changed User Model
    yoga_exists = db.query(YogaDataModel).filter(YogaDataModel.id == yoga_pose_log.yoga_id).first()
    if not user_exists:
        raise HTTPException(status_code=400, detail="User not found")
    if not yoga_exists:
        raise HTTPException(status_code=400, detail="Yoga data not found")

    new_yoga_pose_log = YogaPoseLogModel(**yoga_pose_log.dict())
    db.add(new_yoga_pose_log)
    db.commit()
    db.refresh(new_yoga_pose_log)
    return new_yoga_pose_log



