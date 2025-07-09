from fastapi import APIRouter, Depends, HTTPException
from typing import List
from sqlalchemy.orm import Session
from sqlalchemy import or_
from core.database import get_db
from schema.goal import GoalCreate, GoalResponse
from models.models import Goal
from fastapi import Query

goal_router = APIRouter()

@goal_router.post("/create", response_model=List[GoalResponse])
def create_goals(goals: List[GoalCreate], db: Session = Depends(get_db)):
    db_goals = []
    for goal in goals:
        db_goal = Goal(
            goal=goal.goal,
            desc=goal.desc,
            yoga_needed=goal.yoga_needed,
            good_to_have_foods=goal.good_to_have_foods,
            avoid_foods=goal.avoid_foods
        )
        db.add(db_goal)
        db_goals.append(db_goal)
    db.commit()
    for goal in db_goals:
        db.refresh(goal)
    return db_goals



@goal_router.get("/all", response_model=List[GoalResponse])
def get_all_goals(
    db: Session = Depends(get_db),
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1),
):
    skip = (page - 1) * limit
    goals = db.query(Goal).offset(skip).limit(limit).all()
    return goals


@goal_router.get("/search/", response_model=List[GoalResponse])
def search_goals(query: str, db: Session = Depends(get_db)):
    results = db.query(Goal).filter(
        or_(
            Goal.goal.ilike(f"%{query}%"),
            Goal.desc.ilike(f"%{query}%")
        )
    ).all()
    return results

@goal_router.get("/{goal_id}", response_model=GoalResponse)
def get_goal(goal_id: int, db: Session = Depends(get_db)):
    goal = db.query(Goal).filter(Goal.id == goal_id).first()
    if not goal:
        raise HTTPException(status_code=404, detail="Goal not found")
    return goal


