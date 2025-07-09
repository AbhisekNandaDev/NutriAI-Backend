# routes/user_routes.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session, joinedload
from core.database import get_db
from models.models import User, Goal, Diseases
from schema.user import UserCreate, UserResponse, UserUpdate
from fastapi.security import OAuth2PasswordRequestForm
from core.security import get_password_hash, verify_password, create_access_token, get_current_user

user_router = APIRouter()


@user_router.post("/create_user")
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    email_lower = user.email.lower()
    existing_user = db.query(User).filter(User.email == email_lower).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed_password = get_password_hash(user.password)

    db_user = User(
        email=email_lower,
        name=user.name,
        password=hashed_password,
        age=user.age,
        gender=user.gender,
        height_ft=user.height_ft,
        height_in=user.height_in,
        weight_kg=user.weight_kg,
        weight_gm=user.weight_gm,
        preference=user.preference
    )

    if user.medical_conditions:
        diseases = db.query(Diseases).filter(Diseases.id.in_(user.medical_conditions)).all()
        if not diseases:
            raise HTTPException(status_code=400, detail="Invalid disease IDs")
        db_user.diseases = diseases

    if user.goal_id:
        goals = db.query(Goal).filter(Goal.id.in_(user.goal_id)).all()
        if not goals:
            raise HTTPException(status_code=400, detail="Invalid goal IDs")
        db_user.goals = goals

    db.add(db_user)
    db.commit()
    db.refresh(db_user)

    access_token = create_access_token(data={"sub": db_user.email})
    return {"access_token": access_token, "token_type": "bearer"}

@user_router.post("/login")
def login_user(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    email_lower = form_data.username.lower()
    user = db.query(User).filter(User.email == email_lower).first()
    if not user or not verify_password(form_data.password, user.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    access_token = create_access_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}

@user_router.get("/", response_model=UserResponse)
def get_user(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    user = (
      db.query(User)
        .options(joinedload(User.diseases), joinedload(User.goals))
        .get(current_user.id)
    )
    return user


def calculate_bmr(user: User):
    if user.gender == "male":
        bmr = (10 * user.weight_kg) + (6.25 * ((user.height_ft * 12) + user.height_in) * 2.54) - (5 * user.age) + 5
    elif user.gender == "female":
        bmr = (10 * user.weight_kg) + (6.25 * ((user.height_ft * 12) + user.height_in) * 2.54) - (5 * user.age) - 161
    else:
        return None
    return bmr

def estimate_daily_calories_for_loss(bmr: float, activity_level: str = "sedentary", weight_loss_rate_kg_per_week: float = 0.5):
    """
    Estimates daily calorie needs for weight loss based on BMR and activity level.
    This is a very simplified estimate.
    """
    if bmr is None:
        return None

    activity_multipliers = {
        "sedentary": 1.2,
        "lightly_active": 1.375,
        "moderately_active": 1.55,
        "very_active": 1.725,
        "extra_active": 1.9,
    }
    multiplier = activity_multipliers.get(activity_level.lower(), 1.2)  # Default to sedentary

    maintenance_calories = bmr * multiplier
    calorie_deficit = weight_loss_rate_kg_per_week * 7700 / 7  # Roughly 500 calories per 0.5 kg loss per week
    estimated_calories_for_loss = maintenance_calories - calorie_deficit

    return max(1200, estimated_calories_for_loss) # Ensure a minimum intake

@user_router.get("/reports")
def reports(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    user = (
        db.query(User)
        .options(joinedload(User.diseases), joinedload(User.goals))
        .get(current_user.id)
    )

    if not user:
        return {"error": "User not found"}

    bmr = calculate_bmr(user)
    estimated_calories_for_loss = estimate_daily_calories_for_loss(bmr)

    return {
        "bmr": f"{bmr:.2f}" if bmr else "Could not calculate BMR (missing gender or height).",
        "estimated_calories_for_loss": f"{estimated_calories_for_loss:.2f}" if estimated_calories_for_loss else "Could not estimate calorie needs."
    }


@user_router.put("/update_user", response_model=UserResponse)
def update_user(update_data: UserUpdate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == current_user.id).first()

    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    for field, value in update_data.dict(exclude_unset=True).items():
        if field == "medical_conditions" and value:
            diseases = db.query(Diseases).filter(Diseases.id.in_(value)).all()
            user.diseases = diseases
        elif field == "goal_id" and value:
            goals = db.query(Goal).filter(Goal.id.in_(value)).all()
            user.goals = goals
        else:
            setattr(user, field, value)

    db.commit()
    db.refresh(user)

    return user
