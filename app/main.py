from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.user_routes import user_router
from routes.diseases_routers import diseases_router
from routes.goal_routes import goal_router
from routes.food_routes import food_router
from routes.analytics_router import analytics_router
from routes.weight_router import weight_router
from routes.yoga_routes import yoga_router
from models.models import *
app = FastAPI(title="NutriAI API")

origins = [
    "http://localhost",
    "http://localhost:3000",  # Add your frontend URL (e.g., React dev server)
    "http://127.0.0.1:8000",  # Default FastAPI URL
    "*"  # Wildcard for testing (not recommended for production)
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # Must be False if using "*"
    allow_methods=["*"],
    allow_headers=["*"],
)

from core.database import Base, engine
Base.metadata.create_all(bind=engine)

# Include the user router with a prefix
app.include_router(user_router, prefix="/users", tags=["users"])
app.include_router(diseases_router, prefix="/diseases", tags=["diseases"])
app.include_router(goal_router, prefix="/goal", tags=["goal"])
app.include_router(food_router, prefix="/food", tags=["food"])
app.include_router(analytics_router, prefix="/analytics", tags=["analytics"])
app.include_router(yoga_router, prefix="/yoga", tags=["yoga"])
app.include_router(weight_router, prefix="/weight", tags=["weight"])

@app.get("/")
def read_root():
    return {"message": "Welcome to NutriAI API"}