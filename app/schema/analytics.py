# schema/analytics.py
from pydantic import BaseModel
from datetime import date
from typing import List

class DailyCalories(BaseModel):
    date: date
    calories: float

class MacroMetrics(BaseModel):
    carbohydrate: float
    protein: float
    fat: float

class MacroDistribution(BaseModel):
    carbohydrate: float  # percentage
    protein: float       # percentage
    fat: float           # percentage

class TopFood(BaseModel):
    food_name: str
    count: int

class AnalyticsRequest(BaseModel):
    start_date: date

class AnalyticsResponse(BaseModel):
    total_calories: float
    daily_calories: List[DailyCalories]
    avg_macros: MacroMetrics
    macro_distribution: MacroDistribution
    top_foods: List[TopFood]
