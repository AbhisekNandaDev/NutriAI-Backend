from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session, joinedload
from core.database import get_db
from collections import Counter
from models.models import FoodLog, User
from datetime import datetime
from core.security import get_current_user  # Import the dependency
from google.cloud import storage
from google.oauth2 import service_account
from sqlalchemy import func, desc # For database functions like count, sum, desc

from vertexai.preview.generative_models import GenerativeModel, Part
import vertexai
import json
from datetime import datetime, timedelta
from typing import Dict,List,DefaultDict
from collections import defaultdict
from schema.analytics import AnalyticsRequest, AnalyticsResponse, DailyCalories, MacroMetrics, MacroDistribution, TopFood
from schema.yoga import YogaPoseLogCreate, FrequentPoseAnalytic, UserYogaAnalyticsResponse
from fastapi import APIRouter, Depends, HTTPException,Query
from sqlalchemy.orm import Session, joinedload
from datetime import date
from dateutil import parser 
from models.models import User, FoodLog, YogaData, YogaPoseLog


analytics_router = APIRouter()

# Initialize Vertex AI (assumed to be set up globally in your project)
# vertexai.init(project="nutri-ai-453006", location="us-central1")

def calculate_bmr(user: User) -> float:
    """Calculate Basal Metabolic Rate (BMR) using Harris-Benedict equation."""
    height_cm = (user.height_ft * 30.48) + (user.height_in * 2.54)
    weight = user.weight_kg
    age = user.age
    gender = user.gender.lower() if user.gender else "unknown"

    if gender == "male":
        bmr = 88.362 + (13.397 * weight) + (4.799 * height_cm) - (5.677 * age)
    elif gender == "female":
        bmr = 447.593 + (9.247 * weight) + (3.098 * height_cm) - (4.330 * age)
    else:
        raise ValueError("Gender not specified or invalid")
    return bmr


@analytics_router.get("/daily_nutrition", response_model=List[Dict])
def get_daily_nutrition(
    start_date: str,
    end_date: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get daily nutritional data including calories, macronutrients, and surplus/deficit for a date range."""
    # Validate and parse dates
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=400, detail="Dates must be in YYYY-MM-DD format")

    if start > end:
        raise HTTPException(status_code=400, detail="start_date must be before or equal to end_date")

    # Fetch user data
    user = db.query(User).filter(User.id == current_user.id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Calculate BMR and daily caloric needs (sedentary lifestyle)
    try:
        bmr = calculate_bmr(user)
        daily_caloric_needs = bmr * 1.2  # Sedentary activity factor
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Fetch food logs within date range
    food_logs = db.query(FoodLog).filter(
        FoodLog.user_id == current_user.id,
        FoodLog.date >= start_date,
        FoodLog.date <= end_date
    ).all()

    # Aggregate nutrition data by date
    nutrition_by_date = defaultdict(lambda: {"calories": 0.0, "carbs": 0.0, "protein": 0.0, "fat": 0.0})
    for log in food_logs:
        # Ensure date is in string format (YYYY-MM-DD)
        log_date = log.date.split("T")[0] if "T" in log.date else log.date
        nutrition_by_date[log_date]["calories"] += log.food_calories
        nutrition_by_date[log_date]["carbs"] += log.food_carbohydrate
        nutrition_by_date[log_date]["protein"] += log.food_protein
        nutrition_by_date[log_date]["fat"] += log.food_fat

    # Prepare response with analytics
    result = []
    for date_str, nutrition in nutrition_by_date.items():
        total_calories = nutrition["calories"]
        surplus_deficit = total_calories - daily_caloric_needs
        carb_calories = nutrition["carbs"] * 4
        protein_calories = nutrition["protein"] * 4
        fat_calories = nutrition["fat"] * 9
        total_macro_calories = carb_calories + protein_calories + fat_calories

        if total_macro_calories > 0:
            carb_percentage = (carb_calories / total_macro_calories) * 100
            protein_percentage = (protein_calories / total_macro_calories) * 100
            fat_percentage = (fat_calories / total_macro_calories) * 100
        else:
            carb_percentage = protein_percentage = fat_percentage = 0.0

        result.append({
            "date": date_str,
            "total_calories": total_calories,
            "daily_caloric_needs": daily_caloric_needs,
            "surplus_deficit": surplus_deficit,
            "total_carbs": nutrition["carbs"],
            "total_protein": nutrition["protein"],
            "total_fat": nutrition["fat"],
            "carb_percentage": carb_percentage,
            "protein_percentage": protein_percentage,
            "fat_percentage": fat_percentage
        })

    # Sort results by date
    result.sort(key=lambda x: x["date"])
    return result

@analytics_router.get("/user_health_metrics", response_model=Dict)
def get_user_health_metrics(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user health metrics including BMI and BMR."""
    user = db.query(User).filter(User.id == current_user.id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Calculate BMI
    height_cm = (user.height_ft * 30.48) + (user.height_in * 2.54)
    height_m = height_cm / 100
    weight = user.weight_kg
    bmi = weight / (height_m ** 2) if height_m > 0 and weight > 0 else None

    # Calculate BMR
    try:
        bmr = calculate_bmr(user)
    except ValueError:
        bmr = None

    return {
        "bmi": bmi,
        "bmr": bmr,
    }



@analytics_router.get("/behavioral_insights", response_model=dict)
def get_behavioral_insights(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict:
    """Analyze eating patterns and provide motivational feedback."""
    user = db.query(User).options(joinedload(User.goal), joinedload(User.disease)).filter(User.id == current_user.id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Fetch food logs from the last 7 days
    seven_days_ago = datetime.utcnow() - timedelta(days=2)
    food_logs = db.query(FoodLog).filter(
        FoodLog.user_id == current_user.id,
        FoodLog.date >= seven_days_ago
    ).all()

    # Serialize food logs with timestamps
    serialized_logs = [
        {
            "food_name": log.food_name,
            "calories": log.food_calories,
            "carbohydrate": log.food_carbohydrate,
            "protein": log.food_protein,
            "fat": log.food_fat,
            "date": log.date
        } for log in food_logs
    ]

    # Calculate caloric needs
    try:
        bmr = calculate_bmr(user)
        caloric_needs = bmr * 1.2  # Sedentary activity factor
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Calculate average daily calories
    daily_calories = {}
    for log in serialized_logs:
        date = log["date"].split("T")[0]  # Extract date part
        daily_calories[date] = daily_calories.get(date, 0) + log["calories"]
    avg_daily_calories = sum(daily_calories.values()) / max(len(daily_calories), 1)

    # LLM prompt for behavioral insights
    prompt = f"""
    You are a behavioral nutritionist specializing in Indian cuisine. Analyze the following food logs for a user with:
    - Age: {user.age}, Gender: {user.gender}, Goal: {user.goal.goal if user.goal else 'general health'}
    - Medical Condition: {user.disease.disease_name if user.disease else 'none'}
    - Average Daily Caloric Intake: {avg_daily_calories:.1f} kcal
    - Recommended Caloric Needs: {caloric_needs:.1f} kcal
    - Food Logs: {json.dumps(serialized_logs, indent=2)}
    
    Identify eating patterns (e.g., late-night eating, irregular meal timing, low meal frequency) based on the 'date' field.
    Provide motivational feedback tailored to the user's goal and caloric intake.

    Instructions:
    - Highlight any patterns observed in the food logs.
    - Provide a motivational message based on the user's goal and progress.
    - Suggest behavioral changes to improve eating habits.
    - Return a JSON list with this format:

    Return a JSON object:"""+"""[{
      "patterns": ["Pattern 1 description", ...],
      "motivational_message": "Encouraging message based on goal and progress",
      "recommendations": ["Behavioral suggestion 1", ...]}]

    Do not include any extra explanation. Just return valid JSON only.
    """
    model = GenerativeModel("gemini-1.5-flash")
    try:
        response = model.generate_content(prompt)
        print(f"LLM Response: {response.text}")
        result = json.loads(response.text.strip().replace("```json", "").replace("```", ""))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse LLM response: {str(e)}")

    return {"user_id": user.id, "behavioral_insights": result}

@analytics_router.get("/nutritional_deficiencies", response_model=dict)
def get_nutritional_deficiencies(

    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict:
    """Identify potential nutritional deficiencies and suggest corrective foods."""
    user = db.query(User).options(joinedload(User.goal), joinedload(User.disease)).filter(User.id == current_user.id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Fetch food logs from the last 7 days
    seven_days_ago = datetime.utcnow() - timedelta(days=7)
    food_logs = db.query(FoodLog).filter(
        FoodLog.user_id == current_user.id,
        FoodLog.date >= seven_days_ago
    ).all()

    # Serialize food logs
    serialized_logs = [
        {
            "food_name": log.food_name,
            "carbohydrate": log.food_carbohydrate,
            "protein": log.food_protein,
            "fat": log.food_fat,
            "calories": log.food_calories
        } for log in food_logs
    ]

    # Calculate average macronutrient intake
    total_logs = max(len(food_logs), 1)
    avg_carbs = sum(log.food_carbohydrate for log in food_logs) / total_logs
    avg_protein = sum(log.food_protein for log in food_logs) / total_logs
    avg_fat = sum(log.food_fat for log in food_logs) / total_logs

    # LLM prompt for nutritional deficiencies
    prompt = f"""
    You are a nutritionist specializing in Indian cuisine. Analyze the following food logs for a user with:
    - Age: {user.age}, Gender: {user.gender}, Goal: {user.goal.goal if user.goal else 'general health'}
    - Medical Condition: {user.disease.disease_name if user.disease else 'none'}
    - Average Intake: Carbohydrates: {avg_carbs:.1f}g, Protein: {avg_protein:.1f}g, Fat: {avg_fat:.1f}g
    - Food Logs: {json.dumps(serialized_logs, indent=2)}
    
    Identify potential nutritional deficiencies (e.g., low protein, insufficient healthy fats) based on macronutrient intake and user profile.
    Suggest Indian foods to address deficiencies, considering {user.disease.food_good_to_have if user.disease else 'healthy foods'} and avoiding {user.disease.food_to_avoid if user.disease else 'none'}.

    Instructions:
    - Highlight any deficiencies observed in the food logs.
    - Suggest Indian foods that can help address these deficiencies
    - Return a JSON list with this format:

    Return a JSON object:"""+"""[{
      "deficiencies": ["Deficiency 1 description", ...],
      "recommendations": [
        {{"food_name": "Food", "reason": "Why it addresses the deficiency"}},
        ...
      ]
    }]
    Do not include any extra explanation. Just return valid JSON only.
    """
    model = GenerativeModel("gemini-1.5-flash")
    try:
        response = model.generate_content(prompt)
        print(f"LLM Response: {response.text}")
        result = json.loads(response.text.strip().replace("```json", "").replace("```", ""))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse LLM response: {str(e)}")

    return {"user_id": user.id, "nutritional_deficiencies": result}

@analytics_router.get("/nutrition_analytics", response_model=AnalyticsResponse)
def get_nutrition_analytics(
    start_date: date = Query(..., description="Start date in YYYY-MM-DD format"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    today = date.today()

    # 1) Fetch all logs for this user
    raw_logs = (
        db.query(FoodLog)
          .filter(FoodLog.user_id == current_user.id)
          .all()
    )

    # 2) Parse & filter by date range
    filtered = []
    for log in raw_logs:
        # parse log.date (string) into a datetime.date
        try:
            log_dt = parser.parse(log.date).date() if isinstance(log.date, str) else log.date
        except Exception:
            continue
        if start_date <= log_dt <= today:
            filtered.append((log_dt, log))

    if not filtered:
        raise HTTPException(status_code=404, detail="No food logs in given range")

    # 3) Aggregate totals
    daily_cal = defaultdict(float)
    total_carbs = total_prot = total_fat = total_cal = 0.0
    food_counter = Counter()

    for log_day, log in filtered:
        daily_cal[log_day] += log.food_calories
        total_cal    += log.food_calories
        total_carbs  += log.food_carbohydrate
        total_prot   += log.food_protein
        total_fat    += log.food_fat
        food_counter[log.food_name] += 1

    # 4) Build result lists
    days_count = (today - start_date).days + 1

    daily_list = [
        DailyCalories(date=d, calories=round(c, 2))
        for d, c in sorted(daily_cal.items())
    ]

    avg_macros = MacroMetrics(
        carbohydrate=round(total_carbs / days_count, 2),
        protein=     round(total_prot  / days_count, 2),
        fat=         round(total_fat   / days_count, 2),
    )

    total_macros = total_carbs + total_prot + total_fat
    macro_dist = MacroDistribution(
        carbohydrate=round((total_carbs / total_macros) * 100, 1),
        protein=     round((total_prot  / total_macros) * 100, 1),
        fat=         round((total_fat   / total_macros) * 100, 1),
    )

    top_foods = [
        TopFood(food_name=name, count=count)
        for name, count in food_counter.most_common(5)
    ]

    return AnalyticsResponse(
        total_calories=round(total_cal, 2),
        daily_calories=daily_list,
        avg_macros=avg_macros,
        macro_distribution=macro_dist,
        top_foods=top_foods
    )


@analytics_router.get("/analytics/yoga", response_model=UserYogaAnalyticsResponse) # Use the Pydantic model defined above
async def get_user_yoga_analytics(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db), # Your dependency to get a DB session
    top_n_poses: int = 5 # Optional query parameter to control how many frequent poses are returned
):
    """
    Retrieve comprehensive yoga analytics for a specific user, including
    total time, session counts, average duration, and most frequent poses.
    """
    # 0. Validate if user exists (optional, but good practice)
    user_id = current_user.id
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail=f"User with id {user_id} not found")

    # 1. Calculate Total Yoga Time (sum of yoga_time_in_sec for all logged poses)
    total_time_result = (
        db.query(func.sum(YogaData.yoga_time_in_sec))
        .join(YogaPoseLog, YogaData.id == YogaPoseLog.yoga_id)
        .filter(YogaPoseLog.user_id == user_id)
        .scalar() # Returns a single value or None
    )
    total_yoga_time_seconds = total_time_result if total_time_result is not None else 0

    # 2. Calculate Total Sessions Logged
    total_sessions_logged = (
        db.query(func.count(YogaPoseLog.id))
        .filter(YogaPoseLog.user_id == user_id)
        .scalar()
    )
    total_sessions_logged = total_sessions_logged if total_sessions_logged is not None else 0
    
    # 3. Calculate Average Duration per Session
    average_duration_per_session_seconds = (
        (total_yoga_time_seconds / total_sessions_logged)
        if total_sessions_logged > 0
        else None
    )

    # 4. Determine Most Frequent Poses
    # This query gets (yoga_id, count_of_that_yoga_id)
    frequent_poses_data = (
        db.query(
            YogaPoseLog.yoga_id,
            func.count(YogaPoseLog.yoga_id).label("count")
        )
        .filter(YogaPoseLog.user_id == user_id)
        .group_by(YogaPoseLog.yoga_id)
        .order_by(desc("count"))
        .limit(top_n_poses)
        .all()
    )

    most_frequent_poses_analytics: List[FrequentPoseAnalytic] = []
    if frequent_poses_data:
        # Extract yoga_ids to fetch their names and durations in a single query
        yoga_ids = [fp_data.yoga_id for fp_data in frequent_poses_data]
        
        yoga_details_map = {
            yd.id: yd for yd in db.query(YogaData).filter(YogaData.id.in_(yoga_ids)).all()
        }

        for fp_data in frequent_poses_data: # fp_data is a RowProxy with yoga_id and count
            yoga_detail = yoga_details_map.get(fp_data.yoga_id)
            if yoga_detail:
                # Calculate total duration for this specific pose
                duration_per_instance = yoga_detail.yoga_time_in_sec if yoga_detail.yoga_time_in_sec is not None else 0
                pose_total_duration = fp_data.count * duration_per_instance
                
                most_frequent_poses_analytics.append(
                    FrequentPoseAnalytic(
                        yoga_id=fp_data.yoga_id,
                        yoga_name=yoga_detail.yoga_name,
                        count=fp_data.count,
                        total_duration_seconds=pose_total_duration
                    )
                )

    return UserYogaAnalyticsResponse(
        user_id=user_id,
        total_yoga_time_seconds=total_yoga_time_seconds,
        total_sessions_logged=total_sessions_logged,
        average_duration_per_session_seconds=average_duration_per_session_seconds,
        most_frequent_poses=most_frequent_poses_analytics,
    )
