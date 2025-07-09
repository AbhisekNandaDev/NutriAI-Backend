from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session, joinedload
from core.database import get_db
from schema.food import FoodCreate, FoodResponse, FoodLogCreate, FoodLogResponse,UserPreferanceBase, UserPreferanceFoodResponse
from collections import Counter
from fastapi import status
from models.models import Food, FoodLog, User,UserPreferanceFood
from datetime import datetime
from fastapi import File, UploadFile
import os
from core.security import get_current_user  # Import the dependency
from google.cloud import storage
from google.oauth2 import service_account
from vertexai.preview.generative_models import GenerativeModel, Part
import vertexai
import json
from datetime import datetime, timedelta
from fastapi import Query
from typing import List

food_router = APIRouter()

# Load credentials from environment variables
credentials_path = r"routes\nutri-ai-453006-857e0a3faa1b.json"
project_id = "nutri-ai-453006"
bucket_name = "nutriai_food"

# Initialize Vertex AI with your GCP project and region
vertexai.init(project=project_id, location="us-central1")

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

if not all([credentials_path, project_id, bucket_name]):
    raise ValueError(
        "Please set the environment variables: GOOGLE_APPLICATION_CREDENTIALS, GCP_PROJECT_ID, and GCP_BUCKET_NAME"
    )

# Create a credentials object
credentials = service_account.Credentials.from_service_account_file(credentials_path)

# Create a storage client
client = storage.Client(credentials=credentials, project=project_id)
bucket = client.bucket(bucket_name)


@food_router.post("/create", response_model=dict)
def create_food(food: list[FoodCreate], db: Session = Depends(get_db)):
    for item in food:
        try:
            db_food = Food(
                food_name=item.food_name,
                food_desc=item.food_desc,
                food_carbohydrate=item.food_carbohydrate,
                food_protein=item.food_protein,
                food_fat=item.food_fat,
                food_calories=item.food_calories
            )
            db.add(db_food)
            db.commit()
            db.refresh(db_food)
        except Exception as e:  
            pass
    
    return {"message": "Food items created successfully"}


@food_router.get("/all", response_model=List[FoodResponse])
def get_all_food(
    db: Session = Depends(get_db),
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1),
):
    skip = (page - 1) * limit
    foods = db.query(Food).offset(skip).limit(limit).all()
    return foods


@food_router.post("/upload", response_model=dict)
def upload_image(file: UploadFile = File(...),current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    print(current_user.id)

    try:
        # Upload to GCS
        blob = bucket.blob(file.filename)
        blob.upload_from_file(file.file)
        image_url = f"https://storage.googleapis.com/{bucket_name}/{file.filename}"

        # Load Gemini Pro Vision
        model = GenerativeModel("gemini-2.0-flash-lite-001")

        # Get the image from URL
        image = Part.from_uri(image_url, mime_type="image/jpeg")

        # Prompt Gemini to identify the food item
        prompt="""
            You are a nutrition expert.

            Look at the image of indian food and identify all distinct food items or dishes present. For each item, return a JSON object in this format:

            [
            {
                "food_name": "name of the food item for example 'rice'",
                "food_desc": "what is this food item like 'rice is a grain'",
                "food_carbohydrate": number (grams),
                "food_protein": number (grams),
                "food_fat": number (grams),
                "food_calories": number (kcal)
            }
            ]

            Do not include any extra explanation. Just return valid JSON only.
        """
        response = model.generate_content(
            [image, prompt]
        )

        prediction = json.loads(response.text.replace('```json', '').replace('```', ''))

        # Store each item into DB
        for item in prediction:
            food_log = FoodLog(
                user_id=current_user.id,
                food_image=image_url,
                food_name=item["food_name"],
                food_desc=item["food_desc"],
                food_carbohydrate=item["food_carbohydrate"],
                food_protein=item["food_protein"],
                food_fat=item["food_fat"],
                food_calories=item["food_calories"],
                date=str(datetime.now())  # or use a better format if needed
            )
            db.add(food_log)

        db.commit()
        db.refresh(food_log)

        return {
            "message": "File uploaded and analyzed successfully",
            "file_url": image_url,
            "prediction": prediction
        }

    except Exception as e:
        return {"error": f"Failed to upload or analyze image: {e}"}

@food_router.post("/predict", response_model=List[dict])
def predict_image(file: UploadFile = File(...),current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    print(current_user.id)

    try:
        # Upload to GCS
        blob = bucket.blob(file.filename)
        blob.upload_from_file(file.file)
        image_url = f"https://storage.googleapis.com/{bucket_name}/{file.filename}"

        # Load Gemini Pro Vision
        model = GenerativeModel("gemini-2.0-flash-lite-001")

        # Get the image from URL
        image = Part.from_uri(image_url, mime_type="image/jpeg")

        # Prompt Gemini to identify the food item
        prompt="""
            You are a nutrition expert.

            Look at the image of indian food and identify all distinct food items or dishes present. For each item, return a JSON object in this format:

            [
            {
                "food_name": "name of the food item for example 'rice'",
                "food_desc": "what is this food item like 'rice is a grain'",
                "food_carbohydrate": number (grams),
                "food_protein": number (grams),
                "food_fat": number (grams),
                "food_calories": number (kcal)
            }
            ]

            Do not include any extra explanation. Just return valid JSON only.
        """
        response = model.generate_content(
            [image, prompt]
        )

        prediction = json.loads(response.text.replace('```json', '').replace('```', ''))
        data=[]
        for x in prediction:
            x["file_url"] = image_url
            data.append(x)
        
        
        return data
    except Exception as e:
        return {"error": f"Failed to upload or analyze image: {e}"}


@food_router.post("/add_to_log", response_model=dict)
async def add_food_log(log_data: List[dict], current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Stores the food log data into the database."""
    try:
        for item in log_data:
            food_log = FoodLog(
                user_id=current_user.id,
                food_image=item.get("file_url"),
                food_name=item.get("food_name"),
                food_desc=item.get("food_desc"),
                food_carbohydrate=item.get("food_carbohydrate"),
                food_protein=item.get("food_protein"),
                food_fat=item.get("food_fat"),
                food_calories=item.get("food_calories"),
                date=str(datetime.now())
            )
            db.add(food_log)

        db.commit()
        return {"message": "Food log added successfully"}

    except Exception as e:
        db.rollback()
        return {"error": f"Failed to add food log: {e}"}



@food_router.get("/preferred_food", response_model=list)
def get_preferred_food_items(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1),
):
    user = db.query(User).filter(User.id == current_user.id).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    skip = (page - 1) * limit

    # Fetch preferred food records for the current user, joining the UserPreferanceFood table
    preferred_foods = (
        db.query(Food)
        .join(UserPreferanceFood, UserPreferanceFood.food_id == Food.id)
        .filter(UserPreferanceFood.user_id == current_user.id)
        .offset(skip)
        .limit(limit)
        .all()
    )

    # Convert SQLAlchemy objects to dicts (no Pydantic validation)
    preferred_foods_response = [
        {
            "id": food.id,
            "food_name": food.food_name,
            "food_desc": food.food_desc,
            "food_carbohydrate": food.food_carbohydrate,
            "food_protein": food.food_protein,
            "food_fat": food.food_fat,
            "food_calories": food.food_calories
        }
        for food in preferred_foods
    ]


    return preferred_foods_response

@food_router.get("/search", response_model=list[FoodResponse])
def search_food(query: str, db: Session = Depends(get_db)):
    results = db.query(Food).filter(
        Food.food_name.ilike(f"%{query}%") | Food.food_desc.ilike(f"%{query}%")
    ).all()
    return results

@food_router.get("/{food_id}", response_model=FoodResponse)
def get_food(food_id: int, db: Session = Depends(get_db)):
    food = db.query(Food).filter(Food.id == food_id).first()
    if food is None:
        raise HTTPException(status_code=404, detail="Food not found")
    return food

@food_router.post("/log", response_model=FoodLogResponse)
def create_food_log(
    food_log: FoodLogCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    db_food_log = FoodLog(
        user_id=current_user.id,
        food_name=food_log.food_name,
        food_desc=food_log.food_desc,
        food_carbohydrate=food_log.food_carbohydrate,
        food_protein=food_log.food_protein,
        food_fat=food_log.food_fat,
        food_calories=food_log.food_calories,
        food_image=food_log.food_image,
        date=datetime.utcnow().isoformat()
    )
    db.add(db_food_log)
    db.commit()
    db.refresh(db_food_log)
    return db_food_log


@food_router.get("/log/{log_id}", response_model=FoodLogResponse)
def get_food_log(log_id: int, db: Session = Depends(get_db)):
    food_log = db.query(FoodLog).filter(FoodLog.id == log_id).first()
    if food_log is None:
        raise HTTPException(status_code=404, detail="Food log not found")
    return food_log



@food_router.get("/logs/user", response_model=List[FoodLogResponse])
def get_user_food_logs(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1),
):
    skip = (page - 1) * limit
    food_logs = (
        db.query(FoodLog)
        .filter(FoodLog.user_id == current_user.id)
        .offset(skip)
        .limit(limit)
        .all()
    )
    return food_logs





@food_router.post("/preferred_food_items", response_model=dict)
def add_preferred_food_items(preferred_foods: list[int], current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == current_user.id).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    
    # Validate food items
    valid_food_items = db.query(Food).filter(Food.id.in_(preferred_foods)).all()
    valid_food_ids = {food.id for food in valid_food_items}
    
    # Filter out invalid food items
    filtered_preferred_foods = [food_id for food_id in preferred_foods if food_id in valid_food_ids]
    
    # Clear existing preferences
    db.query(UserPreferanceFood).filter(UserPreferanceFood.user_id == current_user.id).delete()
    
    # Add new preferences
    for food_id in filtered_preferred_foods:
        user_preference = UserPreferanceFood(user_id=current_user.id, food_id=food_id)
        db.add(user_preference)
    
    db.commit()
    db.refresh(user)
    return {"message": "Preferred food items updated successfully", "preferred_foods": filtered_preferred_foods}


@food_router.get("/food/recommend")
def get_recommendation(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # 1) Fetch user
    user = db.query(User).get(current_user.id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    # 2) Serialize user info
    user_info = {
        "age": user.age,
        "gender": user.gender,
        "goal": ", ".join([goal.goal for goal in user.goals]) if user.goals else None,
        "medical_conditions": ", ".join([disease.disease_name for disease in user.diseases]) if user.diseases else None,
        "preference": user.preference

    }

    # 3) Fetch user's diseases (medical conditions)
    diseases = [disease.disease_name for disease in user.diseases]

    # 4) Fetch user's preferred foods
    preferred_foods = [food.food_name for food in user.user_preferance_food]

    # 5) Decide which meals to recommend based on current time (UTC)
    now = datetime.utcnow().time()
    if now < datetime.strptime("09:00:00", "%H:%M:%S").time():
        meals = ["breakfast", "lunch", "dinner"]
    elif now < datetime.strptime("15:00:00", "%H:%M:%S").time():
        meals = ["lunch", "snacks", "dinner"]
    elif now < datetime.strptime("18:00:00", "%H:%M:%S").time():
        meals = ["snacks", "dinner"]
    else:
        meals = ["dinner"]

    # 6) Pull last 3 days of logs
    three_days_ago = datetime.utcnow() - timedelta(days=3)
    logs = (
        db.query(FoodLog)
        .filter(FoodLog.user_id == current_user.id, FoodLog.date >= three_days_ago)
        .all()
    )

    # 7) Sum today's calories
    today = datetime.utcnow().date()
    total_calories_today = 0
    for log in logs:
        dt = log.date
        if isinstance(dt, str):
            dt = datetime.fromisoformat(dt)
        if dt.date() == today:
            total_calories_today += log.food_calories

    # 8) Serialize logs for the prompt
    serialized_logs = [
        {
            "food_name": l.food_name,
            "carbohydrate": l.food_carbohydrate,
            "protein": l.food_protein,
            "fat": l.food_fat,
            "calories": l.food_calories,
            "date": str(l.date),
        }
        for l in logs
    ]

    fooditems=[
  {
    "food_name": "Chole Bhature",
    "food_desc": "Spicy chickpea curry served with deep-fried bread.",
    "food_carbohydrate": 52.9,
    "food_protein": 11.8,
    "food_fat": 25.9,
    "food_calories": 502,
    "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/91/Chole_Bhature_At_Local_Street.jpg/960px-Chole_Bhature_At_Local_Street.jpg"
  },
  {
    "food_name": "Aloo Paratha",
    "food_desc": "Wheat flatbread stuffed with spiced mashed potatoes.",
    "food_carbohydrate": 45,
    "food_protein": 7,
    "food_fat": 15,
    "food_calories": 350,
    "url": "https://static.toiimg.com/photo/53109843.cms"
  },
  {
    "food_name": "Biryani",
    "food_desc": "Fragrant rice dish cooked with spices and meat or vegetables.",
    "food_carbohydrate": 60,
    "food_protein": 20,
    "food_fat": 15,
    "food_calories": 500,
    "url": "hhttps://www.cubesnjuliennes.com/wp-content/uploads/2020/07/Chicken-Biryani-Recipe.jpg"
  },
  {
    "food_name": "Dal Makhani",
    "food_desc": "Creamy lentil dish made with black lentils, butter, and cream.",
    "food_carbohydrate": 30,
    "food_protein": 12,
    "food_fat": 20,
    "food_calories": 400,
    "url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSrqlmuM6fhCcLstM0zaHv2AFs-cusaSe5lUw&s"
  },
  {
    "food_name": "Butter Chicken",
    "food_desc": "Tandoori chicken cooked in a smooth, buttery, and creamy tomato-based gravy.",
    "food_carbohydrate": 10,
    "food_protein": 25,
    "food_fat": 20,
    "food_calories": 350,
    "url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQTY53Zww8_j72B0sOb6m4o4fOMO0eDtsvvwQ&s"
  },
  {
    "food_name": "Masala Dosa",
    "food_desc": "Crispy rice crepe filled with spiced mashed potatoes.",
    "food_carbohydrate": 50,
    "food_protein": 10,
    "food_fat": 15,
    "food_calories": 400,
    "url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTyLIFrcWSOk1C3cgNvs-Lcyaq7ZmYvmevfzQ&s"
  },
  {
    "food_name": "Paneer Tikka",
    "food_desc": "Marinated paneer cubes grilled to perfection.",
    "food_carbohydrate": 5,
    "food_protein": 15,
    "food_fat": 20,
    "food_calories": 300,
    "url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRkTmoN6HX1YlataD_t0nUSL1Re6-zhsCPikQ&s"
  },
  {
    "food_name": "Samosa",
    "food_desc": "Deep-fried pastry filled with spiced potatoes and peas.",
    "food_carbohydrate": 30,
    "food_protein": 5,
    "food_fat": 20,
    "food_calories": 300,
    "url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRxnVgCV-6sbWMQ-OaALcEztviIakZJgzo5Jg&s"
  },
  {
    "food_name": "Rajma Chawal",
    "food_desc": "Red kidney bean curry served with steamed rice.",
    "food_carbohydrate": 60,
    "food_protein": 15,
    "food_fat": 10,
    "food_calories": 450,
    "url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSrY-YDbNH_JLFnMOh26A1l7WKFgORHz6C3uQ&s"
  },
  {
    "food_name": "Pav Bhaji",
    "food_desc": "Spiced vegetable mash served with buttered bread rolls.",
    "food_carbohydrate": 50,
    "food_protein": 10,
    "food_fat": 20,
    "food_calories": 450,
    "url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQzJVYMvK9RwzZ5LOfGG5wjAZU1r-664As2hA&s"
  },
  {
    "food_name": "Chhena Poda",
    "food_desc": "A caramelized dessert made from roasted cottage cheese, sugar, and semolina.",
    "food_carbohydrate": 40,
    "food_protein": 10,
    "food_fat": 15,
    "food_calories": 350,
    "url": "https://www.bigbasket.com/media/uploads/recipe/w-l/4570_2_1.jpg"
  },
  {
    "food_name": "Pakhala Bhata",
    "food_desc": "Fermented rice soaked in water, served with fried or mashed vegetables.",
    "food_carbohydrate": 35,
    "food_protein": 5,
    "food_fat": 5,
    "food_calories": 200,
    "url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS3Rk2GMl0ra5KR1nJJha-2eMsQ1L3Oip0Luw&s"
  }
]
    fooditems_json_string = json.dumps(fooditems, indent=2)

    # 9) Most‐common foods
    freq = Counter([l.food_name for l in logs]).most_common(3)
    preferred = [name for name, _ in freq]

    # 10) Build a dynamic prompt
    meal_list_str = ", ".join(meals)

    prompt = f"""
    You are a smart nutritionist. Your task is to recommend appropriate meals for the user based on the current time, their profile, and a specific list of available food items.
    You MUST choose food items ONLY from the 'Available Food Items' list provided below.
    For each recommended food item, you MUST use its exact 'food_name', its 'food_calories' value for the 'cal' field, and its 'url' value directly from the 'Available Food Items' list.

    Available Food Items:
    {fooditems_json_string}

    User Info:
    - Age: {user_info['age']}
    - Gender: {user_info['gender']}
    - Goal: {user_info['goal']}
    - Food choice : {user_info["preference"]}

    Medical Conditions:
    - Diseases: {', '.join(diseases) if diseases else 'None'}

    Preferred Foods (from recent logs, consider if suitable but prioritize overall goals and ensure they are from the 'Available Food Items' list if chosen):
    - {', '.join(preferred_foods) if preferred_foods else 'None'}

    Instructions:
    - For the current meal time(s) specified in the "Meals for the current time" section below, recommend 2 to 5 suitable food items.
    - Your recommendations MUST be selected EXCLUSIVELY from the 'Available Food Items' list.
    - Ensure balance and variety, considering the user's preferences, goal, and medical conditions when selecting from the 'Available Food Items'.
    - Do not recommend food items that are listed in the 'Recent Food Logs'.
    - Prioritize nutritional goals (like calorie targets if implied by 'Goal', carbohydrate/protein/fat balance) and medical compatibility when making your selections from the list.
    - For each selected food item, you will provide its name (exactly as in the list), your reason for recommending it (considering user profile, current meal, goals, conditions), its calorie count (from 'food_calories' in the list, as a number), and its image URL (from 'url' in the list).

    Recent Food Logs:
    {serialized_logs}

    Meals for the current time:
    {chr(10).join([f'      "{m}": [{{ "food_name": "Exact food_name from Available Food Items list", "reason": "Your reason for recommending this item from the list based on user profile and current meal time", "cal": "The numeric food_calories value from the selected item in the Available Food Items list", "url": "The exact url string from the selected item in the Available Food Items list" }}],' for m in meals])}

    Do not include any extra explanation. Just return valid JSON only.
    """

    # 11) Call the LLM
    model = GenerativeModel("gemini-1.5-flash")
    resp = model.generate_content(prompt)
    # 12) Parse its JSON
    text = resp.text.strip().replace("```json", "").replace("```", "")
    try:
        recommendations = json.loads(text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM JSON parse error: {e}")

    return {
        "calories_today": total_calories_today,
        "number_of_food_items": len(meals),
        "total_calories_burnt": 120,  # This is a placeholder value
        "recommendations": recommendations
    }