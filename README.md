
# NutriAI API

NutriAI is a comprehensive FastAPI-powered backend for a wellness and nutrition application. It offers a rich set of features including personalized food and yoga recommendations, nutritional analysis, real-time pose correction, and health tracking. The API is designed to be robust, scalable, and easy to integrate with a frontend application.

## Features

- **User Management**: Secure user registration and login using JWT-based authentication. Users can manage their profile, including personal metrics (age, weight, height), dietary preferences, medical conditions, and wellness goals.

- **AI-Powered Food Analysis**:
  - *Image-to-Nutrition*: Upload an image of a meal, and the API uses Google's Gemini Vision model to identify food items and return detailed nutritional information (calories, macros).
  - *Food Logging*: Log meals manually or from image analysis to track daily intake.

- **Real-Time Yoga Pose Correction**:
  - A WebSocket endpoint streams real-time feedback on yoga poses.
  - Utilizes a TensorFlow Lite (Movenet) model to compare the user's pose from their camera feed against a reference image, providing live corrective instructions.

- **Personalized Recommendations & Analytics**:
  - *Food Recommendations*: Get daily meal recommendations based on user goals, medical conditions, dietary preferences, and past consumption logs.
  - *Health Analytics*: Access detailed reports on daily nutritional intake, macro distribution, top consumed foods, and progress over time.
  - *Yoga Analytics*: Track total yoga time, number of sessions, and identify the most frequently practiced poses.
  - *Health Metrics*: Automatically calculates BMI and BMR based on user profile data.

- **Comprehensive Database**:
  - Pre-populated data for various foods, diseases with dietary advice, and wellness goals.
  - Extensive list of yoga poses with benefits and suggested durations.

- **Dynamic API**:
  - Endpoints for searching, paginating, and retrieving all data.
  - CRUD operations for managing users, goals, diseases, and food logs.

## Tech Stack

- **Backend**: FastAPI, Python 3.x
- **Database**: SQLAlchemy ORM, SQLite
- **Authentication**: Passlib (for hashing), python-jose (for JWT)
- **AI & Machine Learning**:
  - google-cloud-aiplatform (for Gemini)
  - tensorflow-lite (for Movenet pose estimation)
- **Cloud Services**: Google Cloud Storage for image hosting
- **WebSockets**: For real-time communication in the yoga feature
- **Server**: Uvicorn

## Project Structure

```
app/
├── core/
│   ├── database.py         # Database engine and session setup
│   └── security.py         # Authentication, password hashing, JWT creation
├── data/
│   ├── diseases.json       # Pre-defined diseases data
│   ├── foods.json          # Pre-defined food data
│   ├── goals.json          # Pre-defined goals data
│   └── raw_yoga.json       # Pre-defined yoga poses data
├── models/
│   └── models.py           # SQLAlchemy database models
├── routes/
│   ├── analytics_router.py # Endpoints for analytics
│   ├── diseases_routers.py # Endpoints for diseases
│   ├── food_routes.py      # Endpoints for food, logging, and recommendations
│   ├── goal_routes.py      # Endpoints for goals
│   ├── user_routes.py      # Endpoints for user management
│   ├── yoga_routes.py      # Endpoints for yoga data and logging
│   ├── yoga_routes1.py     # WebSocket for real-time pose correction
│   └── weight_router.py    # Endpoints for weight tracking
├── schema/
│   ├── __init__.py
│   ├── analytics.py        # Pydantic schemas for analytics
│   ├── diseases.py         # Pydantic schemas for diseases
│   └── ... (other schemas)
├── main.py                 # Main FastAPI application, mounts all routers
├── requirements.txt        # Project dependencies
└── crate_table.py          # Script to create database tables
```

## Setup and Installation

### 1. Prerequisites

- Python 3.8+
- A Google Cloud Platform (GCP) project with the Vertex AI and Cloud Storage APIs enabled.

### 2. Clone the Repository

```bash
git clone https://github.com/AbhisekNandaDev/NutriAI-Backend.git
cd NutriAI-Backend
```

### 3. Set Up a Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scriptsctivate

# macOS & Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r app/requirements.txt
```

### 5. Configure Environment

- Create a GCP service account key (Vertex AI User + Storage Object Admin).
- Save the JSON key in `app/routes/` as `nutri-ai-453006-857e0a3faa1b.json` or update `credentials_path` in `food_routes.py`.

### 6. Add Movenet TensorFlow Lite Model

- Download and place `movenet.tflite` in `app/routes/`.

### 7. Set Up the Database

```bash
python app/crate_table.py create_tables
```

### 8. Populate Initial Data (Optional)

Use the data in `app/data/*.json` with respective POST endpoints.

## Running the Application

```bash
uvicorn app.main:app --reload
```

- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc

## API Endpoints Overview

**Users (`/users`)**: `/create_user`, `/login`, `/`, `/reports`, `/update_user`  
**Diseases (`/diseases`)**: `/create`, `/all`, `/{disease_id}`, `/search-diseases/`  
**Goals (`/goal`)**: `/create`, `/all`, `/search/`, `/{goal_id}`  
**Food (`/food`)**: `/create`, `/all`, `/upload`, `/predict`, `/add_to_log`, `/preferred_food`, `/search`, `/{food_id}`, `/log`, `/logs/user`, `/preferred_food_items`, `/food/recommend`  
**Analytics (`/analytics`)**: `/daily_nutrition`, `/user_health_metrics`, `/behavioral_insights`, `/nutritional_deficiencies`, `/nutrition_analytics`, `/analytics/yoga`  
**Yoga (`/yoga`)**: `/data/`, `/log/`, WebSocket: `ws://127.0.0.1:8000/ws/pose-compare/{yoga_data_id}`  
**Weight (`/weight`)**: `/weight/`, `/{weight_id}`
