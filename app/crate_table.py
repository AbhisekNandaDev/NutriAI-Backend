from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, Integer, String, ForeignKey, Float, Table  # Import all your column types
from sqlalchemy.orm import relationship
from sqlalchemy import create_engine, Table, Column, Integer, String, Float, ForeignKey  # Import all your column types
from sqlalchemy.orm import relationship, sessionmaker
import sys

# SQLite for simplicity; replace with your DB URL (e.g., PostgreSQL)
DATABASE_URL = "sqlite:///./test.db"  # Make sure this matches your database URL

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})  # Add connect_args for SQLite if needed
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Copy your entire models.py content here
# ----------------------------------------------------------------------
# models/models.py
# ----------------------------------------------------------------------

# Association tables for Many-to-Many
user_disease_association = Table(
    'user_disease_association', Base.metadata,
    Column('user_id', Integer, ForeignKey('users.id')),
    Column('disease_id', Integer, ForeignKey('diseases.id'))
)

user_goal_association = Table(
    'user_goal_association', Base.metadata,
    Column('user_id', Integer, ForeignKey('users.id')),
    Column('goal_id', Integer, ForeignKey('goals.id'))
)

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    password = Column(String, nullable=False)
    age = Column(Integer, nullable=True)
    gender = Column(String, nullable=True)
    height_ft = Column(Integer, nullable=True)
    height_in = Column(Integer, nullable=True)
    weight_kg = Column(Integer, nullable=True)
    weight_gm = Column(Integer, nullable=True)
    preference = Column(String, nullable=True)

    diseases = relationship("Diseases", secondary=user_disease_association, back_populates="users")
    goals = relationship("Goal", secondary=user_goal_association, back_populates="users")

    food_logs = relationship("FoodLog", back_populates="user")
    user_preferance_food = relationship("UserPreferanceFood", back_populates="user")
    yoga_pose_logs = relationship("YogaPoseLog", back_populates="user")
    weight_logs = relationship("Weight", back_populates="user")

class Goal(Base):
    __tablename__ = "goals"
    id = Column(Integer, primary_key=True, index=True)
    goal = Column(String, unique=True, index=True, nullable=False)
    desc = Column(String, nullable=False)
    yoga_needed = Column(String, nullable=True)
    good_to_have_foods = Column(String, nullable=True)
    avoid_foods = Column(String, nullable=True)
    users = relationship("User", secondary=user_goal_association, back_populates="goals")

class Diseases(Base):
    __tablename__ = "diseases"
    id = Column(Integer, primary_key=True, index=True)
    disease_name = Column(String, unique=True, index=True, nullable=False)
    disease_desc = Column(String, nullable=False)
    food_good_to_have = Column(String, nullable=False)
    food_to_avoid = Column(String, nullable=False)
    users = relationship("User", secondary=user_disease_association, back_populates="diseases")

class Food(Base):
    __tablename__ = "foods"
    id = Column(Integer, primary_key=True, index=True)
    food_name = Column(String, unique=True, index=True, nullable=False)
    food_desc = Column(String, nullable=False)
    food_carbohydrate = Column(Float, nullable=False)
    food_protein = Column(Float, nullable=False)
    food_fat = Column(Float, nullable=False)
    food_calories = Column(Float, nullable=False)
    user_preferance_food = relationship("UserPreferanceFood", back_populates="food")

class FoodLog(Base):
    __tablename__ = "food_logs"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    food_image = Column(String, nullable=True)
    food_name = Column(String, index=True, nullable=False)
    food_desc = Column(String, nullable=False)
    food_carbohydrate = Column(Float, nullable=False)
    food_protein = Column(Float, nullable=False)
    food_fat = Column(Float, nullable=False)
    food_calories = Column(Float, nullable=False)
    date = Column(String, nullable=False)
    user = relationship("User", back_populates="food_logs")

class UserPreferanceFood(Base):
    __tablename__ = "user_preferance_food"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    food_id = Column(Integer, ForeignKey("foods.id"), nullable=False)
    user = relationship("User", back_populates="user_preferance_food")
    food = relationship("Food", back_populates="user_preferance_food")



# New models for Yoga
class YogaData(Base):
    __tablename__ = "yoga_data"
    id = Column(Integer, primary_key=True, index=True)
    yoga_name = Column(String, unique=True, index=True, nullable=False)
    yoga_benefit = Column(String, nullable=False)
    yoga_pose_url = Column(String, nullable=True)
    yoga_time_in_sec = Column(Integer, nullable=True) # Time in seconds suggested for the pose
    yoga_category = Column(String, nullable=True)

    # Added relationship for yoga pose logs
    yoga_pose_logs = relationship("YogaPoseLog", back_populates="yoga")

class YogaPoseLog(Base):
    __tablename__ = "yoga_pose_logs"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    yoga_id = Column(Integer, ForeignKey("yoga_data.id"), nullable=False)
    # Consider using appropriate types for time and date
    time = Column(String, nullable=False) # e.g., "10:30 AM" or total seconds/minutes
    date = Column(String, nullable=False) # e.g., "YYYY-MM-DD" or using Date/DateTime type

    # Relationships
    user = relationship("User", back_populates="yoga_pose_logs")
    yoga = relationship("YogaData", back_populates="yoga_pose_logs")


class Weight(Base):
    __tablename__ = "weight_logs"  # Changed table name to weight_logs
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    weight = Column(Float, nullable=False)  # Changed column name from Weight to weight
    date = Column(String, nullable=False)
    user = relationship("User", back_populates="weight_logs") #Should be weight_logs




# ----------------------------------------------------------------------
# End of copied models.py content
# ----------------------------------------------------------------------

# Create the tables
def create_db_tables():
    Base.metadata.create_all(engine)
    print("Database tables created!")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "create_tables":
        create_db_tables()
    else:
        print("To create tables, run: python your_script_name.py create_tables")

