from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship
from core.database import Base

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
    medical_conditions = Column(Integer, ForeignKey("diseases.id"), nullable=True)
    preference = Column(String, nullable=True)  # Fixed typo: "preferance" -> "preference"
    goal_id = Column(Integer, ForeignKey("goals.id"), nullable=True)  # Renamed to goal_id

    # Relationships
    disease = relationship("Diseases", back_populates="users")
    goal = relationship("Goal", back_populates="users")  # Refers to goal_id