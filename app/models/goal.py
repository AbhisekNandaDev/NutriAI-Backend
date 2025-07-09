from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship
from core.database import Base

class Goal(Base):
    __tablename__ = "goals"

    id = Column(Integer, primary_key=True, index=True)
    goal = Column(String, unique=True, index=True, nullable=False)
    desc = Column(String, nullable=False)
    yoga_needed = Column(String, nullable=True)  # Aligned with Pydantic
    good_to_have_foods = Column(String, nullable=True)  # Fixed typo
    avoid_foods = Column(String, nullable=True)

    # Many-to-one: Multiple users can have this goal
    users = relationship("User", back_populates="goal")