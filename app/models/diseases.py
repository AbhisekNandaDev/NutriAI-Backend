from sqlalchemy import Column, Integer, String
from core.database import Base
from sqlalchemy.orm import relationship

class Diseases(Base):
    __tablename__ = "diseases"

    id = Column(Integer, primary_key=True, index=True)
    diseases_name = Column(String, unique=True, index=True, nullable=False)
    diseases_desc = Column(String, nullable=False)
    food_good_to_have = Column(String, nullable=False)  # Comma-separated string
    food_to_avoid = Column(String, nullable=False)  # Comma-separated string

    users = relationship("User", back_populates="diseases")
    