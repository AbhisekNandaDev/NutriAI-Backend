# yoga.py
from typing import Optional
from pydantic import BaseModel
from typing import List


# Schemas (Pydantic models) for request/response validation
class YogaDataSchema(BaseModel):
    id: int
    yoga_name: str
    yoga_benefit: str
    yoga_pose_url: Optional[str] = None
    yoga_time_in_sec: Optional[int] = None
    yoga_category: Optional[str] = None

    class Config:
        orm_mode = True

class YogaPoseLogSchema(BaseModel):
    id: int
    user_id: int
    yoga_id: int
    time: str
    date: str

    class Config:
        orm_mode = True

class YogaDataCreate(BaseModel):
    yoga_name: str
    yoga_benefit: str
    yoga_pose_url: Optional[str] = None
    yoga_time_in_sec: Optional[int] = None
    yoga_category: Optional[str] = None
    
class YogaPoseLogCreate(BaseModel):
    yoga_id: int
    time: str
    date: str


class FrequentPoseAnalytic(BaseModel):
    """
    Represents analytics for a frequently performed yoga pose.
    """
    yoga_id: int
    yoga_name: str
    count: int  # How many times this pose was logged
    total_duration_seconds: int  # Total time spent on this pose (count * yoga_time_in_sec_per_instance)

    class Config:
        orm_mode = True

class UserYogaAnalyticsResponse(BaseModel):
    """
    Overall yoga analytics for a user.
    """
    user_id: int
    total_yoga_time_seconds: int
    total_sessions_logged: int
    average_duration_per_session_seconds: Optional[float] = None
    most_frequent_poses: List[FrequentPoseAnalytic]

    class Config:
        orm_mode = True
