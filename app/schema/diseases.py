from pydantic import BaseModel

class DiseasesCreate(BaseModel):
    disease_name: str
    disease_desc: str
    food_good_to_have: str
    food_to_avoid: str

class DiseasesResponse(BaseModel):
    id: int
    disease_name: str
    disease_desc: str
    food_good_to_have: str
    food_to_avoid: str

    class Config:
        orm_mode = True