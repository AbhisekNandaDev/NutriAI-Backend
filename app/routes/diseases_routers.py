# routes/diseases_routes.py
from fastapi import APIRouter, Depends, HTTPException
from typing import List
from sqlalchemy.orm import Session
from core.database import get_db
from models.models import Diseases  # Import from models.py
from schema.diseases import DiseasesCreate, DiseasesResponse
from fastapi import Query


diseases_router = APIRouter()

@diseases_router.post("/create", response_model=List[DiseasesResponse])
def create_diseases(diseases: List[DiseasesCreate], db: Session = Depends(get_db)):
    db_diseases = []

    for disease in diseases:
        db_disease = Diseases(
            disease_name=disease.disease_name,
            disease_desc=disease.disease_desc,
            food_good_to_have=disease.food_good_to_have,
            food_to_avoid=disease.food_to_avoid
        )
        db.add(db_disease)
        db_diseases.append(db_disease)

    db.commit()
    for disease in db_diseases:
        db.refresh(disease)

    return db_diseases

@diseases_router.get("/all", response_model=List[DiseasesResponse])
def get_all_diseases(
    db: Session = Depends(get_db),
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1),
):
    skip = (page - 1) * limit
    diseases = db.query(Diseases).offset(skip).limit(limit).all()
    return diseases


@diseases_router.get("/{disease_id}", response_model=DiseasesResponse)
def get_disease(disease_id: int, db: Session = Depends(get_db)):
    disease = db.query(Diseases).filter(Diseases.id == disease_id).first()
    if disease is None:
        raise HTTPException(status_code=404, detail="Disease not found")
    return disease



@diseases_router.post("/search-diseases/", response_model=List[DiseasesResponse])
def search_diseases(query: str, db: Session = Depends(get_db)):
    results = db.query(Diseases).filter(Diseases.disease_name.ilike(f"%{query}%")).all()
    print(f"Search query: {query}")
    print(f"Search results: {results}")
    return results
