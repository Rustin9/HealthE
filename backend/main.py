import os
import json
from datetime import datetime
from typing import Dict, Any, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Text,
    DateTime,
)
from sqlalchemy.orm import sessionmaker, declarative_base

# ======== Config ========

# DB_URL = os.getenv(
#     "DB_URL",
#     "postgresql+psycopg2://healthe:healthe@db:5432/healthe_db"
# )

DB_URL = os.getenv("DB_URL")
if not DB_URL:
    raise RuntimeError("DB_URL environment variable is not set")


engine = create_engine(DB_URL, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()


# ======== DB Model ========

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    prediction_type = Column(String(50), nullable=False)
    inputs_json = Column(Text, nullable=False)
    output_json = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


# Create tables at startup
Base.metadata.create_all(bind=engine)


# ======== Schemas ========

class LogPredictionRequest(BaseModel):
    prediction_type: str
    inputs: Dict[str, Any]
    output: Dict[str, Any]


class PredictionItem(BaseModel):
    id: int
    prediction_type: str
    inputs: Dict[str, Any]
    output: Dict[str, Any]
    created_at: datetime


# ======== App ========

app = FastAPI(title="HealthE Backend", version="1.0.0")


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/log_prediction")
def log_prediction(req: LogPredictionRequest):
    try:
        session = SessionLocal()
        db_obj = Prediction(
            prediction_type=req.prediction_type,
            inputs_json=json.dumps(req.inputs),
            output_json=json.dumps(req.output),
            created_at=datetime.utcnow(),
        )
        session.add(db_obj)
        session.commit()
        session.refresh(db_obj)
        return {"id": db_obj.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()


@app.get("/history", response_model=List[PredictionItem])
def get_history(limit: int = 10):
    session = SessionLocal()
    try:
        q = (
            session.query(Prediction)
            .order_by(Prediction.id.desc())
            .limit(limit)
            .all()
        )

        result: List[PredictionItem] = []
        for row in q:
            result.append(
                PredictionItem(
                    id=row.id,
                    prediction_type=row.prediction_type,
                    inputs=json.loads(row.inputs_json),
                    output=json.loads(row.output_json),
                    created_at=row.created_at,
                )
            )
        return result
    finally:
        session.close()
