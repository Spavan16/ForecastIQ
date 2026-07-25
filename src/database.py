import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import json

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base, Session

from src.utils import get_logger, DB_PATH

logger = get_logger("DatabaseManager")

# NOTE (P2 audit finding, resolved): this module was originally fully implemented but not
# imported or called anywhere in backend/src/main.py or the run.sh grading pipeline
# (generate_features.py / predict.py). It has since been wired into main.py's
# get_analytics_state() — every completed analysis run now calls save_forecast_run(), and
# GET /api/runs exposes the resulting history via get_recent_forecast_runs(). The CLI grading
# pipeline still never touches this module (by design — run.sh's contract is model.pkl in,
# predictions.csv out, no persistence step), so this remains purely a backend/dashboard-side
# feature.

Base = declarative_base()

def _utcnow():
    return datetime.datetime.now(datetime.timezone.utc)

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True)
    agency_name = Column(String)
    created_at = Column(DateTime, default=_utcnow)


class UploadedDataset(Base):
    __tablename__ = "uploaded_datasets"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    channel_name = Column(String)
    file_path = Column(String)
    record_count = Column(Integer)
    data_quality_score = Column(Float)
    uploaded_at = Column(DateTime, default=_utcnow)


class ForecastRun(Base):
    __tablename__ = "forecast_runs"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    run_label = Column(String)
    parameters_json = Column(Text)
    predictions_summary_json = Column(Text)
    created_at = Column(DateTime, default=_utcnow)


class Scenario(Base):
    __tablename__ = "scenarios"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    scenario_id = Column(String)
    scenario_name = Column(String)
    modifications_json = Column(Text)
    results_json = Column(Text)
    created_at = Column(DateTime, default=_utcnow)


class Report(Base):
    __tablename__ = "reports"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    report_title = Column(String)
    pdf_file_path = Column(String)
    summary_json = Column(Text)
    created_at = Column(DateTime, default=_utcnow)


class DatabaseManager:
    """
    SQLite Database Manager for ForecastIQ.
    Handles sessions, schema migrations, and persistent analytical run storage.
    """
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.engine = create_engine(f"sqlite:///{self.db_path}", connect_args={"check_same_thread": False})
        Base.metadata.create_all(bind=self.engine)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self._seed_default_user()

    def get_session(self) -> Session:
        return self.SessionLocal()

    def _seed_default_user(self):
        db = self.get_session()
        try:
            user = db.query(User).filter(User.username == "executive_lead").first()
            if not user:
                user = User(
                    username="executive_lead",
                    email="lead@forecastiq.app",
                    agency_name="ForecastIQ Demo Agency"
                )
                db.add(user)
                db.commit()
                logger.info("Default executive user account seeded into SQLite DB.")
        except Exception as e:
            logger.error(f"Error seeding default DB user: {str(e)}")
            db.rollback()
        finally:
            db.close()

    def log_dataset(self, channel_name: str, file_path: str, record_count: int, quality_score: float):
        db = self.get_session()
        try:
            user = db.query(User).first()
            ds = UploadedDataset(
                user_id=user.id if user else 1,
                channel_name=channel_name,
                file_path=file_path,
                record_count=record_count,
                data_quality_score=quality_score
            )
            db.add(ds)
            db.commit()
        except Exception as e:
            logger.error(f"Failed to log dataset: {str(e)}")
            db.rollback()
        finally:
            db.close()

    def save_forecast_run(self, label: str, params: Dict[str, Any], predictions_summary: Dict[str, Any]) -> int:
        db = self.get_session()
        try:
            user = db.query(User).first()
            run = ForecastRun(
                user_id=user.id if user else 1,
                run_label=label,
                parameters_json=json.dumps(params),
                predictions_summary_json=json.dumps(predictions_summary)
            )
            db.add(run)
            db.commit()
            db.refresh(run)
            logger.info(f"Forecast run '{label}' successfully persisted (Run ID: {run.id}).")
            return run.id
        except Exception as e:
            logger.error(f"Failed to persist forecast run: {str(e)}")
            db.rollback()
            return -1
        finally:
            db.close()

    def get_recent_forecast_runs(self, limit: int = 10) -> List[Dict[str, Any]]:
        db = self.get_session()
        try:
            runs = db.query(ForecastRun).order_by(ForecastRun.created_at.desc()).limit(limit).all()
            res = []
            for r in runs:
                res.append({
                    "id": r.id,
                    "run_label": r.run_label,
                    "created_at": r.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                    "parameters": json.loads(r.parameters_json),
                    "summary": json.loads(r.predictions_summary_json)
                })
            return res
        except Exception as e:
            logger.error(f"Failed to query recent forecast runs: {str(e)}")
            return []
        finally:
            db.close()
