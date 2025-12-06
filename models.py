"""
Modèles de base de données SQLAlchemy pour ORACX PRED
"""
from datetime import datetime
from typing import Optional

from sqlalchemy import Column, Integer, Float, String, DateTime, Boolean, Index, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
import os

Base = declarative_base()


class Match(Base):
    """Modèle pour les matchs collectés et sauvegardés."""
    __tablename__ = 'matches'

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp_utc = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    event_id = Column(Integer, nullable=True, index=True)  # ID de l'événement chez 1xBet
    league = Column(String(255), nullable=True, index=True)
    team1 = Column(String(255), nullable=False)
    team2 = Column(String(255), nullable=False)
    odds_1 = Column(Float, nullable=True)
    odds_x = Column(Float, nullable=True)
    odds_2 = Column(Float, nullable=True)
    score1 = Column(Integer, nullable=True)
    score2 = Column(Integer, nullable=True)
    status = Column(String(255), nullable=True)
    outcome = Column(String(1), nullable=True, index=True)  # '1', 'N', '2'
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Indexes pour améliorer les performances
    __table_args__ = (
        Index('idx_event_id', 'event_id'),
        Index('idx_outcome', 'outcome'),
        Index('idx_timestamp', 'timestamp_utc'),
        Index('idx_league', 'league'),
        Index('idx_teams', 'team1', 'team2'),
    )

    def __repr__(self):
        return f"<Match(id={self.id}, {self.team1} vs {self.team2}, outcome={self.outcome})>"


class ModelVersion(Base):
    """Modèle pour versionner les modèles ML entraînés."""
    __tablename__ = 'model_versions'

    id = Column(Integer, primary_key=True, autoincrement=True)
    version = Column(String(50), nullable=False, unique=True, index=True)
    model_path = Column(String(500), nullable=False)
    accuracy = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    training_samples = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    is_active = Column(Boolean, default=False, nullable=False, index=True)
    notes = Column(Text, nullable=True)

    def __repr__(self):
        return f"<ModelVersion(version={self.version}, accuracy={self.accuracy}, active={self.is_active})>"


class TrainingLog(Base):
    """Logs des entraînements pour traçabilité."""
    __tablename__ = 'training_logs'

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_version_id = Column(Integer, nullable=True, index=True)
    training_started_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    training_completed_at = Column(DateTime, nullable=True)
    status = Column(String(50), nullable=False, index=True)  # 'started', 'completed', 'failed'
    samples_used = Column(Integer, nullable=False)
    duration_seconds = Column(Float, nullable=True)
    error_message = Column(Text, nullable=True)
    metrics = Column(Text, nullable=True)  # JSON string avec les métriques

    def __repr__(self):
        return f"<TrainingLog(id={self.id}, status={self.status}, samples={self.samples_used})>"


# Configuration de la base de données
def get_database_url() -> str:
    """Récupère l'URL de la base de données depuis les variables d'environnement."""
    return os.environ.get(
        'DATABASE_URL',
        'postgresql://oracxpred:oracxpred123@localhost:5432/oracxpred'
    )


def create_engine_instance():
    """Crée une instance de l'engine SQLAlchemy."""
    database_url = get_database_url()
    return create_engine(
        database_url,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
        pool_recycle=3600,
        echo=False  # Mettre à True pour voir les requêtes SQL
    )


def get_session_factory():
    """Crée une factory de sessions."""
    engine = create_engine_instance()
    return sessionmaker(bind=engine)


def init_db():
    """Initialise la base de données en créant toutes les tables."""
    engine = create_engine_instance()
    Base.metadata.create_all(engine)
    print("[DB] ✅ Base de données initialisée avec succès")


def drop_db():
    """Supprime toutes les tables (ATTENTION: destructif!)."""
    engine = create_engine_instance()
    Base.metadata.drop_all(engine)
    print("[DB] ⚠️ Toutes les tables ont été supprimées")


if __name__ == "__main__":
    # Pour tester la création de la base de données
    init_db()

