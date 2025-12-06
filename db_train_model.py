"""
Système d'entraînement automatique robuste avec PostgreSQL
Utilise TOUS les matchs sauvegardés dans la base de données pour entraîner le modèle
"""
import traceback
from datetime import datetime
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

from models import Match, ModelVersion, TrainingLog, get_session_factory, Base
from pathlib import Path

SessionLocal = get_session_factory()
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)


def load_data_from_db() -> pd.DataFrame:
    """Charge TOUS les matchs depuis la base de données PostgreSQL."""
    session = SessionLocal()
    try:
        # Récupérer tous les matchs avec résultat final et cotes complètes
        matches = session.query(Match).filter(
            Match.outcome.in_(['1', 'N', '2']),
            Match.odds_1.isnot(None),
            Match.odds_x.isnot(None),
            Match.odds_2.isnot(None)
        ).all()
        
        if not matches:
            raise ValueError("Aucune donnée exploitable pour entraîner le modèle dans la base de données.")
        
        print(f"[TRAIN] 📊 {len(matches)} match(s) récupéré(s) depuis la base de données")
        
        # Convertir en DataFrame
        data = []
        for match in matches:
            data.append({
                'odds_1': match.odds_1,
                'odds_x': match.odds_x,
                'odds_2': match.odds_2,
                'outcome': match.outcome,
            })
        
        df = pd.DataFrame(data)
        
        # Statistiques
        print(f"[TRAIN] 📊 Répartition des résultats:")
        print(f"  - Victoire équipe 1 (1): {len(df[df['outcome'] == '1'])} matchs")
        print(f"  - Match nul (N): {len(df[df['outcome'] == 'N'])} matchs")
        print(f"  - Victoire équipe 2 (2): {len(df[df['outcome'] == '2'])} matchs")
        
        return df
        
    except Exception as e:
        print(f"[TRAIN] ❌ Erreur lors du chargement des données: {e}")
        traceback.print_exc()
        raise
    finally:
        session.close()


def train_and_save_model() -> None:
    """
    Entraîne le modèle de machine learning avec TOUS les matchs de la base de données.
    Sauvegarde le modèle et enregistre les métriques dans la base de données.
    """
    session = SessionLocal()
    training_log = None
    
    try:
        # Créer un log d'entraînement
        training_log = TrainingLog(
            training_started_at=datetime.utcnow(),
            status='started',
            samples_used=0
        )
        session.add(training_log)
        session.commit()
        
        print(f"[TRAIN] 🔄 Début de l'entraînement à {datetime.utcnow().isoformat()}")
        
        # Charger les données depuis la base de données
        df = load_data_from_db()
        training_log.samples_used = len(df)
        session.commit()

        X = df[["odds_1", "odds_x", "odds_2"]]
        y = df["outcome"]

        print(f"[TRAIN] 🔄 Division des données (train/test split)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"[TRAIN] 📊 Données d'entraînement: {len(X_train)} échantillons")
        print(f"[TRAIN] 📊 Données de test: {len(X_test)} échantillons")

        print(f"[TRAIN] 🔄 Création du modèle Random Forest...")
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            random_state=42,
            n_jobs=-1,
        )

        print(f"[TRAIN] 🔄 Entraînement du modèle en cours...")
        start_time = datetime.utcnow()
        model.fit(X_train, y_train)
        training_duration = (datetime.utcnow() - start_time).total_seconds()
        print(f"[TRAIN] ✅ Modèle entraîné avec succès en {training_duration:.2f} secondes")

        print(f"[TRAIN] 🔄 Évaluation du modèle...")
        y_pred = model.predict(X_test)
        
        # Calculer les métriques
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        report = classification_report(y_test, y_pred, digits=3)
        print("[TRAIN] 📊 Rapport de performance du modèle :")
        print(report)
        print(f"[TRAIN] 📊 Métriques globales:")
        print(f"  - Accuracy: {accuracy:.3f}")
        print(f"  - Precision: {precision:.3f}")
        print(f"  - Recall: {recall:.3f}")
        print(f"  - F1-Score: {f1:.3f}")

        # Générer un nom de version basé sur la date et l'heure
        version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        model_filename = f"model_{version}.joblib"
        model_path = MODELS_DIR / model_filename

        print(f"[TRAIN] 🔄 Sauvegarde du modèle...")
        joblib.dump(model, model_path)
        print(f"[TRAIN] ✅ Modèle sauvegardé dans : {model_path}")

        # Désactiver les anciens modèles
        session.query(ModelVersion).update({ModelVersion.is_active: False})
        
        # Créer une nouvelle version du modèle dans la base de données
        model_version = ModelVersion(
            version=version,
            model_path=str(model_path),
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            training_samples=len(df),
            is_active=True,
            notes=f"Entraîné automatiquement avec {len(df)} matchs"
        )
        session.add(model_version)
        
        # Mettre à jour le log d'entraînement
        training_log.training_completed_at = datetime.utcnow()
        training_log.status = 'completed'
        training_log.duration_seconds = training_duration
        training_log.model_version_id = model_version.id
        training_log.metrics = f"accuracy={accuracy:.3f}, precision={precision:.3f}, recall={recall:.3f}, f1={f1:.3f}"
        
        session.commit()
        
        print(f"[TRAIN] ✅ Modèle version {version} enregistré dans la base de données")
        print(f"[TRAIN] ✅ Modèle activé et prêt à être utilisé")
        
    except Exception as e:
        error_msg = f"Erreur lors de l'entraînement du modèle: {str(e)}"
        print(f"[TRAIN] ❌ {error_msg}")
        traceback.print_exc()
        
        if training_log:
            training_log.status = 'failed'
            training_log.training_completed_at = datetime.utcnow()
            training_log.error_message = str(e)
            session.commit()
        
        raise
    finally:
        session.close()


def get_active_model_path() -> str:
    """Récupère le chemin du modèle actif depuis la base de données."""
    session = SessionLocal()
    try:
        model_version = session.query(ModelVersion).filter(
            ModelVersion.is_active == True
        ).order_by(ModelVersion.created_at.desc()).first()
        
        if model_version:
            return model_version.model_path
        return None
    finally:
        session.close()


if __name__ == "__main__":
    train_and_save_model()

