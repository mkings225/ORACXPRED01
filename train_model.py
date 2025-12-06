import os
import traceback
from typing import List

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "data", "matches.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.joblib")


def load_data() -> pd.DataFrame:
    """Charge et prépare les données pour l'entraînement."""
    if not os.path.isfile(DATA_PATH):
        error_msg = f"Fichier de données introuvable : {DATA_PATH}\nVeuillez d'abord collecter des données avec la fonction append_matches_to_csv()"
        print(f"[TRAIN] ❌ {error_msg}")
        raise FileNotFoundError(error_msg)

    try:
        df = pd.read_csv(DATA_PATH)
        print(f"[TRAIN] 📊 Fichier chargé: {len(df)} lignes au total")
    except Exception as e:
        error_msg = f"Erreur lors de la lecture du fichier CSV: {str(e)}"
        print(f"[TRAIN] ❌ {error_msg}")
        raise

    # Garder uniquement les lignes avec un résultat clair + cotes complètes
    initial_count = len(df)
    df = df.dropna(subset=["outcome", "odds_1", "odds_x", "odds_2"])
    df = df[df["outcome"].isin(["1", "N", "2"])]
    
    filtered_count = len(df)
    print(f"[TRAIN] 📊 Données filtrées: {filtered_count} lignes exploitables (sur {initial_count})")

    if df.empty:
        error_msg = "Aucune donnée exploitable pour entraîner le modèle. Assurez-vous d'avoir collecté des matchs avec des résultats finaux."
        print(f"[TRAIN] ❌ {error_msg}")
        raise ValueError(error_msg)

    return df


def train_and_save_model() -> None:
    """Entraîne le modèle de machine learning et le sauvegarde."""
    try:
        df = load_data()

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
        model.fit(X_train, y_train)
        print(f"[TRAIN] ✅ Modèle entraîné avec succès")

        print(f"[TRAIN] 🔄 Évaluation du modèle...")
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, digits=3)
        print("[TRAIN] 📊 Rapport de performance du modèle :")
        print(report)

        print(f"[TRAIN] 🔄 Sauvegarde du modèle...")
        joblib.dump(model, MODEL_PATH)
        print(f"[TRAIN] ✅ Modèle sauvegardé dans : {MODEL_PATH}")
    except FileNotFoundError:
        # Erreur déjà gérée dans load_data()
        raise
    except ValueError as e:
        # Erreur déjà gérée dans load_data()
        raise
    except Exception as e:
        error_msg = f"Erreur lors de l'entraînement du modèle: {str(e)}"
        print(f"[TRAIN] ❌ {error_msg}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    train_and_save_model()


