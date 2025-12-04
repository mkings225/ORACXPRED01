import os
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
    if not os.path.isfile(DATA_PATH):
        raise FileNotFoundError(f"Fichier de données introuvable : {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # Garder uniquement les lignes avec un résultat clair + cotes complètes
    df = df.dropna(subset=["outcome", "odds_1", "odds_x", "odds_2"])
    df = df[df["outcome"].isin(["1", "N", "2"])]

    if df.empty:
        raise ValueError("Aucune donnée exploitable pour entraîner le modèle.")

    return df


def train_and_save_model() -> None:
    df = load_data()

    X = df[["odds_1", "odds_x", "odds_2"]]
    y = df["outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, digits=3)
    print("Rapport de performance du modèle :")
    print(report)

    joblib.dump(model, MODEL_PATH)
    print(f"Modèle sauvegardé dans : {MODEL_PATH}")


if __name__ == "__main__":
    train_and_save_model()


