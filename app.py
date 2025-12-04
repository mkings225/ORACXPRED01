import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import requests
from flask import Flask, jsonify, render_template, request

from collector import append_matches_to_csv
from train_model import train_and_save_model

app = Flask(__name__)

API_URL = (
    "https://1xbet.com/service-api/LiveFeed/Get1x2_VZip"
    "?sports=85&count=40&lng=fr&gr=285&mode=4&country=96"
    "&getEmpty=true&virtualSports=true&noFilterBlockEvent=true"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.joblib"


def _load_model():
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        return None


_MODEL = _load_model()


def extract_1x2_odds(event: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Extrait les cotes 1 / N / 2 depuis le champ E."""
    odds_1 = odds_x = odds_2 = None
    for m in event.get("E", []):
        if m.get("G") == 1:  # marché 1X2
            t = m.get("T")
            c = m.get("C")
            if c is None:
                continue
            try:
                coef = float(c)
            except (TypeError, ValueError):
                continue
            if t == 1:
                odds_1 = coef
            elif t == 2:
                odds_x = coef
            elif t == 3:
                odds_2 = coef
    return odds_1, odds_x, odds_2


def extract_score(event: Dict[str, Any]) -> Tuple[Optional[int], Optional[int], str]:
    """
    Récupère le score et le statut de match depuis le champ SC.
    - SC.FS.S1 / SC.FS.S2 : buts équipe 1 / équipe 2 (si dispo)
    - SC.SLS ou SC.CPS : texte de statut (minute, 'Début dans 13 minutes', etc.)
    """
    sc = event.get("SC") or {}
    fs = sc.get("FS") or {}

    s1 = fs.get("S1")
    s2 = fs.get("S2")

    try:
        score1 = int(s1) if s1 is not None else None
    except (TypeError, ValueError):
        score1 = None

    try:
        score2 = int(s2) if s2 is not None else None
    except (TypeError, ValueError):
        score2 = None

    status = sc.get("SLS") or sc.get("CPS") or ""
    return score1, score2, status


def ai_predict(
    odds_1: Optional[float],
    odds_x: Optional[float],
    odds_2: Optional[float],
) -> Dict[str, Any]:
    """
    IA de prédiction :
    - si un modèle ML entraîné est disponible, on l'utilise ;
    - sinon, on retombe sur la pseudo-IA basée sur les probabilités implicites.
    """
    if not all([odds_1, odds_x, odds_2]):
        return {"prediction": "indécis", "confidence": 0.0, "probs": {}}

    # 1) Si un modèle est chargé, on l'utilise
    if _MODEL is not None:
        try:
            X = np.array([[odds_1, odds_x, odds_2]], dtype=float)
            proba = _MODEL.predict_proba(X)[0]
            classes: List[str] = list(_MODEL.classes_)  # type: ignore[attr-defined]

            probs = {cls: float(p) for cls, p in zip(classes, proba)}
            # On ne garde que 1 / N / 2
            for key in list(probs.keys()):
                if key not in {"1", "N", "2"}:
                    probs.pop(key, None)

            if not probs:
                raise ValueError("Aucune proba exploitable depuis le modèle.")

            best = max(probs, key=probs.get)
            conf = probs[best]

            return {
                "prediction": best,
                "confidence": round(conf * 100, 1),
                "probs": {k: round(v * 100, 1) for k, v in probs.items()},
            }
        except Exception:
            # En cas de souci avec le modèle, on repasse sur la méthode simple
            pass

    # 2) Fallback : probabilités implicites à partir des cotes
    inv1 = 1.0 / odds_1
    invx = 1.0 / odds_x
    inv2 = 1.0 / odds_2
    total = inv1 + invx + inv2

    p1 = inv1 / total
    px = invx / total
    p2 = inv2 / total

    probs = {"1": p1, "N": px, "2": p2}
    best = max(probs, key=probs.get)
    conf = probs[best]

    return {
        "prediction": best,
        "confidence": round(conf * 100, 1),
        "probs": {k: round(v * 100, 1) for k, v in probs.items()},
    }


def fetch_matches() -> List[Dict[str, Any]]:
    """Récupère les matchs depuis l’API 1xBet et prépare les données pour l’UI."""
    resp = requests.get(API_URL, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    events = data.get("Value", [])

    matches: List[Dict[str, Any]] = []

    for ev in events:
        odds_1, odds_x, odds_2 = extract_1x2_odds(ev)
        ai = ai_predict(odds_1, odds_x, odds_2)
        score1, score2, status = extract_score(ev)

        matches.append(
            {
                "league": ev.get("L"),
                "team1": ev.get("O1"),
                "team2": ev.get("O2"),
                "odds_1": odds_1,
                "odds_x": odds_x,
                "odds_2": odds_2,
                "prediction": ai["prediction"],
                "confidence": ai["confidence"],
                "prob_1": ai["probs"].get("1"),
                "prob_x": ai["probs"].get("N"),
                "prob_2": ai["probs"].get("2"),
                "score1": score1,
                "score2": score2,
                "status": status,
            }
        )

    return matches


def _check_task_token() -> bool:
    """Vérifie le token de sécurité pour les tâches 'cron' externes."""
    expected = os.environ.get("TASK_TOKEN")
    if not expected:
        # Si aucun token défini côté serveur, on bloque par sécurité
        return False
    provided = request.args.get("token") or request.headers.get("X-Task-Token")
    return provided == expected


@app.route("/")
def index() -> str:
    # On envoie une première liste pour ne pas avoir une page vide
    matches = fetch_matches()
    return render_template("index.html", matches=matches)


@app.route("/api/matches")
def api_matches():
    # Endpoint JSON utilisé par l'interface dynamique (AJAX)
    return jsonify(fetch_matches())


@app.route("/api/predict", methods=["GET", "POST"])
def api_predict():
    """
    API de prédictions pour le modèle entraîné.
    
    Paramètres (GET ou POST):
    - odds_1: Cote pour la victoire de l'équipe 1 (float, requis)
    - odds_x: Cote pour le match nul (float, requis)
    - odds_2: Cote pour la victoire de l'équipe 2 (float, requis)
    
    Retourne:
    {
        "success": true/false,
        "model_used": "ml" ou "fallback",
        "prediction": "1" ou "N" ou "2",
        "confidence": 0.0-100.0,
        "probabilities": {
            "1": 0.0-100.0,
            "N": 0.0-100.0,
            "2": 0.0-100.0
        },
        "error": "message d'erreur" (si success=false)
    }
    """
    # Récupération des paramètres (GET ou POST)
    if request.method == "POST":
        odds_1 = request.json.get("odds_1") if request.is_json else request.form.get("odds_1")
        odds_x = request.json.get("odds_x") if request.is_json else request.form.get("odds_x")
        odds_2 = request.json.get("odds_2") if request.is_json else request.form.get("odds_2")
    else:
        odds_1 = request.args.get("odds_1")
        odds_x = request.args.get("odds_x")
        odds_2 = request.args.get("odds_2")
    
    # Validation des paramètres
    if not all([odds_1, odds_x, odds_2]):
        return jsonify({
            "success": False,
            "error": "Paramètres manquants. Requis: odds_1, odds_x, odds_2"
        }), 400
    
    # Conversion en float
    try:
        odds_1 = float(odds_1)
        odds_x = float(odds_x)
        odds_2 = float(odds_2)
    except (ValueError, TypeError):
        return jsonify({
            "success": False,
            "error": "Les cotes doivent être des nombres valides"
        }), 400
    
    # Validation des valeurs (cotes doivent être > 1.0)
    if any(odd <= 1.0 for odd in [odds_1, odds_x, odds_2]):
        return jsonify({
            "success": False,
            "error": "Les cotes doivent être supérieures à 1.0"
        }), 400
    
    # Utilisation du modèle ML si disponible, sinon fallback
    model_used = "ml" if _MODEL is not None else "fallback"
    result = ai_predict(odds_1, odds_x, odds_2)
    
    # Si le modèle n'est pas disponible, on informe l'utilisateur
    if _MODEL is None:
        return jsonify({
            "success": True,
            "model_used": "fallback",
            "message": "Modèle ML non encore entraîné. Utilisation des probabilités implicites.",
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "probabilities": result["probs"]
        })
    
    # Modèle ML utilisé
    return jsonify({
        "success": True,
        "model_used": "ml",
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "probabilities": result["probs"],
        "odds_input": {
            "odds_1": odds_1,
            "odds_x": odds_x,
            "odds_2": odds_2
        }
    })


@app.route("/tasks/collect", methods=["POST", "GET"])
def task_collect():
    """
    Endpoint à appeler depuis un service de cron externe (ex: cron-job.org)
    pour lancer la collecte et l'append dans data/matches.csv.
    Protéger avec le paramètre ?token=... ou l'entête X-Task-Token.
    """
    if not _check_task_token():
        return jsonify({"ok": False, "error": "unauthorized"}), 401

    append_matches_to_csv()
    return jsonify({"ok": True})


@app.route("/tasks/train", methods=["POST", "GET"])
def task_train():
    """
    Endpoint à appeler depuis un service de cron externe pour entraîner
    le modèle et rafraîchir model.joblib, puis recharger le modèle en mémoire.
    """
    global _MODEL
    if not _check_task_token():
        return jsonify({"ok": False, "error": "unauthorized"}), 401

    train_and_save_model()
    _MODEL = _load_model()
    return jsonify({"ok": True})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
