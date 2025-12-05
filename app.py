import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import requests
from apscheduler.schedulers.background import BackgroundScheduler
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

# Scheduler pour les tâches automatiques
scheduler = BackgroundScheduler(daemon=True)

# Tracking des dernières exécutions
from datetime import datetime
_last_collect_time = None
_last_train_time = None
_collect_count = 0
_train_count = 0
_last_collect_error = None
_last_train_error = None


def job_collect():
    """Tâche planifiée : collecte des matchs."""
    global _last_collect_time, _collect_count, _last_collect_error
    try:
        append_matches_to_csv()
        _last_collect_time = datetime.now().isoformat()
        _collect_count += 1
        _last_collect_error = None
        print(f"[SCHEDULER] ✅ Collecte #{_collect_count} effectuée avec succès à {_last_collect_time}")
    except Exception as e:
        _last_collect_error = str(e)
        print(f"[SCHEDULER] ❌ Erreur lors de la collecte #{_collect_count + 1}: {e}")


def job_train():
    """Tâche planifiée : entraînement du modèle."""
    global _MODEL, _last_train_time, _train_count, _last_train_error
    try:
        train_and_save_model()
        _MODEL = _load_model()
        _last_train_time = datetime.now().isoformat()
        _train_count += 1
        _last_train_error = None
        print(f"[SCHEDULER] ✅ Modèle #{_train_count} entraîné et rechargé avec succès à {_last_train_time}")
        print(f"[SCHEDULER] 📊 Modèle ML maintenant actif: {_MODEL is not None}")
    except Exception as e:
        _last_train_error = str(e)
        print(f"[SCHEDULER] ❌ Erreur lors de l'entraînement #{_train_count + 1}: {e}")


# Configuration des tâches planifiées
# Collecte toutes les 5 minutes
scheduler.add_job(
    func=job_collect,
    trigger="interval",
    minutes=5,
    id="collect_job",
    name="Collecte des matchs",
    replace_existing=True,
)

# Entraînement tous les jours à 3h du matin
scheduler.add_job(
    func=job_train,
    trigger="cron",
    hour=3,
    minute=0,
    id="train_job",
    name="Entraînement du modèle",
    replace_existing=True,
)

# Démarrer le scheduler
scheduler.start()
print("[SCHEDULER] Tâches planifiées démarrées:")
print("  - Collecte: toutes les 5 minutes")
print("  - Entraînement: tous les jours à 3h00")


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


@app.route("/")
def index() -> str:
    """Page principale : liste des matchs."""
    matches = fetch_matches()
    return render_template("matches.html", matches=matches)


@app.route("/predictions")
def predictions() -> str:
    """Page dédiée aux prédictions IA avec détails."""
    matches = fetch_matches()
    return render_template("predictions.html", matches=matches)


@app.route("/api/matches")
def api_matches():
    """
    API : Liste de tous les matchs avec prédictions.
    Retourne un tableau JSON de matchs.
    """
    try:
        matches = fetch_matches()
        return jsonify(matches)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


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
            "probabilities": result["probs"],
            "note": "Pour activer le modèle ML, attendez l'entraînement automatique (3h00) ou lancez /tasks/train manuellement"
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
    Endpoint pour lancer la collecte manuellement.
    Peut être appelé avec ?token=... pour sécurité (optionnel si TASK_TOKEN non défini).
    """
    # Vérification du token si défini, sinon autorisé
    expected_token = os.environ.get("TASK_TOKEN")
    if expected_token:
        provided = request.args.get("token") or request.headers.get("X-Task-Token")
        if provided != expected_token:
            return jsonify({"ok": False, "error": "unauthorized"}), 401

    try:
        append_matches_to_csv()
        return jsonify({"ok": True, "message": "Collecte effectuée avec succès"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/tasks/train", methods=["POST", "GET"])
def task_train():
    """
    Endpoint pour lancer l'entraînement manuellement.
    Peut être appelé avec ?token=... pour sécurité (optionnel si TASK_TOKEN non défini).
    """
    global _MODEL
    # Vérification du token si défini, sinon autorisé
    expected_token = os.environ.get("TASK_TOKEN")
    if expected_token:
        provided = request.args.get("token") or request.headers.get("X-Task-Token")
        if provided != expected_token:
            return jsonify({"ok": False, "error": "unauthorized"}), 401

    try:
        train_and_save_model()
        _MODEL = _load_model()
        return jsonify({"ok": True, "message": "Modèle entraîné et rechargé avec succès"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/health")
def api_health():
    """API de santé : vérifie que le service fonctionne."""
    return jsonify({
        "status": "ok",
        "model_loaded": _MODEL is not None,
        "scheduler_running": scheduler.running if scheduler else False,
    })


@app.route("/api/stats")
def api_stats():
    """API : Statistiques du système."""
    from pathlib import Path
    
    data_path = Path(__file__).parent / "data" / "matches.csv"
    model_path = Path(__file__).parent / "model.joblib"
    
    csv_exists = data_path.exists()
    csv_size = data_path.stat().st_size if csv_exists else 0
    
    model_exists = model_path.exists()
    model_size = model_path.stat().st_size if model_exists else 0
    
    # Compter les lignes dans le CSV (sans le header)
    csv_lines = 0
    if csv_exists:
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                csv_lines = sum(1 for line in f) - 1  # -1 pour le header
        except:
            pass
    
    return jsonify({
        "model": {
            "loaded": _MODEL is not None,
            "file_exists": model_exists,
            "file_size": model_size,
            "prediction_method": "ML (modèle entraîné)" if _MODEL is not None else "Mathématique (probabilités implicites)"
        },
        "data": {
            "file_exists": csv_exists,
            "file_size": csv_size,
            "matches_count": csv_lines
        },
        "scheduler": {
            "running": scheduler.running if scheduler else False,
            "collect": {
                "count": _collect_count,
                "last_time": _last_collect_time,
                "last_error": _last_collect_error
            },
            "train": {
                "count": _train_count,
                "last_time": _last_train_time,
                "last_error": _last_train_error
            }
        }
    })


@app.route("/api/scheduler")
def api_scheduler():
    """API : État détaillé du scheduler."""
    jobs_info = []
    if scheduler and scheduler.running:
        for job in scheduler.get_jobs():
            jobs_info.append({
                "id": job.id,
                "name": job.name,
                "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
                "trigger": str(job.trigger)
            })
    
    return jsonify({
        "running": scheduler.running if scheduler else False,
        "jobs": jobs_info,
        "collect_stats": {
            "total_executions": _collect_count,
            "last_execution": _last_collect_time,
            "last_error": _last_collect_error
        },
        "train_stats": {
            "total_executions": _train_count,
            "last_execution": _last_train_time,
            "last_error": _last_train_error
        }
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
