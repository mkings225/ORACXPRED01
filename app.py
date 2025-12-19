import hashlib
import os
import threading
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import requests
from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask, jsonify, redirect, render_template, request

# Utiliser PostgreSQL au lieu de CSV
USE_POSTGRESQL = False
try:
    from db_collector import save_matches_to_db as append_matches_to_db
    from db_train_model import train_and_save_model, get_active_model_path
    from models import init_db, get_session_factory
    
    # Tester la connexion à PostgreSQL
    try:
        SessionLocal = get_session_factory()
        session = SessionLocal()
        session.close()
        USE_POSTGRESQL = True
        print("[APP] OK Mode PostgreSQL active et connecte")
        append_matches_to_csv = append_matches_to_db
    except Exception as db_error:
        # PostgreSQL installé mais non accessible
        USE_POSTGRESQL = False
        from collector import append_matches_to_csv
        from train_model import train_and_save_model
        print(f"[APP] ⚠️ Mode CSV (fallback) - PostgreSQL non accessible: {db_error}")
        print("[APP] ℹ️ Pour activer PostgreSQL, installez et démarrez le serveur PostgreSQL")
        
except ImportError as e:
    # Fallback vers CSV si les modules PostgreSQL ne sont pas installés
    from collector import append_matches_to_csv
    from train_model import train_and_save_model
    USE_POSTGRESQL = False
    print(f"[APP] ⚠️ Mode CSV (fallback) - Modules PostgreSQL non disponibles: {e}")
    print("[APP] ℹ️ Installez avec: pip install sqlalchemy psycopg2-binary")

import pandas as pd

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
# Scheduler en arrière-plan - fonctionne PERMANENTEMENT même sans utilisateurs
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
        print(f"[SCHEDULER] Demarrage de la collecte #{_collect_count + 1}...")
        append_matches_to_csv()
        _last_collect_time = datetime.now().isoformat()
        _collect_count += 1
        _last_collect_error = None
        print(f"[SCHEDULER] OK Collecte #{_collect_count} effectuee avec succes a {_last_collect_time}")
    except Exception as e:
        _last_collect_error = f"{str(e)}\n{traceback.format_exc()}"
        print(f"[SCHEDULER] ERREUR lors de la collecte #{_collect_count + 1}: {e}")
        traceback.print_exc()


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
        _last_train_error = f"{str(e)}\n{traceback.format_exc()}"
        print(f"[SCHEDULER] ❌ Erreur lors de l'entraînement #{_train_count + 1}: {e}")
        traceback.print_exc()


# Configuration des tâches planifiées
# Collecte toutes les 1 minute pour détection rapide des matchs terminés
scheduler.add_job(
    func=job_collect,
    trigger="interval",
    minutes=1,  # Réduit à 1 minute pour détection rapide
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

# Démarrer le scheduler avec gestion d'erreurs
try:
    if not scheduler.running:
        scheduler.start()
        print("[SCHEDULER] OK Taches planifiees demarrees avec succes:")
        print("  - Collecte: toutes les 1 minute (permanent, meme sans utilisateurs)")
        print("  - Entrainement: tous les jours a 3h00")
    else:
        print("[SCHEDULER] ATTENTION Scheduler deja en cours d'execution")
except Exception as e:
    print(f"[SCHEDULER] ERREUR lors du demarrage du scheduler: {e}")
    traceback.print_exc()


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
    """Récupère les matchs depuis l'API 1xBet et prépare les données pour l'UI."""
    try:
        resp = requests.get(API_URL, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        events = data.get("Value", [])

    matches: List[Dict[str, Any]] = []

    for ev in events:
        odds_1, odds_x, odds_2 = extract_1x2_odds(ev)
        ai = ai_predict(odds_1, odds_x, odds_2)
        score1, score2, status = extract_score(ev)

        # Générer un ID unique si l'ID de l'événement n'est pas disponible
        event_id = ev.get("I")
        if event_id is None:
            # Utiliser un hash basé sur les équipes et la ligue comme ID de secours
            match_str = f"{ev.get('L', '')}_{ev.get('O1', '')}_{ev.get('O2', '')}"
            event_id = int(hashlib.md5(match_str.encode()).hexdigest()[:8], 16)
        
        matches.append(
            {
                "id": event_id,
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
    except requests.exceptions.ConnectionError as e:
        print(f"[APP] ❌ Erreur de connexion à l'API 1xBet: {e}")
        return []  # Retourner une liste vide au lieu de planter
    except requests.exceptions.Timeout as e:
        print(f"[APP] ❌ Timeout lors de la connexion à l'API 1xBet: {e}")
        return []
    except requests.exceptions.RequestException as e:
        print(f"[APP] ❌ Erreur lors de la requête à l'API 1xBet: {e}")
        return []
    except Exception as e:
        print(f"[APP] ❌ Erreur inattendue lors de la récupération des matchs: {e}")
        traceback.print_exc()
        return []


def load_collected_matches() -> List[Dict[str, Any]]:
    """Charge les matchs collectés depuis PostgreSQL ou CSV (fallback)."""
    if USE_POSTGRESQL:
        try:
            from models import Match, get_session_factory
            SessionLocal = get_session_factory()
            session = SessionLocal()
            try:
                matches_db = session.query(Match).order_by(Match.timestamp_utc.desc()).all()
                matches = []
                for m in matches_db:
                    matches.append({
                        "timestamp": m.timestamp_utc.isoformat() if m.timestamp_utc else "",
                        "event_id": m.event_id,
                        "league": m.league or "",
                        "team1": m.team1 or "",
                        "team2": m.team2 or "",
                        "odds_1": m.odds_1,
                        "odds_x": m.odds_x,
                        "odds_2": m.odds_2,
                        "score1": m.score1,
                        "score2": m.score2,
                        "status": m.status or "",
                        "outcome": m.outcome or "",
                    })
                return matches
            finally:
                session.close()
        except Exception as e:
            print(f"[ERROR] Erreur lors du chargement depuis PostgreSQL: {e}")
            traceback.print_exc()
            return []
    else:
        # Fallback CSV
        csv_path = BASE_DIR / "data" / "matches.csv"
        if not csv_path.exists():
            return []
        try:
            df = pd.read_csv(csv_path)
            matches = []
            for _, row in df.iterrows():
                matches.append({
                    "timestamp": row.get("timestamp_utc", ""),
                    "event_id": row.get("event_id", ""),
                    "league": row.get("league", ""),
                    "team1": row.get("team1", ""),
                    "team2": row.get("team2", ""),
                    "odds_1": row.get("odds_1"),
                    "odds_x": row.get("odds_x"),
                    "odds_2": row.get("odds_2"),
                    "score1": row.get("score1"),
                    "score2": row.get("score2"),
                    "status": row.get("status", ""),
                    "outcome": row.get("outcome", ""),
                })
            matches.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            return matches
        except Exception as e:
            print(f"[ERROR] Erreur lors du chargement des matchs collectés: {e}")
            traceback.print_exc()
            return []


@app.route("/")
def index() -> str:
    """Page principale : liste des matchs."""
    try:
        matches = fetch_matches()
        # Si pas de matchs et erreur API, on peut afficher un message
        return render_template("matches.html", matches=matches)
    except Exception as e:
        print(f"[APP] ❌ Erreur dans la route index: {e}")
        traceback.print_exc()
        # Retourner une liste vide plutôt que de planter
        return render_template("matches.html", matches=[])


@app.route("/collected")
def collected_matches() -> str:
    """Page des matchs collectés depuis le CSV."""
    matches = load_collected_matches()
    total_count = len(matches)
    return render_template("collected.html", matches=matches, total_count=total_count)


@app.route("/predictions/<int:match_id>")
def prediction_detail(match_id: int):
    """Page de prédiction détaillée pour un match spécifique."""
    try:
        matches = fetch_matches()
        match = next((m for m in matches if m.get("id") == match_id), None)
        
        if not match:
            return render_template("prediction_not_found.html", match_id=match_id), 404
        
        return render_template("prediction_detail.html", match=match)
    except Exception as e:
        # Log l'erreur pour le debug
        print(f"[ERROR] Erreur dans prediction_detail: {e}")
        import traceback
        traceback.print_exc()
        return f"<h1>Erreur serveur</h1><p>{str(e)}</p><a href='/'>Retour</a>", 500


@app.route("/predictions")
def predictions_redirect():
    """Redirection vers la page des matchs si accès direct."""
    return redirect("/")


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
    Accessible publiquement pour usage manuel.
    Pour sécuriser, définir TASK_TOKEN dans les variables d'environnement.
    """
    # Vérification du token si défini, sinon autorisé (pour usage manuel facile)
    expected_token = os.environ.get("TASK_TOKEN")
    if expected_token:
        provided = request.args.get("token") or request.headers.get("X-Task-Token")
        if not provided or provided != expected_token:
            return jsonify({
                "ok": False, 
                "error": "unauthorized",
                "message": "Token requis. Utilisez ?token=VOTRE_TOKEN ou définissez TASK_TOKEN dans les variables d'environnement."
            }), 401

    try:
        append_matches_to_csv()
        return jsonify({"ok": True, "message": "Collecte effectuée avec succès"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/collect", methods=["POST", "GET"])
def collect_public():
    """
    Endpoint public pour lancer la collecte manuellement (sans token requis).
    Version simplifiée pour usage manuel depuis le navigateur.
    """
    try:
        append_matches_to_csv()
        return jsonify({"ok": True, "message": "Collecte effectuée avec succès"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/collect-debug")
def api_collect_debug():
    """
    Diagnostic de collecte :
    - Nombre d'événements récupérés
    - Statuts distincts
    - Combien détectés comme terminés
    - Combien ignorés faute de score ou statut live
    """
    try:
        # Importer les fonctions nécessaires depuis le collecteur approprié
        if USE_POSTGRESQL:
            from db_collector import fetch_events, is_match_finished
        else:
            from collector import fetch_events, is_match_finished
        
        events = fetch_events()
        total = len(events)

        status_counts: Dict[str, int] = {}
        finished = 0
        no_scores = 0
        live = 0
        finished_samples: List[Dict[str, Any]] = []
        ignored_samples: List[Dict[str, Any]] = []

        for ev in events:
            score1, score2, status = extract_score(ev)
            status_key = (status or "").lower() or "vide"
            status_counts[status_key] = status_counts.get(status_key, 0) + 1

            has_scores = score1 is not None and score2 is not None
            is_finished = is_match_finished(status, score1, score2)

            if is_finished:
                finished += 1
                if len(finished_samples) < 5:
                    finished_samples.append(
                        {
                            "teams": f"{ev.get('O1', '')} vs {ev.get('O2', '')}",
                            "score": f"{score1}-{score2}",
                            "status": status,
                            "event_id": ev.get("I"),
                        }
                    )
            else:
                if not has_scores:
                    no_scores += 1
                else:
                    live += 1
                if len(ignored_samples) < 5:
                    ignored_samples.append(
                        {
                            "teams": f"{ev.get('O1', '')} vs {ev.get('O2', '')}",
                            "score": f"{score1}-{score2}",
                            "status": status,
                            "event_id": ev.get("I"),
                        }
                    )

        return jsonify(
            {
                "ok": True,
                "events_total": total,
                "finished_detected": finished,
                "without_scores": no_scores,
                "live_or_in_progress": live,
                "status_counts": status_counts,
                "samples": {
                    "finished": finished_samples,
                    "ignored": ignored_samples,
                },
            }
        )
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


# Initialisation du scheduler au démarrage de l'application
# Cette fonction est appelée automatiquement par Flask dans certains environnements
def init_scheduler():
    """Initialise le scheduler si ce n'est pas déjà fait."""
    global scheduler
    try:
        if scheduler and not scheduler.running:
            scheduler.start()
            print("[SCHEDULER] ✅ Scheduler démarré via init_scheduler()")
    except Exception as e:
        print(f"[SCHEDULER] ⚠️ Erreur lors de l'initialisation du scheduler: {e}")
        traceback.print_exc()

# Hook Flask pour s'assurer que le scheduler démarre (compatible Flask 2.2+)
@app.before_request
def ensure_scheduler_running():
    """S'assure que le scheduler est démarré avant chaque requête (une seule fois)."""
    global scheduler
    if scheduler and not scheduler.running:
        try:
            scheduler.start()
            print("[SCHEDULER] ✅ Scheduler démarré via ensure_scheduler_running()")
        except Exception as e:
            print(f"[SCHEDULER] ⚠️ Erreur lors du démarrage du scheduler: {e}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # S'assurer que le scheduler est démarré avant de lancer Flask
    init_scheduler()
    app.run(host="0.0.0.0", port=port, debug=False)
