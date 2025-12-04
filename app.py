import os
from typing import Any, Dict, List, Optional, Tuple

import requests
from flask import Flask, render_template, jsonify

app = Flask(__name__)

API_URL = (
    "https://1xbet.com/service-api/LiveFeed/Get1x2_VZip"
    "?sports=85&count=40&lng=fr&gr=285&mode=4&country=96"
    "&getEmpty=true&virtualSports=true&noFilterBlockEvent=true"
)


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
    Pseudo-IA : transforme les cotes en probabilités implicites
    et choisit l’issue avec la probabilité la plus élevée.
    """
    if not all([odds_1, odds_x, odds_2]):
        return {"prediction": "indécis", "confidence": 0.0, "probs": {}}

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
    # On envoie une première liste pour ne pas avoir une page vide
    matches = fetch_matches()
    return render_template("index.html", matches=matches)


@app.route("/api/matches")
def api_matches():
    # Endpoint JSON utilisé par l’interface dynamique (AJAX)
    return jsonify(fetch_matches())


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
