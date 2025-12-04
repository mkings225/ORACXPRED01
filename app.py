import requests
from flask import Flask, render_template, jsonify

app = Flask(__name__)

API_URL = "https://1xbet.com/service-api/LiveFeed/Get1x2_VZip?sports=85&count=40&lng=fr&gr=285&mode=4&country=96&getEmpty=true&virtualSports=true&noFilterBlockEvent=true"


def extract_1x2_odds(event):
    """Extrait les cotes 1 / N / 2 depuis le champ E."""
    odds_1 = odds_x = odds_2 = None
    for m in event.get("E", []):
        if m.get("G") == 1:  # marché 1X2
            if m.get("T") == 1:
                odds_1 = m.get("C")
            elif m.get("T") == 2:
                odds_x = m.get("C")
            elif m.get("T") == 3:
                odds_2 = m.get("C")
    return odds_1, odds_x, odds_2


def ai_predict(odds_1, odds_x, odds_2):
    """
    Pseudo-IA : transforme les cotes en probabilités implicites,
    choisit l’issue avec la plus forte proba.
    """
    if not all([odds_1, odds_x, odds_2]):
        return {"prediction": "indécis", "confidence": 0.0}

    inv1 = 1.0 / odds_1
    invx = 1.0 / odds_x
    inv2 = 1.0 / odds_2
    s = inv1 + invx + inv2

    p1 = inv1 / s
    px = invx / s
    p2 = inv2 / s

    probs = {"1": p1, "N": px, "2": p2}
    best_outcome = max(probs, key=probs.get)
    confidence = probs[best_outcome]

    return {"prediction": best_outcome, "confidence": round(confidence * 100, 1)}


def fetch_matches():
    resp = requests.get(API_URL, timeout=10)
    data = resp.json()
    events = data.get("Value", [])

    matches = []
    for ev in events:
        odds_1, odds_x, odds_2 = extract_1x2_odds(ev)
        ai = ai_predict(odds_1, odds_x, odds_2)

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
            }
        )
    return matches


@app.route("/")
def index():
    matches = fetch_matches()
    return render_template("index.html", matches=matches)


@app.route("/api/matches")
def api_matches():
    matches = fetch_matches()
    return jsonify(matches)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
