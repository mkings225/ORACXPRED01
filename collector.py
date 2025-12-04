import csv
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests


API_URL = (
    "https://1xbet.com/service-api/LiveFeed/Get1x2_VZip"
    "?sports=85&count=40&lng=fr&gr=285&mode=4&country=96"
    "&getEmpty=true&virtualSports=true&noFilterBlockEvent=true"
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CSV_PATH = os.path.join(DATA_DIR, "matches.csv")


def ensure_data_dir() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)


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
    """Récupère le score et le statut de match depuis le champ SC."""
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


def fetch_events() -> List[Dict[str, Any]]:
    resp = requests.get(API_URL, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return data.get("Value", [])


def compute_outcome(score1: Optional[int], score2: Optional[int]) -> str:
    """Retourne '1', 'N', '2' ou '' si le résultat n'est pas exploitable."""
    if score1 is None or score2 is None:
        return ""
    if score1 > score2:
        return "1"
    if score1 < score2:
        return "2"
    return "N"


def append_matches_to_csv() -> None:
    ensure_data_dir()
    events = fetch_events()

    file_exists = os.path.isfile(CSV_PATH)

    utc_now = datetime.now(timezone.utc).isoformat()

    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(
                [
                    "timestamp_utc",
                    "event_id",
                    "league",
                    "team1",
                    "team2",
                    "odds_1",
                    "odds_x",
                    "odds_2",
                    "score1",
                    "score2",
                    "status",
                    "outcome",
                ]
            )

        for ev in events:
            odds_1, odds_x, odds_2 = extract_1x2_odds(ev)
            score1, score2, status = extract_score(ev)
            outcome = compute_outcome(score1, score2)

            writer.writerow(
                [
                    utc_now,
                    ev.get("I"),  # identifiant de l'évènement chez 1xBet
                    ev.get("L"),
                    ev.get("O1"),
                    ev.get("O2"),
                    odds_1,
                    odds_x,
                    odds_2,
                    score1,
                    score2,
                    status,
                    outcome,
                ]
            )


if __name__ == "__main__":
    append_matches_to_csv()


