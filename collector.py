import csv
import os
import traceback
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
    """Récupère les événements depuis l'API 1xBet avec gestion d'erreurs améliorée."""
    try:
    resp = requests.get(API_URL, timeout=10)
    resp.raise_for_status()
    data = resp.json()
        events = data.get("Value", [])
        if not events:
            print(f"[COLLECTOR] ATTENTION: Aucun evenement recupere depuis l'API")
        return events
    except requests.exceptions.ConnectionError as e:
        error_msg = f"Erreur de connexion a l'API 1xBet: {str(e)}"
        print(f"[COLLECTOR] ERREUR: {error_msg}")
        raise ConnectionError(error_msg) from e
    except requests.exceptions.Timeout as e:
        error_msg = f"Timeout lors de la connexion a l'API 1xBet: {str(e)}"
        print(f"[COLLECTOR] ERREUR: {error_msg}")
        raise TimeoutError(error_msg) from e
    except requests.exceptions.RequestException as e:
        error_msg = f"Erreur lors de la requete a l'API 1xBet: {str(e)}"
        print(f"[COLLECTOR] ERREUR: {error_msg}")
        raise RuntimeError(error_msg) from e
    except Exception as e:
        error_msg = f"Erreur inattendue lors de la recuperation des evenements: {str(e)}"
        print(f"[COLLECTOR] ERREUR: {error_msg}")
        traceback.print_exc()
        raise


def is_match_finished(status: str, score1: Optional[int], score2: Optional[int]) -> bool:
    """
    Vérifie si un match est terminé.
    Un match est considéré comme terminé si :
    - Les scores sont disponibles (score1 et score2 non None)
    - ET (le statut contient des mots-clés indiquant la fin OU le statut est vide/None mais les scores existent)
    
    STRATÉGIE AMÉLIORÉE:
    - Si les scores existent ET le statut indique "terminé" -> TERMINÉ
    - Si les scores existent ET le statut est vide/None -> TERMINÉ (scores = match fini)
    - Si les scores existent ET le statut ne contient pas "live", "en cours", "playing" -> TERMINÉ
    """
    # Pas de scores = match pas terminé
    if score1 is None or score2 is None:
        return False
    
    # Si on a des scores, on vérifie le statut
    status_lower = (status or "").lower().strip()
    
    # Mots-clés indiquant que le match est EN COURS (donc pas terminé)
    live_keywords = ["live", "en cours", "playing", "ongoing", "in progress", "started", "commencé"]
    if any(keyword in status_lower for keyword in live_keywords):
        return False
    
    # Mots-clés indiquant que le match est TERMINÉ
    finished_keywords = [
        "terminé", "terminé", "terminé", "finished", "fin", "end", "fini",
        "match terminé", "game finished", "ended", "final", "ft", "full time",
        "complete", "complet", "closed", "fermé"
    ]
    if any(keyword in status_lower for keyword in finished_keywords):
        return True
    
    # Si le statut est vide ou None mais qu'on a des scores, on considère que c'est terminé
    # (les scores ne sont généralement disponibles qu'après la fin du match)
    if not status_lower or status_lower in ["", "none", "null"]:
        return True
    
    # Par défaut, si on a des scores et que le statut n'indique pas "live", on considère terminé
    # C'est plus permissif pour capturer tous les matchs finis
    return True


def match_exists_in_csv(event_id: Optional[int], team1: str, team2: str, league: str) -> bool:
    """
    Vérifie si un match existe déjà dans le CSV.
    Utilise l'event_id si disponible, sinon compare les équipes et la ligue.
    """
    if not os.path.isfile(CSV_PATH):
        return False
    
    try:
        with open(CSV_PATH, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Si on a un event_id, on compare par ID
                if event_id is not None:
                    try:
                        row_event_id = row.get("event_id")
                        if row_event_id and str(row_event_id) == str(event_id):
                            return True
                    except (ValueError, TypeError):
                        pass
                
                # Sinon, on compare par équipes et ligue
                row_team1 = row.get("team1", "").strip()
                row_team2 = row.get("team2", "").strip()
                row_league = row.get("league", "").strip()
                
                if (row_team1 == team1.strip() and 
                    row_team2 == team2.strip() and 
                    row_league == league.strip()):
                    return True
    except Exception as e:
        print(f"[COLLECTOR] ⚠️ Erreur lors de la vérification des doublons: {e}")
        # En cas d'erreur, on considère que le match n'existe pas pour éviter de bloquer
    
    return False


def compute_outcome(score1: Optional[int], score2: Optional[int]) -> str:
    """Retourne '1', 'N', '2' ou '' si le résultat n'est pas exploitable."""
    if score1 is None or score2 is None:
        return ""
    if score1 > score2:
        return "1"
    if score1 < score2:
        return "2"
    return "N"


def ensure_csv_header() -> None:
    """S'assure que le fichier CSV existe avec l'en-tête si nécessaire."""
    ensure_data_dir()
    file_exists = os.path.isfile(CSV_PATH)

    if not file_exists:
        try:
            with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
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
            print("[COLLECTOR] ✅ Fichier CSV créé avec l'en-tête")
        except Exception as e:
            print(f"[COLLECTOR] ⚠️ Erreur lors de la création du fichier CSV: {e}")


def append_matches_to_csv() -> None:
    """
    Collecte les matchs TERMINÉS et les ajoute au fichier CSV.
    Ne sauvegarde QUE les matchs qui :
    - Ont un statut indiquant que le match est terminé
    - Ont des scores disponibles (score1 et score2)
    - N'existent pas déjà dans le CSV (évite les doublons)
    """
    try:
        # S'assurer que le fichier CSV existe avec l'en-tête
        ensure_csv_header()
        
        # Tenter de récupérer les événements
        try:
            events = fetch_events()
        except Exception as fetch_error:
            print(f"[COLLECTOR] ATTENTION: Impossible de recuperer les evenements: {fetch_error}")
            print("[COLLECTOR] INFO: Le fichier CSV existe mais aucune nouvelle donnee n'a ete ajoutee")
            raise  # Re-lancer l'erreur pour que le scheduler la capture

        if not events:
            print("[COLLECTOR] ATTENTION: Aucun evenement a traiter")
            return

        utc_now = datetime.now(timezone.utc).isoformat()
        matches_added = 0
        matches_skipped_not_finished = 0
        matches_skipped_duplicate = 0

        with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

        for ev in events:
                try:
                    event_id = ev.get("I")
                    team1 = ev.get("O1", "")
                    team2 = ev.get("O2", "")
                    league = ev.get("L", "")
                    
            odds_1, odds_x, odds_2 = extract_1x2_odds(ev)
            score1, score2, status = extract_score(ev)
                    
                    # Vérifier si le match est terminé
                    is_finished = is_match_finished(status, score1, score2)
                    if not is_finished:
                        matches_skipped_not_finished += 1
                        # Log détaillé pour debug
                        if score1 is not None and score2 is not None:
                            print(f"[COLLECTOR] DEBUG Match ignore (non termine): {team1} vs {team2} - Score: {score1}-{score2} - Statut: {status!r}")
                        continue
                    
                    # Vérifier si le match existe déjà dans le CSV
                    if match_exists_in_csv(event_id, team1, team2, league):
                        matches_skipped_duplicate += 1
                        continue
                    
                    # Le match est terminé et n'existe pas encore : on le sauvegarde
            outcome = compute_outcome(score1, score2)

            writer.writerow(
                [
                    utc_now,
                            event_id,  # identifiant de l'évènement chez 1xBet
                            league,
                            team1,
                            team2,
                    odds_1,
                    odds_x,
                    odds_2,
                    score1,
                    score2,
                    status,
                    outcome,
                ]
            )
                    matches_added += 1
                    print(f"[COLLECTOR] OK Match termine sauvegarde: {team1} vs {team2} ({score1}-{score2})")
                    
                except Exception as e:
                    print(f"[COLLECTOR] ATTENTION: Erreur lors du traitement d'un evenement: {e}")
                    continue

        print(f"[COLLECTOR] RESUME de la collecte:")
        print(f"  OK {matches_added} match(s) termine(s) ajoute(s) au fichier CSV")
        print(f"  EN ATTENTE {matches_skipped_not_finished} match(s) ignore(s) (pas encore termine)")
        print(f"  DOUBLON {matches_skipped_duplicate} match(s) ignore(s) (deja dans la base)")
        
    except Exception as e:
        error_msg = f"Erreur lors de la collecte des matchs: {str(e)}"
        print(f"[COLLECTOR] ERREUR: {error_msg}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    append_matches_to_csv()


