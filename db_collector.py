"""
Système de collecte et sauvegarde robuste avec PostgreSQL
Sauvegarde uniquement les matchs terminés avec scores disponibles
"""
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from models import Match, get_session_factory


API_URL = (
    "https://1xbet.com/service-api/LiveFeed/Get1x2_VZip"
    "?sports=85&count=40&lng=fr&gr=285&mode=4&country=96"
    "&getEmpty=true&virtualSports=true&noFilterBlockEvent=true"
)

SessionLocal = get_session_factory()


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


def compute_outcome(score1: Optional[int], score2: Optional[int]) -> str:
    """Retourne '1', 'N', '2' ou '' si le résultat n'est pas exploitable."""
    if score1 is None or score2 is None:
        return ""
    if score1 > score2:
        return "1"
    if score1 < score2:
        return "2"
    return "N"


def match_exists_in_db(session: Session, event_id: Optional[int], team1: str, team2: str, league: str) -> bool:
    """
    Vérifie si un match existe déjà dans la base de données.
    Utilise l'event_id si disponible, sinon compare les équipes et la ligue.
    """
    # Si on a un event_id, on compare par ID
    if event_id is not None:
        existing = session.query(Match).filter(Match.event_id == event_id).first()
        if existing:
            return True
    
    # Sinon, on compare par équipes et ligue (sur les 24 dernières heures pour éviter les faux positifs)
    from datetime import timedelta
    yesterday = datetime.utcnow() - timedelta(days=1)
    
    existing = session.query(Match).filter(
        Match.team1 == team1.strip(),
        Match.team2 == team2.strip(),
        Match.league == league.strip(),
        Match.timestamp_utc >= yesterday
    ).first()
    
    return existing is not None


def fetch_events() -> List[Dict[str, Any]]:
    """Récupère les événements depuis l'API 1xBet avec gestion d'erreurs améliorée."""
    try:
        resp = requests.get(API_URL, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        events = data.get("Value", [])
        if not events:
            print(f"[COLLECTOR] ⚠️ Aucun événement récupéré depuis l'API")
        return events
    except requests.exceptions.ConnectionError as e:
        error_msg = f"Erreur de connexion à l'API 1xBet: {str(e)}"
        print(f"[COLLECTOR] ❌ {error_msg}")
        raise ConnectionError(error_msg) from e
    except requests.exceptions.Timeout as e:
        error_msg = f"Timeout lors de la connexion à l'API 1xBet: {str(e)}"
        print(f"[COLLECTOR] ❌ {error_msg}")
        raise TimeoutError(error_msg) from e
    except requests.exceptions.RequestException as e:
        error_msg = f"Erreur lors de la requête à l'API 1xBet: {str(e)}"
        print(f"[COLLECTOR] ❌ {error_msg}")
        raise RuntimeError(error_msg) from e
    except Exception as e:
        error_msg = f"Erreur inattendue lors de la récupération des événements: {str(e)}"
        print(f"[COLLECTOR] ❌ {error_msg}")
        traceback.print_exc()
        raise


def save_matches_to_db() -> None:
    """
    Système de collecte et sauvegarde ROBUSTE.
    Collecte les matchs TERMINÉS et les sauvegarde dans PostgreSQL.
    Ne sauvegarde QUE les matchs qui :
    - Ont un statut indiquant que le match est terminé
    - Ont des scores disponibles (score1 et score2)
    - N'existent pas déjà dans la base de données (évite les doublons)
    """
    session: Optional[Session] = None
    try:
        # Créer une session de base de données
        session = SessionLocal()
        
        # Tenter de récupérer les événements
        try:
            events = fetch_events()
        except Exception as fetch_error:
            print(f"[COLLECTOR] ATTENTION: Impossible de recuperer les evenements: {fetch_error}")
            print("[COLLECTOR] INFO: Aucune nouvelle donnee n'a ete ajoutee")
            return

        if not events:
            print("[COLLECTOR] ATTENTION: Aucun evenement a traiter")
            return

        utc_now = datetime.now(timezone.utc)
        matches_added = 0
        matches_skipped_not_finished = 0
        matches_skipped_duplicate = 0
        matches_error = 0

        for ev in events:
            try:
                event_id = ev.get("I")
                team1 = ev.get("O1", "").strip()
                team2 = ev.get("O2", "").strip()
                league = ev.get("L", "").strip()
                
                if not team1 or not team2:
                    continue
                
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
                
                # Vérifier si le match existe déjà dans la base de données
                if match_exists_in_db(session, event_id, team1, team2, league):
                    matches_skipped_duplicate += 1
                    continue
                
                # Le match est terminé et n'existe pas encore : on le sauvegarde
                outcome = compute_outcome(score1, score2)

                match = Match(
                    timestamp_utc=utc_now,
                    event_id=event_id,
                    league=league,
                    team1=team1,
                    team2=team2,
                    odds_1=odds_1,
                    odds_x=odds_x,
                    odds_2=odds_2,
                    score1=score1,
                    score2=score2,
                    status=status,
                    outcome=outcome,
                )

                session.add(match)
                session.commit()
                
                matches_added += 1
                print(f"[COLLECTOR] OK Match termine sauvegarde: {team1} vs {team2} ({score1}-{score2}) [ID: {match.id}]")
                
            except IntegrityError as e:
                # Doublon détecté au niveau de la base de données
                session.rollback()
                matches_skipped_duplicate += 1
                print(f"[COLLECTOR] ATTENTION: Doublon detecte: {team1} vs {team2}")
            except Exception as e:
                session.rollback()
                matches_error += 1
                print(f"[COLLECTOR] ATTENTION: Erreur lors du traitement d'un evenement: {e}")
                traceback.print_exc()
                continue

        print(f"[COLLECTOR] RESUME de la collecte:")
        print(f"  OK {matches_added} match(s) termine(s) sauvegarde(s) dans la base de donnees")
        print(f"  EN ATTENTE {matches_skipped_not_finished} match(s) ignore(s) (pas encore termine)")
        print(f"  DOUBLON {matches_skipped_duplicate} match(s) ignore(s) (deja dans la base)")
        if matches_error > 0:
            print(f"  ERREUR {matches_error} match(s) avec erreur(s)")
        
    except Exception as e:
        error_msg = f"Erreur lors de la collecte des matchs: {str(e)}"
        print(f"[COLLECTOR] ❌ {error_msg}")
        traceback.print_exc()
        if session:
            session.rollback()
        raise
    finally:
        if session:
            session.close()


if __name__ == "__main__":
    save_matches_to_db()

