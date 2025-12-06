"""
Script de migration des données CSV vers PostgreSQL
Migre toutes les données existantes du fichier CSV vers la base de données PostgreSQL
"""
import csv
import os
import traceback
from pathlib import Path
from datetime import datetime

from models import Match, get_session_factory, init_db
from db_collector import is_match_finished, compute_outcome

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "data" / "matches.csv"


def migrate_csv_to_postgresql():
    """Migre les données du CSV vers PostgreSQL."""
    print("[MIGRATION] 🔄 Début de la migration CSV → PostgreSQL")
    
    # Initialiser la base de données si nécessaire
    try:
        init_db()
        print("[MIGRATION] ✅ Base de données initialisée")
    except Exception as e:
        print(f"[MIGRATION] ⚠️ Erreur lors de l'initialisation: {e}")
        return
    
    if not CSV_PATH.exists():
        print(f"[MIGRATION] ⚠️ Fichier CSV introuvable: {CSV_PATH}")
        print("[MIGRATION] ℹ️ Aucune migration nécessaire")
        return
    
    SessionLocal = get_session_factory()
    session = SessionLocal()
    
    try:
        # Lire le CSV
        print(f"[MIGRATION] 📖 Lecture du fichier CSV: {CSV_PATH}")
        matches_added = 0
        matches_skipped = 0
        matches_error = 0
        
        with open(CSV_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row_num, row in enumerate(reader, start=2):  # start=2 car ligne 1 = header
                try:
                    # Vérifier que le match est terminé (avec scores)
                    score1 = int(row['score1']) if row.get('score1') and row['score1'].strip() else None
                    score2 = int(row['score2']) if row.get('score2') and row['score2'].strip() else None
                    status = row.get('status', '')
                    outcome = row.get('outcome', '').strip()
                    
                    # Ne migrer que les matchs terminés avec résultat
                    if not is_match_finished(status, score1, score2) or not outcome:
                        matches_skipped += 1
                        continue
                    
                    # Vérifier si le match existe déjà
                    event_id = int(row['event_id']) if row.get('event_id') and row['event_id'].strip() else None
                    team1 = row.get('team1', '').strip()
                    team2 = row.get('team2', '').strip()
                    league = row.get('league', '').strip()
                    
                    if event_id:
                        existing = session.query(Match).filter(Match.event_id == event_id).first()
                    else:
                        # Comparer par équipes et ligue
                        existing = session.query(Match).filter(
                            Match.team1 == team1,
                            Match.team2 == team2,
                            Match.league == league
                        ).first()
                    
                    if existing:
                        matches_skipped += 1
                        continue
                    
                    # Créer le match
                    match = Match(
                        timestamp_utc=datetime.fromisoformat(row['timestamp_utc'].replace('Z', '+00:00')) if row.get('timestamp_utc') else datetime.utcnow(),
                        event_id=event_id,
                        league=league,
                        team1=team1,
                        team2=team2,
                        odds_1=float(row['odds_1']) if row.get('odds_1') and row['odds_1'].strip() else None,
                        odds_x=float(row['odds_x']) if row.get('odds_x') and row['odds_x'].strip() else None,
                        odds_2=float(row['odds_2']) if row.get('odds_2') and row['odds_2'].strip() else None,
                        score1=score1,
                        score2=score2,
                        status=status,
                        outcome=outcome,
                    )
                    
                    session.add(match)
                    matches_added += 1
                    
                    # Commit par batch de 100 pour performance
                    if matches_added % 100 == 0:
                        session.commit()
                        print(f"[MIGRATION] ✅ {matches_added} matchs migrés...")
                    
                except Exception as e:
                    matches_error += 1
                    print(f"[MIGRATION] ⚠️ Erreur ligne {row_num}: {e}")
                    continue
        
        # Commit final
        session.commit()
        
        print(f"[MIGRATION] 📊 Résumé de la migration:")
        print(f"  ✅ {matches_added} match(s) migré(s) avec succès")
        print(f"  ⏭️ {matches_skipped} match(s) ignoré(s) (doublons ou non terminés)")
        if matches_error > 0:
            print(f"  ❌ {matches_error} match(s) avec erreur(s)")
        print(f"[MIGRATION] ✅ Migration terminée avec succès!")
        
    except Exception as e:
        session.rollback()
        print(f"[MIGRATION] ❌ Erreur lors de la migration: {e}")
        traceback.print_exc()
        raise
    finally:
        session.close()


if __name__ == "__main__":
    migrate_csv_to_postgresql()

