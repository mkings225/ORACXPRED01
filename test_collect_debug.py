"""
Script de diagnostic pour comprendre pourquoi les matchs ne sont pas sauvegardés
"""
import sys
from collector import fetch_events, extract_score, is_match_finished, extract_1x2_odds, match_exists_in_csv
from db_collector import save_matches_to_db, is_match_finished as db_is_finished
import traceback

print("=" * 80)
print("DIAGNOSTIC DU SYSTEME DE COLLECTE")
print("=" * 80)

# Test 1: Récupération des événements
print("\n[TEST 1] Récupération des événements depuis l'API...")
try:
    events = fetch_events()
    print(f"OK {len(events)} evenements recuperes")
except Exception as e:
    print(f"ERREUR: {e}")
    traceback.print_exc()
    sys.exit(1)

if not events:
    print("⚠️ Aucun événement récupéré. L'API ne retourne rien.")
    sys.exit(1)

# Test 2: Analyse des statuts et scores
print("\n[TEST 2] Analyse des statuts et scores des matchs...")
finished_count = 0
with_scores = 0
no_scores = 0
statuses = {}

for i, ev in enumerate(events[:20]):  # Analyser les 20 premiers
    try:
        event_id = ev.get("I")
        team1 = ev.get("O1", "")
        team2 = ev.get("O2", "")
        score1, score2, status = extract_score(ev)
        
        # Collecter les statuts uniques
        status_key = status.lower() if status else "vide"
        if status_key not in statuses:
            statuses[status_key] = 0
        statuses[status_key] += 1
        
        # Vérifier les scores
        has_scores = score1 is not None and score2 is not None
        if has_scores:
            with_scores += 1
        else:
            no_scores += 1
        
        # Vérifier si terminé
        is_finished = is_match_finished(status, score1, score2)
        if is_finished:
            finished_count += 1
            print(f"\n  [OK] MATCH TERMINE #{i+1}:")
            print(f"     {team1} vs {team2}")
            print(f"     Score: {score1}-{score2}")
            print(f"     Statut: {status!r}")
            print(f"     Event ID: {event_id}")
        elif has_scores:
            print(f"\n  [ATTENTION] MATCH AVEC SCORES MAIS NON DETECTE COMME TERMINE #{i+1}:")
            print(f"     {team1} vs {team2}")
            print(f"     Score: {score1}-{score2}")
            print(f"     Statut: {status!r}")
            print(f"     Event ID: {event_id}")
    except Exception as e:
        print(f"  [ERREUR] Erreur lors de l'analyse de l'evenement #{i+1}: {e}")

print(f"\n📊 RÉSUMÉ:")
print(f"   - Matchs avec scores: {with_scores}")
print(f"   - Matchs sans scores: {no_scores}")
print(f"   - Matchs détectés comme terminés: {finished_count}")
print(f"\n📋 STATUTS UNIQUES TROUVÉS:")
for status, count in sorted(statuses.items(), key=lambda x: -x[1]):
    print(f"   - '{status}': {count} fois")

# Test 3: Vérifier la fonction is_match_finished avec différents statuts
print("\n[TEST 3] Test de détection avec différents statuts...")
test_cases = [
    ("Finished", 2, 1, True),
    ("Terminé", 3, 0, True),
    ("Ended", 1, 1, True),
    ("Live", 1, 0, False),
    ("", 2, 1, False),  # Statut vide mais avec scores
    ("FT", 2, 1, True),  # Full Time
    ("FIN", 1, 1, True),
    ("En cours", 1, 0, False),
]

for status, s1, s2, expected in test_cases:
    result = is_match_finished(status, s1, s2)
    icon = "[OK]" if result == expected else "[ERREUR]"
    print(f"  {icon} Status='{status}', Score={s1}-{s2} -> {result} (attendu: {expected})")

# Test 4: Tentative de collecte réelle
print("\n[TEST 4] Tentative de collecte réelle...")
try:
    # Essayer PostgreSQL d'abord
    try:
        from models import get_session_factory
        SessionLocal = get_session_factory()
        session = SessionLocal()
        session.close()
        print("   Mode PostgreSQL détecté")
        save_matches_to_db()
    except:
        print("   Mode CSV détecté (fallback)")
        from collector import append_matches_to_csv
        append_matches_to_csv()
    print("   [OK] Collecte terminee")
except Exception as e:
    print(f"   [ERREUR] ERREUR lors de la collecte: {e}")
    traceback.print_exc()

print("\n" + "=" * 80)
print("FIN DU DIAGNOSTIC")
print("=" * 80)

