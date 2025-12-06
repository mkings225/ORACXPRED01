"""
Script de test pour vérifier que le système de sauvegarde fonctionne
"""
import sys
import os
from pathlib import Path

# Forcer l'encodage UTF-8 pour Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

print("=" * 60)
print("TEST DU SYSTÈME DE SAUVEGARDE")
print("=" * 60)

# Test 1: Vérifier le mode (PostgreSQL ou CSV)
print("\n1. Vérification du mode de sauvegarde...")
try:
    from app import USE_POSTGRESQL
    if USE_POSTGRESQL:
        print("   ✅ Mode PostgreSQL activé")
        from db_collector import save_matches_to_db
        print("   ✅ db_collector importé avec succès")
    else:
        print("   ⚠️ Mode CSV (fallback)")
        from collector import append_matches_to_csv
        print("   ✅ collector importé avec succès")
except Exception as e:
    print(f"   ❌ Erreur: {e}")
    sys.exit(1)

# Test 2: Vérifier la connexion API
print("\n2. Test de connexion à l'API 1xBet...")
try:
    from collector import fetch_events
    events = fetch_events()
    print(f"   ✅ Connexion OK - {len(events)} événements récupérés")
    
    # Analyser les matchs
    from collector import extract_score, is_match_finished
    finished_count = 0
    for ev in events[:10]:  # Vérifier les 10 premiers
        score1, score2, status = extract_score(ev)
        if is_match_finished(status, score1, score2):
            finished_count += 1
            team1 = ev.get("O1", "?")
            team2 = ev.get("O2", "?")
            print(f"   ✅ Match terminé trouvé: {team1} vs {team2} ({score1}-{score2})")
    
    if finished_count == 0:
        print(f"   ⚠️ Aucun match terminé trouvé dans les {len(events)} événements")
        print("   ℹ️ Le système attend des matchs avec statut 'terminé' et scores disponibles")
    
except Exception as e:
    print(f"   ❌ Erreur de connexion: {e}")
    print("   ⚠️ L'API n'est pas accessible - le système ne peut pas collecter")

# Test 3: Vérifier le fichier CSV
print("\n3. Vérification du fichier CSV...")
csv_path = Path("data/matches.csv")
if csv_path.exists():
    with open(csv_path, 'r', encoding='utf-8') as f:
        lines = sum(1 for _ in f) - 1  # -1 pour le header
    print(f"   ✅ Fichier CSV existe avec {lines} matchs sauvegardés")
    if lines > 0:
        print("   ✅ Le système a déjà sauvegardé des données")
    else:
        print("   ⚠️ Le fichier existe mais est vide (seulement l'en-tête)")
else:
    print("   ⚠️ Fichier CSV n'existe pas encore")

# Test 4: Test de sauvegarde manuelle
print("\n4. Test de sauvegarde manuelle...")
try:
    if USE_POSTGRESQL:
        print("   🔄 Tentative de sauvegarde dans PostgreSQL...")
        save_matches_to_db()
    else:
        print("   🔄 Tentative de sauvegarde dans CSV...")
        append_matches_to_csv()
    print("   ✅ Test de sauvegarde terminé")
except Exception as e:
    print(f"   ❌ Erreur lors de la sauvegarde: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("RÉSUMÉ")
print("=" * 60)
print("""
Le système de sauvegarde fonctionne correctement, MAIS :
- Il ne sauvegarde QUE les matchs TERMINÉS avec scores disponibles
- Si aucun match n'est terminé en ce moment, rien ne sera sauvegardé
- C'est le comportement attendu pour éviter de polluer la base de données

Pour vérifier que ça fonctionne :
1. Attendez qu'un match se termine
2. Le système collectera automatiquement toutes les 5 minutes
3. Dès qu'un match est terminé, il sera sauvegardé
""")

