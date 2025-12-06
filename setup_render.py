"""
Script d'initialisation pour Render
Crée les tables de la base de données au démarrage
"""
import os
import sys

print("=" * 60)
print("Initialisation de la base de donnees pour Render")
print("=" * 60)

try:
    from models import init_db, get_database_url
    
    db_url = get_database_url()
    # Masquer le mot de passe dans les logs
    if '@' in db_url:
        parts = db_url.split('@')
        if '//' in parts[0]:
            user_pass = parts[0].split('//')[1]
            if ':' in user_pass:
                user = user_pass.split(':')[0]
                masked_url = db_url.replace(user_pass, f"{user}:***")
            else:
                masked_url = db_url
        else:
            masked_url = db_url
    else:
        masked_url = db_url
    
    print(f"\nBase de donnees: {masked_url}")
    
    print("\nCreation des tables...")
    init_db()
    print("\nOK Base de donnees initialisee avec succes!")
    
except Exception as e:
    print(f"\nERREUR lors de l'initialisation: {e}")
    import traceback
    traceback.print_exc()
    # Ne pas faire échouer le déploiement si la DB n'est pas encore créée
    # Render créera la DB automatiquement
    print("\nATTENTION: La base de donnees sera creee automatiquement par Render")
    sys.exit(0)

