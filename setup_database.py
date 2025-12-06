"""
Script de configuration de la base de données PostgreSQL
Crée la base de données et les tables nécessaires
"""
import os
from models import init_db, get_database_url

print("=" * 60)
print("Configuration de la base de données PostgreSQL")
print("=" * 60)

# Afficher l'URL de la base de données
db_url = get_database_url()
print(f"\n📊 URL de la base de données: {db_url.replace(db_url.split('@')[0].split('//')[1] if '@' in db_url else '', '***@') if '@' in db_url else db_url}")

# Demander confirmation
response = input("\n⚠️  Cette opération va créer les tables dans la base de données. Continuer? (o/n): ")
if response.lower() != 'o':
    print("❌ Opération annulée")
    exit(0)

try:
    print("\n🔄 Création des tables...")
    init_db()
    print("\n✅ Base de données configurée avec succès!")
    print("\n📝 Prochaines étapes:")
    print("  1. Exécutez 'python migrate_csv_to_db.py' pour migrer les données CSV existantes")
    print("  2. Configurez la variable d'environnement DATABASE_URL si nécessaire")
    print("  3. Redémarrez l'application Flask")
except Exception as e:
    print(f"\n❌ Erreur lors de la configuration: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

