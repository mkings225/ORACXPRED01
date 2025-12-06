# 🗄️ Guide de Configuration de la Base de Données PostgreSQL

## 📋 Prérequis

1. **PostgreSQL installé** (version 12+ recommandée)
2. **Python packages** : `pip install -r requirements.txt`

## 🚀 Installation Rapide

### 1. Installer PostgreSQL

**Windows:**
- Télécharger depuis https://www.postgresql.org/download/windows/
- Installer avec les paramètres par défaut
- Noter le mot de passe du superutilisateur `postgres`

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install postgresql postgresql-contrib
```

**macOS:**
```bash
brew install postgresql
brew services start postgresql
```

### 2. Créer la Base de Données

```sql
-- Se connecter à PostgreSQL
psql -U postgres

-- Créer la base de données
CREATE DATABASE oracxpred;

-- Créer un utilisateur (optionnel mais recommandé)
CREATE USER oracxpred WITH PASSWORD 'oracxpred123';
GRANT ALL PRIVILEGES ON DATABASE oracxpred TO oracxpred;
```

### 3. Configurer la Variable d'Environnement

**Windows (PowerShell):**
```powershell
$env:DATABASE_URL = "postgresql://oracxpred:oracxpred123@localhost:5432/oracxpred"
```

**Linux/macOS:**
```bash
export DATABASE_URL="postgresql://oracxpred:oracxpred123@localhost:5432/oracxpred"
```

**Ou créer un fichier `.env`:**
```
DATABASE_URL=postgresql://oracxpred:oracxpred123@localhost:5432/oracxpred
```

### 4. Initialiser la Base de Données

```bash
python setup_database.py
```

### 5. Migrer les Données CSV Existantes (si applicable)

```bash
python migrate_csv_to_db.py
```

## ✅ Vérification

Vérifier que tout fonctionne :

```bash
python -c "from models import init_db; init_db(); print('✅ Base de données OK')"
```

## 🔧 Structure de la Base de Données

### Table `matches`
- **id** : Identifiant unique (auto-increment)
- **timestamp_utc** : Date/heure de collecte
- **event_id** : ID de l'événement chez 1xBet
- **league** : Nom de la ligue
- **team1**, **team2** : Noms des équipes
- **odds_1**, **odds_x**, **odds_2** : Cotes
- **score1**, **score2** : Scores finaux
- **status** : Statut du match
- **outcome** : Résultat ('1', 'N', '2')
- **created_at**, **updated_at** : Timestamps automatiques

### Table `model_versions`
- Versioning des modèles ML entraînés
- Métriques de performance
- Modèle actif

### Table `training_logs`
- Logs de tous les entraînements
- Traçabilité complète

## 🔄 Fonctionnement Automatique

1. **Collecte** : Toutes les 5 minutes, sauvegarde uniquement les matchs terminés
2. **Entraînement** : Tous les jours à 3h00, utilise TOUS les matchs de la base pour entraîner
3. **Pas de doublons** : Détection automatique des matchs déjà sauvegardés

## 🛠️ Commandes Utiles

### Voir le nombre de matchs
```sql
SELECT COUNT(*) FROM matches;
```

### Voir les derniers matchs
```sql
SELECT * FROM matches ORDER BY timestamp_utc DESC LIMIT 10;
```

### Voir les statistiques
```sql
SELECT outcome, COUNT(*) 
FROM matches 
WHERE outcome IS NOT NULL 
GROUP BY outcome;
```

### Voir les modèles entraînés
```sql
SELECT version, accuracy, training_samples, created_at, is_active 
FROM model_versions 
ORDER BY created_at DESC;
```

## ⚠️ Dépannage

### Erreur de connexion
- Vérifier que PostgreSQL est démarré
- Vérifier l'URL de connexion (DATABASE_URL)
- Vérifier les permissions de l'utilisateur

### Erreur "relation does not exist"
- Exécuter `python setup_database.py` pour créer les tables

### Migration échoue
- Vérifier que le fichier CSV existe
- Vérifier les permissions de lecture du fichier

