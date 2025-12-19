# Collecteur Externe - Service Autonome

Le **collecteur externe** est un service indépendant qui peut s'exécuter séparément de l'application Flask principale. Il permet de :

- ✅ Séparer la collecte de données de l'application web
- ✅ Fonctionner même si l'application Flask est arrêtée
- ✅ Déployer sur des serveurs différents si nécessaire
- ✅ Avoir un contrôle indépendant sur la planification
- ✅ Faciliter le scaling horizontal

## Installation

Aucune installation supplémentaire n'est nécessaire. Le service utilise les mêmes dépendances que l'application principale :

```bash
pip install -r requirements.txt
```

## Utilisation

### 1. Exécution Unique (One-Shot)

Exécute une collecte unique puis s'arrête :

```bash
python collector_service.py
```

### 2. Mode Daemon avec Planification Automatique

Exécute le service en arrière-plan avec planification automatique :

```bash
# Par défaut : collecte toutes les 1 minute
python collector_service.py --daemon

# Intervalle personnalisé (ex: toutes les 5 minutes)
python collector_service.py --daemon --interval 5

# Avec logs dans un fichier
python collector_service.py --daemon --interval 1 --log-file logs/collector_service.log
```

### 3. Scripts de Démarrage

#### Windows

```batch
# Lancer avec l'intervalle par défaut (1 minute)
collector_service.bat

# Lancer avec un intervalle personnalisé
collector_service.bat 5
```

#### Linux/Unix/Mac

```bash
# Rendre le script exécutable (première fois)
chmod +x collector_service.sh

# Lancer avec l'intervalle par défaut (1 minute)
./collector_service.sh

# Lancer avec un intervalle personnalisé
./collector_service.sh 5
```

### 4. Tester la Connexion à la Base de Données

```bash
python collector_service.py --test-db
```

### 5. Mode Verbose (Debug)

```bash
python collector_service.py --daemon --verbose
```

## Options en Ligne de Commande

```
--daemon          Exécuter en mode daemon avec planification automatique
--interval N      Intervalle entre les collectes en minutes (défaut: 1)
--log-file PATH   Chemin vers le fichier de log
--test-db         Tester la connexion à la base de données et quitter
--verbose, -v     Mode verbose (DEBUG)
```

## Configuration

Le collecteur externe utilise la même configuration que l'application principale :

- **PostgreSQL** : Si disponible et accessible, utilise PostgreSQL
- **CSV** : Sinon, utilise le fichier `data/matches.csv` (fallback)

La détection se fait automatiquement au démarrage.

### Variables d'Environnement

```bash
# PostgreSQL (optionnel)
export DATABASE_URL="postgresql://user:pass@localhost:5432/oracxpred"

# Sinon, le système utilisera CSV automatiquement
```

## Arrêt du Service

Pour arrêter le service en mode daemon :

- **Windows** : `Ctrl+C` dans le terminal
- **Linux/Unix/Mac** : `Ctrl+C` ou envoyer un signal SIGTERM :
  ```bash
  # Trouver le PID
  ps aux | grep collector_service
  
  # Arrêter proprement
  kill <PID>
  ```

## Logs

Les logs sont affichés dans la console par défaut. Si vous utilisez `--log-file`, les logs sont également écrits dans le fichier spécifié.

Format des logs :
```
[2025-12-06 10:30:45] [INFO] [COLLECTOR-SERVICE] Démarrage de la collecte...
[2025-12-06 10:30:46] [INFO] [COLLECTOR-SERVICE] OK Match terminé sauvegardé: Team1 vs Team2 (2-1)
[2025-12-06 10:30:46] [INFO] [COLLECTOR-SERVICE] Collecte terminée avec succès
```

## Intégration avec Cron (Linux/Unix/Mac)

Vous pouvez également utiliser cron pour exécuter le collecteur périodiquement au lieu du mode daemon :

```bash
# Éditer le crontab
crontab -e

# Exécuter toutes les 5 minutes
*/5 * * * * cd /chemin/vers/projet && python3 collector_service.py >> logs/cron_collector.log 2>&1

# Exécuter toutes les heures
0 * * * * cd /chemin/vers/projet && python3 collector_service.py >> logs/cron_collector.log 2>&1
```

## Intégration avec Task Scheduler (Windows)

1. Ouvrir "Planificateur de tâches" (Task Scheduler)
2. Créer une tâche de base
3. Déclencheur : Répétition (ex: toutes les 5 minutes)
4. Action : Démarrer un programme
   - Programme : `python.exe`
   - Arguments : `collector_service.py`
   - Dossier de départ : Chemin vers le projet

## Avantages du Collecteur Externe

1. **Indépendance** : Peut fonctionner même si l'app Flask est en maintenance
2. **Performance** : N'affecte pas les performances de l'application web
3. **Scalabilité** : Peut être déployé sur plusieurs serveurs
4. **Contrôle** : Planification indépendante, logs dédiés
5. **Robustesse** : En cas de crash de l'app Flask, la collecte continue

## Comparaison : Collecteur Interne vs Externe

| Fonctionnalité | Collecteur Interne (Flask) | Collecteur Externe |
|----------------|---------------------------|-------------------|
| Dépendance Flask | ✅ Oui | ❌ Non |
| Planification | APScheduler dans Flask | APScheduler autonome |
| Logs | Mélangés avec Flask | Logs dédiés |
| Déploiement | Sur le même serveur | Serveur indépendant possible |
| Arrêt Flask | ❌ Collecte s'arrête | ✅ Collecte continue |
| Ressources | Partagées avec Flask | Dédiées |

## Recommandations

- **Développement** : Utilisez le collecteur interne (Flask) pour la simplicité
- **Production** : Utilisez le collecteur externe pour la robustesse et l'indépendance
- **Multi-serveurs** : Utilisez le collecteur externe avec un seul processus actif pour éviter les doublons

## Dépannage

### Le service ne démarre pas

```bash
# Vérifier les dépendances
pip install -r requirements.txt

# Tester la connexion DB
python collector_service.py --test-db

# Mode verbose pour voir les erreurs
python collector_service.py --verbose
```

### Aucun match n'est collecté

1. Vérifier que l'API 1xBet est accessible
2. Vérifier les logs pour voir pourquoi les matchs sont ignorés
3. Utiliser `--verbose` pour plus de détails

### Les logs ne s'affichent pas

- Vérifier que le répertoire `logs/` existe si vous utilisez `--log-file`
- Vérifier les permissions d'écriture

