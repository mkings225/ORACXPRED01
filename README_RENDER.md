# 🚀 Guide de Déploiement sur Render

## 📋 Prérequis

1. Compte Render (gratuit disponible sur [render.com](https://render.com))
2. Repository Git (GitHub, GitLab, ou Bitbucket)
3. Code pushé sur votre repository

## 🔧 Configuration sur Render

### 1. Créer un nouveau Web Service

1. Connectez-vous à [Render Dashboard](https://dashboard.render.com)
2. Cliquez sur **"New +"** → **"Web Service"**
3. Connectez votre repository Git
4. Sélectionnez le repository contenant ce projet

### 2. Configuration du Service

**Settings de base :**
- **Name** : `oracxpred-web` (ou votre choix)
- **Environment** : `Python 3`
- **Build Command** : 
  ```bash
  pip install -r requirements.txt && python setup_render.py
  ```
- **Start Command** : 
  ```bash
  gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --threads 2 --timeout 120
  ```

### 3. Créer une Base de Données PostgreSQL

1. Dans le Dashboard Render, cliquez sur **"New +"** → **"PostgreSQL"**
2. Configurez :
   - **Name** : `oracxpred-db`
   - **Database** : `oracxpred`
   - **User** : `oracxpred`
   - **Plan** : Free (ou supérieur selon vos besoins)
3. **IMPORTANT** : Notez l'**Internal Database URL** qui sera automatiquement ajoutée comme variable d'environnement

### 4. Variables d'Environnement

Dans les **Environment Variables** de votre Web Service, ajoutez :

| Variable | Valeur | Description |
|----------|--------|-------------|
| `DATABASE_URL` | *(Auto-rempli par Render si DB liée)* | URL de connexion PostgreSQL |
| `PORT` | `10000` | Port d'écoute (généralement géré automatiquement) |
| `PYTHON_VERSION` | `3.11.0` | Version Python |
| `TASK_TOKEN` | *(Optionnel)* | Token pour sécuriser `/tasks/collect` |

**Note** : Si vous liez la base de données PostgreSQL au service web dans Render, la variable `DATABASE_URL` sera automatiquement ajoutée.

### 5. Lier la Base de Données au Service Web

1. Dans les settings de votre Web Service
2. Section **"Connections"** ou **"Linked Resources"**
3. Sélectionnez votre base de données PostgreSQL
4. Render ajoutera automatiquement `DATABASE_URL`

## 🔄 Déploiement

1. **Push automatique** : Render déploie automatiquement à chaque push sur la branche principale
2. **Déploiement manuel** : Cliquez sur **"Manual Deploy"** dans le Dashboard

## ✅ Vérification

Une fois déployé, vérifiez :

1. **Logs** : Consultez les logs dans le Dashboard Render
   - Recherchez : `[SCHEDULER] OK Taches planifiees demarrees`
   - Recherchez : `[APP] OK Mode PostgreSQL active et connecte`

2. **Health Check** : Visitez `https://votre-app.onrender.com/`
   - La page d'accueil doit s'afficher

3. **API Status** : Visitez `https://votre-app.onrender.com/api/status`
   - Vérifiez que `scheduler_running: true`

4. **Collecte manuelle** : Visitez `https://votre-app.onrender.com/collect`
   - Devrait retourner `{"ok": true, "message": "Collecte effectuée avec succès"}`

## 🐛 Dépannage

### Erreur : "Database connection failed"
- Vérifiez que la base de données est bien liée au service web
- Vérifiez que `DATABASE_URL` est définie dans les variables d'environnement
- Vérifiez les logs pour voir l'erreur exacte

### Erreur : "Scheduler not running"
- Vérifiez les logs pour voir si le scheduler démarre
- Le scheduler doit démarrer automatiquement au lancement de l'app

### Erreur : "Module not found"
- Vérifiez que `requirements.txt` contient toutes les dépendances
- Vérifiez les logs de build pour voir les erreurs d'installation

### L'application se met en veille (Free Plan)
- Sur le plan gratuit, Render met les services en veille après 15 minutes d'inactivité
- Le premier accès après la veille peut prendre 30-60 secondes
- Pour éviter cela, utilisez un service de monitoring (UptimeRobot, etc.) qui ping votre site toutes les 5 minutes

## 📊 Monitoring

### Logs en temps réel
- Dashboard Render → Votre Service → **"Logs"**
- Surveillez les messages `[SCHEDULER]` pour voir les collectes automatiques

### Métriques
- Dashboard Render → Votre Service → **"Metrics"**
- Surveillez CPU, RAM, et requêtes

## 🔒 Sécurité

1. **TASK_TOKEN** : Définissez un token fort pour protéger `/tasks/collect`
2. **SECRET_KEY** : Si vous utilisez des sessions Flask, définissez `SECRET_KEY`
3. **HTTPS** : Render fournit HTTPS automatiquement

## 💰 Coûts

- **Free Plan** : 
  - Web Service : Gratuit (avec limitations)
  - PostgreSQL : Gratuit jusqu'à 90 jours, puis $7/mois
  - Mise en veille après 15 min d'inactivité

- **Starter Plan** ($7/mois) :
  - Pas de mise en veille
  - Plus de ressources

## 📝 Notes Importantes

1. **Scheduler** : Le scheduler APScheduler fonctionne en arrière-plan même sans utilisateurs
2. **Collecte automatique** : Toutes les 1 minute, même si personne n'est sur le site
3. **Entraînement** : Tous les jours à 3h00 du matin (heure UTC)
4. **Base de données** : Les tables sont créées automatiquement au premier déploiement

## 🆘 Support

En cas de problème :
1. Consultez les logs dans le Dashboard Render
2. Vérifiez la documentation Render : [render.com/docs](https://render.com/docs)
3. Vérifiez que toutes les variables d'environnement sont correctement configurées

