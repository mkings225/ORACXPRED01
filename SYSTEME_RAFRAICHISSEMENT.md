# 🔄 Système de Rafraîchissement Permanent

## ✅ Configuration Actuelle

### 1. Collecte Automatique en Arrière-Plan
- **Fréquence** : Toutes les **1 minute** (au lieu de 5 minutes)
- **Fonctionnement** : **PERMANENT**, même sans utilisateurs sur le site
- **Scheduler** : `BackgroundScheduler(daemon=True)` - fonctionne en arrière-plan
- **Détection** : Détecte les matchs terminés en temps quasi-réel

### 2. Rafraîchissement des Pages Web

#### Page d'accueil (`/` et `/matches`)
- **Fréquence** : Toutes les **5 secondes**
- **Méthode** : `setInterval()` JavaScript
- **Comportement** : Rafraîchissement automatique permanent

#### Page des matchs collectés (`/collected`)
- **Fréquence** : Toutes les **5 secondes**
- **Méthode** : Rechargement du contenu via fetch
- **Comportement** : Mise à jour automatique du tableau et du compteur

#### Page de détail (`/predictions/<id>`)
- **Fréquence** : Toutes les **5 secondes**
- **Méthode** : Vérification des données et rechargement si nécessaire
- **Comportement** : Détection des changements et rafraîchissement

## 🔧 Fonctionnement Technique

### Collecte Automatique (Backend)
```python
# Dans app.py
scheduler.add_job(
    func=job_collect,
    trigger="interval",
    minutes=1,  # Toutes les 1 minute
    id="collect_job",
    name="Collecte des matchs",
)
```

**Caractéristiques** :
- ✅ Fonctionne même sans utilisateurs
- ✅ Détecte les matchs terminés rapidement
- ✅ Sauvegarde automatique dans la base de données
- ✅ Logs détaillés pour le suivi

### Rafraîchissement Frontend (JavaScript)
```javascript
// Sur toutes les pages
setInterval(() => {
    // Rafraîchir les données
    loadMatches(false);
}, 5000); // 5 secondes
```

**Caractéristiques** :
- ✅ Rafraîchissement silencieux (pas de rechargement complet)
- ✅ Mise à jour uniquement des données nécessaires
- ✅ Optimisé pour ne pas surcharger le serveur
- ✅ Continue même si l'onglet n'est pas actif

## 📊 Avantages

1. **Détection Rapide** : Les matchs terminés sont détectés en moins de 1 minute
2. **Sauvegarde Permanente** : Le système collecte même la nuit ou sans visiteurs
3. **Interface Réactive** : Les pages se mettent à jour automatiquement
4. **Expérience Utilisateur** : Données toujours à jour sans action manuelle

## 🎯 Résultat

- **Collecte** : Toutes les 1 minute (60 fois par heure)
- **Affichage** : Toutes les 5 secondes (720 fois par heure)
- **Détection** : Matchs terminés sauvegardés en moins de 1 minute
- **Disponibilité** : 24/7, même sans utilisateurs

## ⚙️ Configuration

Pour modifier les intervalles :

### Backend (Collecte)
```python
# Dans app.py, ligne ~117
minutes=1,  # Changer ici (1 = 1 minute, 0.5 = 30 secondes)
```

### Frontend (Rafraîchissement)
```javascript
// Dans les templates HTML
setInterval(() => loadMatches(false), 5000); // 5000 = 5 secondes
```

## 🔍 Vérification

Pour vérifier que le système fonctionne :

1. **Vérifier les logs** :
   ```
   [SCHEDULER] ✅ Collecte #X effectuée avec succès
   [COLLECTOR] OK Match termine sauvegarde: ...
   ```

2. **Vérifier l'API** :
   ```
   GET /api/scheduler
   ```

3. **Observer les pages** : Les données se mettent à jour automatiquement toutes les 5 secondes

