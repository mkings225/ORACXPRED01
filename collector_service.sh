#!/bin/bash
# Script Linux/Unix pour lancer le collecteur externe en mode daemon
# Usage: ./collector_service.sh [intervalle_en_minutes]

INTERVAL=${1:-1}

echo "[COLLECTOR-SERVICE] Démarrage du collecteur externe..."
echo "[COLLECTOR-SERVICE] Intervalle: $INTERVAL minute(s)"
echo ""

# Créer le répertoire de logs s'il n'existe pas
mkdir -p logs

# Lancer le collecteur en mode daemon
python3 collector_service.py --daemon --interval $INTERVAL --log-file logs/collector_service.log

