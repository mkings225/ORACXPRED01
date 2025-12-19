"""
Collecteur externe autonome pour ORACX PRED
Service de collecte indépendant qui peut s'exécuter séparément de l'application Flask.

Usage:
    # Exécution unique (one-shot)
    python collector_service.py

    # Mode daemon avec planification intégrée (toutes les 1 minute)
    python collector_service.py --daemon

    # Mode daemon avec intervalle personnalisé
    python collector_service.py --daemon --interval 5

    # Mode daemon avec logs dans un fichier
    python collector_service.py --daemon --log-file logs/collector.log

    # Test de connexion à la base de données
    python collector_service.py --test-db
"""
import argparse
import logging
import os
import signal
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

from apscheduler.schedulers.blocking import BlockingScheduler

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] [COLLECTOR-SERVICE] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Import du système de collecte (PostgreSQL ou CSV selon disponibilité)
USE_POSTGRESQL = False
collect_function = None

try:
    from db_collector import save_matches_to_db
    from models import get_session_factory
    
    # Tester la connexion à PostgreSQL
    try:
        SessionLocal = get_session_factory()
        session = SessionLocal()
        session.close()
        USE_POSTGRESQL = True
        collect_function = save_matches_to_db
        logger.info("Mode PostgreSQL active et connecte")
    except Exception as db_error:
        logger.warning(f"PostgreSQL non accessible: {db_error}, utilisation du mode CSV")
        USE_POSTGRESQL = False
except ImportError as e:
    logger.warning(f"Modules PostgreSQL non disponibles: {e}, utilisation du mode CSV")
    USE_POSTGRESQL = False

# Fallback vers CSV si PostgreSQL n'est pas disponible
if not collect_function:
    try:
        from collector import append_matches_to_csv
        collect_function = append_matches_to_csv
        logger.info("Mode CSV active (fallback)")
    except ImportError as e:
        logger.error(f"Impossible d'importer le collecteur CSV: {e}")
        sys.exit(1)

# Variable globale pour contrôler le daemon
running = True


def collect_matches():
    """Fonction de collecte unique qui peut être appelée par le scheduler."""
    try:
        logger.info("Demarrage de la collecte...")
        collect_function()
        logger.info("Collecte terminee avec succes")
        return True
    except Exception as e:
        logger.error(f"Erreur lors de la collecte: {e}")
        logger.debug(traceback.format_exc())
        return False


def signal_handler(signum, frame):
    """Gestionnaire de signaux pour arrêter proprement le daemon."""
    global running
    logger.info(f"Signal {signum} recu, arret du service en cours...")
    running = False


def test_database_connection():
    """Teste la connexion à la base de données."""
    if USE_POSTGRESQL:
        try:
            from models import get_session_factory
            SessionLocal = get_session_factory()
            session = SessionLocal()
            session.execute("SELECT 1")
            session.close()
            logger.info("OK Connexion a PostgreSQL reussie")
            return True
        except Exception as e:
            logger.error(f"ERREUR Connexion a PostgreSQL echouee: {e}")
            return False
    else:
        logger.warning("Mode CSV active - pas de test de base de donnees")
        return True


def run_daemon(interval_minutes: int = 1, log_file: Optional[str] = None):
    """
    Exécute le collecteur en mode daemon avec planification automatique.
    
    Args:
        interval_minutes: Intervalle entre chaque collecte (en minutes)
        log_file: Chemin vers un fichier de log (optionnel)
    """
    global running
    
    # Configuration du logging vers fichier si demandé
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter(
                '[%(asctime)s] [%(levelname)s] [COLLECTOR-SERVICE] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        )
        logger.addHandler(file_handler)
        logger.info(f"Logs ecrits dans: {log_file}")
    
    # Enregistrer les gestionnaires de signaux
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Configuration du scheduler
    scheduler = BlockingScheduler(daemon=False)
    
    # Ajouter la tâche de collecte
    scheduler.add_job(
        func=collect_matches,
        trigger="interval",
        minutes=interval_minutes,
        id="collect_job",
        name="Collecte des matchs terminés",
        replace_existing=True,
        max_instances=1  # Éviter les exécutions simultanées
    )
    
    logger.info(f"Service de collecte demarre (intervalle: {interval_minutes} minute(s))")
    logger.info("Premiere collecte dans quelques secondes...")
    
    try:
        # Exécuter une collecte immédiate au démarrage
        collect_matches()
        
        # Démarrer le scheduler (bloque jusqu'à interruption)
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Interruption recue, arret du scheduler...")
    except Exception as e:
        logger.error(f"Erreur dans le scheduler: {e}")
        logger.debug(traceback.format_exc())
    finally:
        scheduler.shutdown()
        logger.info("Service de collecte arrete")


def main():
    """Point d'entrée principal du collecteur externe."""
    parser = argparse.ArgumentParser(
        description="Collecteur externe autonome pour ORACX PRED",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python collector_service.py                    # Exécution unique
  python collector_service.py --daemon           # Mode daemon (1 min par défaut)
  python collector_service.py --daemon --interval 5  # Daemon avec intervalle de 5 minutes
  python collector_service.py --test-db          # Tester la connexion DB
        """
    )
    
    parser.add_argument(
        '--daemon',
        action='store_true',
        help='Exécuter en mode daemon avec planification automatique'
    )
    
    parser.add_argument(
        '--interval',
        type=int,
        default=1,
        metavar='MINUTES',
        help='Intervalle entre les collectes en minutes (défaut: 1)'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        metavar='PATH',
        help='Chemin vers le fichier de log (ex: logs/collector.log)'
    )
    
    parser.add_argument(
        '--test-db',
        action='store_true',
        help='Tester la connexion à la base de données et quitter'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Mode verbose (DEBUG)'
    )
    
    args = parser.parse_args()
    
    # Configuration du niveau de log
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Mode verbose active")
    
    # Test de la base de données
    if args.test_db:
        success = test_database_connection()
        sys.exit(0 if success else 1)
    
    # Validation de l'intervalle
    if args.interval < 1:
        logger.error("L'intervalle doit etre superieur ou egal a 1 minute")
        sys.exit(1)
    
    # Mode daemon
    if args.daemon:
        run_daemon(interval_minutes=args.interval, log_file=args.log_file)
    else:
        # Mode one-shot (exécution unique)
        logger.info("Mode execution unique")
        success = collect_matches()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

