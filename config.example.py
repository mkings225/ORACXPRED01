"""
Configuration example pour environnement Enterprise
Copier vers config.py et adapter selon l'environnement
"""
import os
from pathlib import Path

class Config:
    """Configuration de base"""
    
    # Application
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'change-me-in-production'
    DEBUG = False
    TESTING = False
    
    # Database
    DATABASE_URL = os.environ.get('DATABASE_URL') or 'postgresql://user:pass@localhost/oracxpred'
    SQLALCHEMY_DATABASE_URI = DATABASE_URL
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_size': 10,
        'pool_recycle': 3600,
        'pool_pre_ping': True,
    }
    
    # Redis
    REDIS_URL = os.environ.get('REDIS_URL') or 'redis://localhost:6379/0'
    CACHE_TYPE = 'redis'
    CACHE_REDIS_URL = REDIS_URL
    CACHE_DEFAULT_TIMEOUT = 300
    
    # Celery (Task Queue)
    CELERY_BROKER_URL = os.environ.get('CELERY_BROKER_URL') or REDIS_URL
    CELERY_RESULT_BACKEND = os.environ.get('CELERY_RESULT_BACKEND') or REDIS_URL
    
    # JWT
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY') or SECRET_KEY
    JWT_ACCESS_TOKEN_EXPIRES = 3600  # 1 hour
    JWT_REFRESH_TOKEN_EXPIRES = 86400  # 24 hours
    
    # Rate Limiting
    RATELIMIT_ENABLED = True
    RATELIMIT_STORAGE_URL = REDIS_URL
    RATELIMIT_DEFAULT = "100 per hour"
    
    # API
    API_VERSION = 'v1'
    API_PREFIX = f'/api/{API_VERSION}'
    
    # External APIs
    API_1XBET_URL = os.environ.get('API_1XBET_URL') or (
        "https://1xbet.com/service-api/LiveFeed/Get1x2_VZip"
        "?sports=85&count=40&lng=fr&gr=285&mode=4&country=96"
        "&getEmpty=true&virtualSports=true&noFilterBlockEvent=true"
    )
    API_TIMEOUT = 10
    
    # Scheduler
    SCHEDULER_COLLECT_INTERVAL = int(os.environ.get('SCHEDULER_COLLECT_INTERVAL', 5))  # minutes
    SCHEDULER_TRAIN_HOUR = int(os.environ.get('SCHEDULER_TRAIN_HOUR', 3))  # 3 AM
    SCHEDULER_TRAIN_MINUTE = int(os.environ.get('SCHEDULER_TRAIN_MINUTE', 0))
    
    # Paths
    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR / 'data'
    MODELS_DIR = BASE_DIR / 'models'
    LOGS_DIR = BASE_DIR / 'logs'
    
    # Logging
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FORMAT = 'json'  # 'json' or 'text'
    LOG_FILE = LOGS_DIR / 'app.log'
    
    # Monitoring
    PROMETHEUS_ENABLED = True
    SENTRY_DSN = os.environ.get('SENTRY_DSN', '')
    
    # ML
    ML_MODEL_PATH = MODELS_DIR / 'model.joblib'
    ML_MODEL_VERSION = os.environ.get('ML_MODEL_VERSION', '1.0.0')
    ML_MIN_SAMPLES_FOR_TRAINING = 100
    
    # Security
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', '*').split(',')
    ALLOWED_HOSTS = os.environ.get('ALLOWED_HOSTS', '*').split(',')
    
    # Feature Flags
    FEATURE_COLLECT_ENABLED = True
    FEATURE_TRAIN_ENABLED = True
    FEATURE_PREDICT_ENABLED = True


class DevelopmentConfig(Config):
    """Configuration pour développement"""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'
    LOG_FORMAT = 'text'


class ProductionConfig(Config):
    """Configuration pour production"""
    DEBUG = False
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = 'json'
    
    # Security
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # Performance
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_size': 20,
        'pool_recycle': 3600,
        'pool_pre_ping': True,
        'max_overflow': 10,
    }


class TestingConfig(Config):
    """Configuration pour tests"""
    TESTING = True
    DATABASE_URL = 'sqlite:///:memory:'
    SQLALCHEMY_DATABASE_URI = DATABASE_URL
    CELERY_TASK_ALWAYS_EAGER = True  # Execute tasks synchronously
    LOG_LEVEL = 'WARNING'


# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig,
}

