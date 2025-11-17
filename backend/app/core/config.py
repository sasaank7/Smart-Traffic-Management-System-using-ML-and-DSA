"""
Configuration management using Pydantic Settings
"""
from functools import lru_cache
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import PostgresDsn, validator


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # Application
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    SECRET_KEY: str
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 4

    # Database
    POSTGRES_HOST: str
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    DATABASE_URL: Optional[str] = None

    @validator("DATABASE_URL", pre=True)
    def assemble_db_connection(cls, v: Optional[str], values: dict) -> str:
        if isinstance(v, str):
            return v
        return f"postgresql+asyncpg://{values.get('POSTGRES_USER')}:{values.get('POSTGRES_PASSWORD')}@{values.get('POSTGRES_HOST')}:{values.get('POSTGRES_PORT')}/{values.get('POSTGRES_DB')}"

    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: Optional[str] = None
    REDIS_DB: int = 0
    CACHE_TTL: int = 300

    # Kafka
    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092"
    KAFKA_TOPIC_TRAFFIC_DATA: str = "traffic-data"
    KAFKA_TOPIC_ANOMALIES: str = "traffic-anomalies"
    KAFKA_TOPIC_PREDICTIONS: str = "traffic-predictions"
    KAFKA_CONSUMER_GROUP: str = "traffic-management-group"

    # ML Service
    ML_SERVICE_HOST: str = "localhost"
    ML_SERVICE_PORT: int = 8001
    ML_SERVICE_URL: Optional[str] = None

    @validator("ML_SERVICE_URL", pre=True)
    def assemble_ml_service_url(cls, v: Optional[str], values: dict) -> str:
        if isinstance(v, str):
            return v
        return f"http://{values.get('ML_SERVICE_HOST')}:{values.get('ML_SERVICE_PORT')}"

    # Authentication
    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000"]
    CORS_CREDENTIALS: bool = True

    @validator("CORS_ORIGINS", pre=True)
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60
    RATE_LIMIT_PER_HOUR: int = 1000

    # External APIs
    GOOGLE_MAPS_API_KEY: Optional[str] = None
    WEATHER_API_KEY: Optional[str] = None

    # Traffic Signal Configuration
    DEFAULT_SIGNAL_DURATION: int = 30
    MIN_SIGNAL_DURATION: int = 10
    MAX_SIGNAL_DURATION: int = 120
    EMERGENCY_PRIORITY_MULTIPLIER: int = 5

    # Data Collection
    SENSOR_POLL_INTERVAL: int = 10
    PREDICTION_INTERVAL: int = 300

    # Logging
    LOG_LEVEL: str = "INFO"
    SENTRY_DSN: Optional[str] = None

    # Performance
    MAX_WORKERS: int = 8
    CONNECTION_POOL_SIZE: int = 20

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


settings = get_settings()
