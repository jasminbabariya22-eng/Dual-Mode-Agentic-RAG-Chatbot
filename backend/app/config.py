"""
Application configuration.

This module centralizes all application settings using Pydantic Settings.
Values are loaded from environment variables and/or a local `.env` file.

Do NOT hardcode configuration values anywhere else in the project.
"""

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ======================================================================
    # Application
    # ======================================================================

    APP_NAME: str = "Dual-Mode Agentic RAG Chatbot"
    APP_VERSION: str = "1.0.0"
    
    # Environment mode (development, testing, production)
    APP_ENV: str = "development"

    DEBUG: bool = True

    HOST: str = "0.0.0.0"
    PORT: int = 8000

    REFERENCE_DATE: str = "2026-06-15"

    # ======================================================================
    # Project Paths
    # ======================================================================

    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent

    DATASET_DIR: Path = BASE_DIR / "Dataset"

    SQLITE_DB_PATH: Path = BASE_DIR / "orders.db"

    CHROMA_DB_PATH: Path = BASE_DIR / "chroma_db"

    LOG_DIR: Path = BASE_DIR / "logs"

    # ======================================================================
    # Primary LLM (Ollama)
    # ======================================================================

    LLM_PROVIDER: str = "ollama"

    LLM_MODEL: str = "qwen3:8b"

    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # ======================================================================
    # Fallback LLM (Groq)
    # ======================================================================

    ENABLE_FALLBACK_MODEL: bool = True

    FALLBACK_PROVIDER: str = "groq"

    FALLBACK_MODEL: str = "llama-3.3-70b-versatile"

    GROQ_API_KEY: Optional[str] = None

    # ======================================================================
    # LLM Runtime
    # ======================================================================

    LLM_TEMPERATURE: float = 0.0

    LLM_MAX_TOKENS: int = 4096

    LLM_TIMEOUT: int = 120

    LLM_MAX_RETRIES: int = 2

    LLM_STREAMING: bool = True

    # ======================================================================
    # Embedding Models
    # ======================================================================

    EMBEDDINGS_MODEL: str = "BAAI/bge-large-en-v1.5"

    RERANKER_MODEL: str = "BAAI/bge-reranker-base"

    # ======================================================================
    # Retrieval
    # ======================================================================

    CHUNK_SIZE: int = 600

    CHUNK_OVERLAP: int = 100

    VECTOR_TOP_K: int = 20

    RERANK_TOP_K: int = 5

    FINAL_TOP_K: int = 5

    # ======================================================================
    # Redis
    # ======================================================================

    REDIS_URL: str = "redis://localhost:6379/0"

    CACHE_TTL: int = 3600

    # ======================================================================
    # Semantic Cache
    # ======================================================================

    USE_SEMANTIC_CACHE: bool = True

    SEMANTIC_CACHE_THRESHOLD: float = 0.85

    # ======================================================================
    # Feature Flags
    # ======================================================================

    ENABLE_HYBRID_SEARCH: bool = True
    ENABLE_RERANKER: bool = True
    ENABLE_STREAMING: bool = True
    ENABLE_MEMORY: bool = True
    ENABLE_SQL_VALIDATION: bool = True
    
    # Guardrails
    ENABLE_GUARDRAILS: bool = True
    ENABLE_PROMPT_GUARD: bool = True
    ENABLE_OUTPUT_VALIDATION: bool = True
    ENABLE_SQL_GUARD: bool = True
    ENABLE_HALLUCINATION_CHECK: bool = True
    
    MIN_CONFIDENCE: float = 0.60
    MAX_QUESTION_LENGTH: int = 1000

    ENABLE_RAGAS: bool = True
    ENABLE_DEEPEVAL: bool = True

    # ======================================================================
    # Logging
    # ======================================================================

    LOG_LEVEL: str = "INFO"

    LOG_FORMAT: str = "json"

    # ======================================================================
    # Langfuse (Optional)
    # ======================================================================

    LANGFUSE_PUBLIC_KEY: Optional[str] = None

    LANGFUSE_SECRET_KEY: Optional[str] = None

    LANGFUSE_HOST: Optional[str] = None

    # ======================================================================
    # Prompt Versioning
    # ======================================================================

    PROMPT_VERSION: str = "v1"

    # ======================================================================
    # Environment Configuration
    # ======================================================================

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()


# Create required directories automatically
settings.LOG_DIR.mkdir(parents=True, exist_ok=True)
settings.CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)