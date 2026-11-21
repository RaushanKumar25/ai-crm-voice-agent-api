import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Base directory for resolving relative paths
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env file if present
load_dotenv(dotenv_path=BASE_DIR / ".env")


class Settings:
    """Application settings loaded from environment variables."""

    # Gemini API Configuration
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "your_api_key_here").strip()
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-4-hydrogen").strip()
    GEMINI_FALLBACK_MODEL: str = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.5-flash-lite").strip()

    # Application Configuration
    APP_ENV: str = os.getenv("APP_ENV", "development").strip()
    APP_HOST: str = os.getenv("APP_HOST", "0.0.0.0").strip()
    APP_PORT: int = int(os.getenv("APP_PORT", "8000"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").strip()

    # Knowledge Base Path
    KNOWLEDGE_BASE_PATH: Path = BASE_DIR / os.getenv("KNOWLEDGE_BASE_PATH", "data/knowledge.txt")

    @property
    def is_api_key_configured(self) -> bool:
        """Check whether a real Gemini API key is configured."""
        return (
            bool(self.GEMINI_API_KEY)
            and self.GEMINI_API_KEY != "your_api_key_here"
            and not self.GEMINI_API_KEY.startswith("YOUR_")
        )


settings = Settings()
