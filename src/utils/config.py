"""Application configuration loaded from environment variables."""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration. Loaded from .env file."""
    
    # SEC EDGAR
    sec_user_agent: str
    
    # OpenAI
    openai_api_key: str
    
    # Paths — computed from project root
    project_root: Path = Path(__file__).parent.parent.parent
    data_raw_dir: Path = project_root / "data" / "raw"
    data_processed_dir: Path = project_root / "data" / "processed"
    data_chunks_dir: Path = project_root / "data" / "chunks"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )


settings = Settings()
