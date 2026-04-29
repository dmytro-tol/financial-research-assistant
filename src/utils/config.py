"""Configuration loaded from environment variables."""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from .env."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # Existing
    sec_user_agent: str
    openai_api_key: str
    
    # Azure OpenAI (new)
    azure_openai_api_key: str = ""
    azure_openai_endpoint: str = ""
    azure_openai_api_version: str = "2024-08-01-preview"
    azure_llm_deployment: str = "gpt-4o-mini"
    azure_embedding_deployment: str = "text-embedding-3-small"
    use_azure: bool = False
    
    # Computed paths
    @property
    def project_root(self) -> Path:
        return Path(__file__).parent.parent.parent
    
    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"
    
    @property
    def data_raw_dir(self) -> Path:
        return self.data_dir / "raw"
    
    @property
    def data_processed_dir(self) -> Path:
        return self.data_dir / "processed"
    
    @property
    def chroma_db_dir(self) -> Path:
        return self.data_dir / "chroma_db"


settings = Settings()
