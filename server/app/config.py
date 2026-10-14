from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "VersionWise"
    app_env: str = "development"

    mongodb_uri: str = ""
    mongodb_database: str = Field(default="versionwise", min_length=1)
    mongodb_server_selection_timeout_ms: int = Field(default=2000, gt=0, le=60000)
    mongodb_connect_timeout_ms: int = Field(default=2000, gt=0, le=60000)

    embedding_provider: str = Field(default="sentence-transformers", description="Embedding provider: 'sentence-transformers' or 'mock'")
    embedding_model_name: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", description="Local sentence transformer model identifier or path")
    embedding_dimensions: int = Field(default=384, gt=0, le=4096, description="Embedding vector dimension")
    vector_search_index_name: str = Field(default="vector_index_chunks", min_length=1, description="MongoDB Atlas Vector Search index name")

    @field_validator("mongodb_database")
    @classmethod
    def validate_mongodb_database(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("mongodb_database cannot be empty or whitespace only")
        return v.strip()

    @field_validator("embedding_provider")
    @classmethod
    def validate_embedding_provider(cls, v: str) -> str:
        clean_v = v.strip().lower()
        if clean_v not in ("sentence-transformers", "mock"):
            raise ValueError(f"Unsupported embedding provider '{v}'. Allowed providers: 'sentence-transformers', 'mock'.")
        return clean_v

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
