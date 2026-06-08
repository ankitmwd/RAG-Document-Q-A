from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    gemini_api_key: str
    pinecone_api_key: str
    pinecone_index_name: str = "rag-document-qa-gemini-3072"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    embedding_model: str = "models/gemini-embedding-001"
    embedding_dimension: int = 3072
    chat_model: str = "gemini-2.5-flash"
    upload_dir: str = "data/uploads"
    chunk_size: int = 1000
    chunk_overlap: int = 150
    top_k: int = 4

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
    return Settings()
