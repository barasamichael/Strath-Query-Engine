import os
import yaml
from pathlib import Path
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project root directory
ROOT_DIR = Path(__file__).parent.parent.absolute()


class LLMConfig(BaseModel):
    provider: str
    model: str
    temperature: float
    max_tokens: int


class EmbeddingConfig(BaseModel):
    model: str
    dimension: int


class VectorDBConfig(BaseModel):
    type: str
    location: str
    collection_name: str


class ChunkingConfig(BaseModel):
    chunk_size: int
    chunk_overlap: int


class DeduplicationConfig(BaseModel):
    enabled: bool
    similarity_threshold: float
    information_weight: float


class APIConfig(BaseModel):
    host: str
    port: int
    debug: bool
    api_key: str


class Settings(BaseModel):
    environment: str
    llm: LLMConfig
    embedding: EmbeddingConfig
    vector_db: VectorDBConfig
    chunking: ChunkingConfig
    deduplication: DeduplicationConfig
    api: APIConfig

    @classmethod
    def from_yaml(cls, file_path: Path) -> "Settings":
        with open(file_path, "r") as file:
            config_dict = yaml.safe_load(file)
        return cls.parse_obj(config_dict)


# Initialize settings from YAML config file
settings_file = ROOT_DIR / "config" / "config.yaml"
settings = Settings.from_yaml(settings_file)

# Override with environment variables if provided
if os.getenv("HOST"):
    settings.api.host = os.getenv("HOST")

if os.getenv("PORT"):
    settings.api.port = int(os.getenv("PORT"))

if os.getenv("DEDUPLICATION_ENABLED"):
    settings.deduplication.enabled = (
        os.getenv("DEDUPLICATION_ENABLED").lower() == "true"
    )
