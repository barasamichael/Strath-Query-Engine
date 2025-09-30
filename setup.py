import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("setup")

# Project root directory
ROOT_DIR = Path(__file__).parent.absolute()

# Define project structure
DIRECTORIES = [
    "data/raw",
    "data/processed",
    "data/chunks",
    "data/embeddings",
    "data/metadata",
    "database/vector_store",
    "database/relational",
    "models/embeddings",
    "models/llm",
    "models/intent",
    "api/endpoints",
    "api/middleware",
    "services/chunking",
    "services/retrieval",
    "services/generation",
    "services/evaluation",
    "config",
    "utils",
    "tests",
]


# Create directories if they don't exist
def create_project_structure():
    logger.info("Creating project directory structure...")
    for directory in DIRECTORIES:
        dir_path = ROOT_DIR / directory
        if not dir_path.exists():
            os.makedirs(dir_path)
            logger.info(f"Created directory: {dir_path}")
        else:
            logger.info(f"Directory already exists: {dir_path}")


# Create initial configuration files
def create_config_files():
    logger.info("Creating initial configuration files...")

    # Main configuration
    config_content = """
# Strathmore RAG System Configuration

# Environment (development, test, production)
environment: development

# LLM Configuration
llm:
  provider: openai
  model: gpt-3.5-turbo  # Using 3.5 for development to minimize costs
  temperature: 0.1
  max_tokens: 1000
  
# Embedding Configuration
embedding:
  model: sentence-transformers/all-MiniLM-L6-v2  # Cost-effective embedding model
  dimension: 384
  
# Database Configuration
vector_db:
  type: qdrant
  location: local  # 'local' or 'cloud'
  collection_name: strathmore_handbook
  
# Chunking Configuration
chunking:
  chunk_size: 500
  chunk_overlap: 50
  
# API Configuration
api:
  host: 0.0.0.0
  port: 8000
  debug: true
"""
    with open(ROOT_DIR / "config" / "config.yaml", "w") as f:
        f.write(config_content)
    logger.info("Created main configuration file")

    # Environment variables template
    env_template = """
# Environment Variables (.env)
# Copy this file to .env and fill in your actual API keys

# OpenAI API Key
OPENAI_API_KEY=your_openai_api_key_here

# Qdrant API Key (if using cloud)
QDRANT_API_KEY=your_qdrant_api_key_here

# Host settings
HOST=0.0.0.0
PORT=8000
"""
    with open(ROOT_DIR / "config" / ".env.template", "w") as f:
        f.write(env_template)
    logger.info("Created environment variables template")


# Create main application files
def create_application_files():
    logger.info("Creating main application files...")

    # Main application entry point
    app_content = """
# app.py
import logging
import uvicorn
from api.endpoints.main import app
from config.settings import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("app")

if __name__ == "__main__":
    logger.info(f"Starting Strathmore RAG API on {settings.api.host}:{settings.api.port}")
    uvicorn.run("app:app", host=settings.api.host, port=settings.api.port, reload=settings.api.debug)
"""
    with open(ROOT_DIR / "app.py", "w") as f:
        f.write(app_content)
    logger.info("Created main application file")

    # Requirements file
    requirements_content = """
# Core dependencies
fastapi>=0.95.0
pydantic>=2.0.0
uvicorn>=0.22.0
python-dotenv>=0.21.0
PyYAML>=6.0
typer>=0.7.0
tqdm>=4.65.0
rich>=13.0.0

# Data processing
langchain>=0.0.200
openai>=1.0.0
numpy>=1.24.0
pandas>=2.0.0
beautifulsoup4>=4.11.0
pypdf>=3.0.0

# Vector storage and search
scikit-learn>=1.0.0
jsonlines>=3.0.0

# Testing
pytest>=7.3.0
"""
    with open(ROOT_DIR / "requirements.txt", "w") as f:
        f.write(requirements_content)
    logger.info("Created requirements.txt")


if __name__ == "__main__":
    logger.info("Setting up Strathmore RAG project structure...")
    create_project_structure()
    create_config_files()
    create_application_files()
    logger.info(
        "Project setup complete! Run 'pip install -r requirements.txt' to install dependencies."
    )
