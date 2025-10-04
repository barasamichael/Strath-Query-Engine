import logging
import uvicorn
from api.main import app
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
