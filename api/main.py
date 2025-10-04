import logging
from typing import Any
from typing import List
from typing import Dict
from typing import Optional
from functools import wraps

from fastapi import Header
from fastapi import Depends
from fastapi import FastAPI
from fastapi import Security
from fastapi import HTTPException
from fastapi.security import APIKeyHeader

from pydantic import BaseModel

from config.settings import settings
from services.vector_db import VectorDBService
from services.embeddings import EmbeddingService
from services.intent_recognizer import IntentRecognizer
from services.response_generator import ResponseGenerator

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

# Initialize FastAPI app
app = FastAPI(
    title="Strathmore University RAG API",
    description="API for the Strathmore University Retrieval Augmented Generation system",
    version="0.1.0",
)

# API Key security
API_KEY = settings.api.api_key
API_KEY_NAME = "Authorization"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Initialize services
embedding_service = EmbeddingService()
vector_db_service = VectorDBService(embedding_service=embedding_service)
intent_recognizer = IntentRecognizer()
response_generator = ResponseGenerator()


# Dependency for API key validation
async def get_api_key(api_key_header: str = Security(api_key_header)):
    if not api_key_header:
        raise HTTPException(
            status_code=401,
            detail="Missing API Key",
        )

    # Extract token from "Bearer {token}" format
    if api_key_header.startswith("Bearer "):
        api_key_header = api_key_header.split(" ")[1]

    if api_key_header != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API Key",
        )
    return api_key_header


# Pydantic models for API requests and responses
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    doc_id: Optional[str] = None


class QueryResponse(BaseModel):
    response: str
    intent_type: str
    topic: str
    confidence: float
    token_usage: Optional[Dict[str, int]] = None
    retrieved_chunks: Optional[List[Dict[str, Any]]] = None


# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    api_key: str = Depends(get_api_key)
):
    try:
        # Recognize intent
        intent_info = intent_recognizer.recognize_intent(request.query)
        logger.info(f"Recognized intent: {intent_info}")

        # Retrieve relevant context
        retrieved_chunks = []
        if intent_info["intent_type"] != "off_topic":
            retrieved_chunks = vector_db_service.search(
                query=request.query,
                top_k=request.top_k,
                filter_doc_id=request.doc_id,
            )

        # Generate response
        response_data = response_generator.generate_response(
            query=request.query,
            retrieved_context=retrieved_chunks,
            intent_info=intent_info,
        )

        # Include retrieved chunks in debug mode
        if settings.api.debug:
            response_data["retrieved_chunks"] = retrieved_chunks

        return response_data
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error processing query: {str(e)}"
        )


@app.post("/multi-query-search")
async def multi_query_search(
    request: QueryRequest,
    api_key: str = Depends(get_api_key)
):
    try:
        # Retrieve relevant context with multi-query approach
        retrieved_chunks = vector_db_service.multi_query_search(
            query=request.query,
            top_k=request.top_k,
            filter_doc_id=request.doc_id,
        )

        return {
            "query": request.query,
            "chunks": retrieved_chunks,
            "count": len(retrieved_chunks)
        }
    except Exception as e:
        logger.error(f"Error in multi-query search: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error in multi-query search: {str(e)}"
        )
