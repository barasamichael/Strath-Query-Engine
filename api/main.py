import logging
from typing import Any
from typing import List
from typing import Dict
from typing import Optional

from fastapi import FastAPI
from fastapi import HTTPException

from pydantic import BaseModel

from config.settings import settings
from services.chunking.document_processor import DocumentProcessor
from services.retrieval.embeddings import EmbeddingService
from services.retrieval.vector_db import VectorDBService
from services.generation.intent_recognizer import IntentRecognizer
from services.generation.response_generator import ResponseGenerator

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

# Initialize FastAPI app
app = FastAPI(
    title="Strathmore University RAG API",
    description="API for the Strathmore University Retrieval Augmented Generation system",
    version="0.1.0",
)

# Initialize services
embedding_service = EmbeddingService()
vector_db_service = VectorDBService(embedding_service=embedding_service)
intent_recognizer = IntentRecognizer()
response_generator = ResponseGenerator()

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
    retrieved_chunks: Optional[List[Dict[str, Any]]] = None
    token_usage: Optional[Dict[str, int]] = None


class ProcessDocumentRequest(BaseModel):
    file_path: str


class ProcessDocumentResponse(BaseModel):
    status: str
    message: str
    document_metadata: Optional[Dict[str, Any]] = None


# Health check endpoint


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


# Process document endpoint


@app.post("/process-document", response_model=ProcessDocumentResponse)
async def process_document(request: ProcessDocumentRequest):
    try:
        document_processor = DocumentProcessor()
        metadata = document_processor.process_document(request.file_path)

        if not metadata:
            return ProcessDocumentResponse(
                status="error",
                message=f"Failed to process document: {request.file_path}",
            )

        # Embed chunks
        chunks_path = metadata["chunks_path"]
        embedding_service.embed_chunks(chunks_path)

        # Index chunks in vector database
        vector_db_service.index_chunks(chunks_path)

        return ProcessDocumentResponse(
            status="success",
            message=f"Document processed successfully: {metadata['doc_id']}",
            document_metadata=metadata,
        )
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        return ProcessDocumentResponse(
            status="error", message=f"Error processing document: {str(e)}"
        )


# Query endpoint


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
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


# Batch process documents endpoint


@app.post("/process-all-documents")
async def process_all_documents():
    try:
        document_processor = DocumentProcessor()
        documents_metadata = document_processor.process_all_documents()

        # Embed and index all documents
        embedding_service.embed_chunks()
        vector_db_service.index_chunks()

        return {
            "status": "success",
            "message": f"Processed {len(documents_metadata)} documents",
            "documents": documents_metadata,
        }
    except Exception as e:
        logger.error(f"Error processing documents: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error processing documents: {str(e)}"
        )


# Initialize collection endpoint


@app.post("/initialize-collection")
async def initialize_collection(recreate: bool = False):
    try:
        vector_db_service.initialize_collection(recreate=recreate)
        return {
            "status": "success",
            "message": f"{'Recreated' if recreate else 'Initialized'} collection "
            + f"{settings.vector_db.collection_name}",
        }
    except Exception as e:
        logger.error(f"Error initializing collection: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error initializing collection: {str(e)}"
        )
