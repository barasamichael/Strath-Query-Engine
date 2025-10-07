import os
import logging
from typing import Any
from typing import Dict
from typing import List
from pathlib import Path
from typing import Optional

from fastapi import File
from fastapi import Depends
from fastapi import FastAPI
from fastapi import Security
from fastapi import UploadFile
from fastapi import HTTPException
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

from config.settings import settings
from services.vector_db import VectorDBService
from services.embeddings import EmbeddingService
from services.intent_recognizer import IntentRecognizer
from services.response_generator import ResponseGenerator
from services.document_processor import DocumentProcessor

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

# Initialize FastAPI app
app = FastAPI(
    title="Strathmore University RAG API",
    description="API for the Strathmore University Retrieval Augmented Generation system",
    version="0.2.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Modify in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
document_processor = DocumentProcessor(
    enable_deduplication=settings.deduplication.enabled,
    similarity_threshold=settings.deduplication.similarity_threshold,
)

# Create temp directory for file uploads if it doesn't exist
UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


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


class DocumentResponse(BaseModel):
    doc_id: str
    file_name: str
    doc_type: str
    num_chunks: int
    success: bool = True
    message: str = "Document processed successfully"


class DeleteDocumentRequest(BaseModel):
    doc_id: str


class DocumentListResponse(BaseModel):
    documents: List[Dict[str, Any]]
    count: int


class DocumentInfoResponse(BaseModel):
    doc_id: str
    file_name: str
    doc_type: str
    num_chunks: int
    file_path: str
    processed_path: str
    chunks_path: str
    success: bool = True


# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest, api_key: str = Depends(get_api_key)):
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
    request: QueryRequest, api_key: str = Depends(get_api_key)
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
            "count": len(retrieved_chunks),
        }
    except Exception as e:
        logger.error(f"Error in multi-query search: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error in multi-query search: {str(e)}"
        )


# Document Management Endpoints
@app.post("/documents/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...), api_key: str = Depends(get_api_key)
):
    try:
        # Create a temporary file
        temp_file_path = UPLOAD_DIR / file.filename

        # Write the file
        with open(temp_file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Process the file
        logger.info(f"Processing uploaded file: {temp_file_path}")
        metadata = document_processor.process_file(temp_file_path)

        if not metadata:
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "message": "Failed to process document",
                },
            )

        # Generate embeddings for chunks
        embedding_service.embed_chunks(metadata["chunks_path"])

        # Index chunks in vector database
        vector_db_service.index_chunks(metadata["chunks_path"])

        # Clean up temp file
        os.remove(temp_file_path)

        return {
            "doc_id": metadata["doc_id"],
            "file_name": metadata["file_name"],
            "doc_type": metadata["doc_type"],
            "num_chunks": metadata["num_chunks"],
            "success": True,
            "message": "Document successfully processed and indexed",
        }

    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        # Clean up temp file if it exists
        if "temp_file_path" in locals() and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

        raise HTTPException(
            status_code=500, detail=f"Error uploading document: {str(e)}"
        )


@app.delete("/documents/{doc_id}", response_model=DocumentResponse)
async def delete_document(doc_id: str, api_key: str = Depends(get_api_key)):
    try:
        # Check if document exists
        doc_info = document_processor.get_document_info(doc_id)
        if not doc_info:
            return JSONResponse(
                status_code=404,
                content={
                    "success": False,
                    "message": f"Document {doc_id} not found",
                },
            )

        # Delete document
        success = document_processor.delete_document(doc_id)

        if not success:
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "message": f"Failed to delete document {doc_id}",
                },
            )

        return {
            "doc_id": doc_id,
            "file_name": doc_info["file_name"],
            "doc_type": doc_info["doc_type"],
            "num_chunks": doc_info["num_chunks"],
            "success": True,
            "message": "Document successfully deleted",
        }

    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error deleting document: {str(e)}"
        )


@app.get("/documents", response_model=DocumentListResponse)
async def list_documents(api_key: str = Depends(get_api_key)):
    try:
        documents = document_processor.list_documents()

        return {"documents": documents, "count": len(documents)}

    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error listing documents: {str(e)}"
        )


@app.get("/documents/{doc_id}", response_model=DocumentInfoResponse)
async def get_document_info(doc_id: str, api_key: str = Depends(get_api_key)):
    try:
        doc_info = document_processor.get_document_info(doc_id)

        if not doc_info:
            return JSONResponse(
                status_code=404,
                content={
                    "success": False,
                    "message": f"Document {doc_id} not found",
                },
            )

        return {**doc_info, "success": True}

    except Exception as e:
        logger.error(f"Error getting document info: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error getting document info: {str(e)}"
        )


@app.post("/documents/update", response_model=DocumentResponse)
async def update_document(
    file: UploadFile = File(...), api_key: str = Depends(get_api_key)
):
    try:
        # Create a temporary file
        temp_file_path = UPLOAD_DIR / file.filename

        # Write the file
        with open(temp_file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Process the file
        logger.info(f"Updating document from file: {temp_file_path}")
        metadata = document_processor.update_document(temp_file_path)

        if not metadata:
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "message": "Failed to update document",
                },
            )

        # Generate embeddings for chunks
        embedding_service.embed_chunks(metadata["chunks_path"])

        # Index chunks in vector database
        vector_db_service.index_chunks(metadata["chunks_path"])

        # Clean up temp file
        os.remove(temp_file_path)

        return {
            "doc_id": metadata["doc_id"],
            "file_name": metadata["file_name"],
            "doc_type": metadata["doc_type"],
            "num_chunks": metadata["num_chunks"],
            "success": True,
            "message": "Document successfully updated and indexed",
        }

    except Exception as e:
        logger.error(f"Error updating document: {str(e)}")
        # Clean up temp file if it exists
        if "temp_file_path" in locals() and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

        raise HTTPException(
            status_code=500, detail=f"Error updating document: {str(e)}"
        )


@app.post("/rebuild-index")
async def rebuild_index(api_key: str = Depends(get_api_key)):
    try:
        # Get all documents
        documents = document_processor.list_documents()

        if not documents:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "message": "No documents found to index",
                },
            )

        # Reinitialize the collection
        vector_db_service.initialize_collection(recreate=True)

        # Index each document's chunks
        indexed_count = 0
        for doc in documents:
            vector_db_service.index_chunks(doc["chunks_path"])
            indexed_count += 1

        return {
            "success": True,
            "message": f"Vector database index rebuilt successfully with {indexed_count} documents",
            "indexed_documents": indexed_count,
            "total_documents": len(documents),
        }

    except Exception as e:
        logger.error(f"Error rebuilding index: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error rebuilding index: {str(e)}"
        )
