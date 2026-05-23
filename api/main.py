import os
import json
import logging
from typing import Any
from typing import Dict
from typing import List
from pathlib import Path
from typing import Optional

from fastapi import File
from fastapi import Query
from fastapi import Depends
from fastapi import FastAPI
from fastapi import Security
from fastapi import UploadFile
from fastapi import HTTPException
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
from pydantic import Field

from config.settings import settings
from config.settings import ROOT_DIR
from services.evaluator import RAGEvaluator
from services.vector_db import VectorDBService
from services.embeddings import EmbeddingService
from services.tavily_service import TavilyService
from services.query_router import SmartQueryRouter
from services.memory_processor import MemoryProcessor
from services.intent_recognizer import IntentRecognizer
from services.response_generator import ResponseGenerator
from services.schedule_extractor import ScheduleExtractor
from services.structured_storage import StructuredDataStorage
from services.document_processor import create_document_processor
from services.document_processor import validate_document_processor_setup

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("optimized_api")

# Initialize FastAPI app
app = FastAPI(
    title="Enhanced Strathmore University RAG API",
    description=(
        "Enhanced API for the Strathmore University Retrieval "
        "Augmented Generation system with optimized service management"
    ),
    version="2.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key security
API_KEY = settings.api.api_key
API_KEY_NAME = "Authorization"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


class ServiceContainer:
    """Centralized service container for API with proper dependency injection."""

    def __init__(self):
        self._services = {}
        self._initialized = False

    def initialize(self):
        """Initialize all services with proper dependency management."""
        if self._initialized:
            return

        try:
            logger.info("Initializing API service container...")

            # Core services first (no dependencies)
            self._services["embedding"] = EmbeddingService()

            # Tavily service (independent)
            if os.getenv("TAVILY_API_KEY"):
                self._services["tavily"] = TavilyService()
                logger.info("Tavily service initialized for real-time features")
            else:
                self._services["tavily"] = None
                logger.warning("Tavily service disabled - no API key")

            # Document processor (depends on embeddings)
            self._services["document_processor"] = create_document_processor(
                config_type=settings.environment,
                enable_deduplication=settings.deduplication.enabled,
                similarity_threshold=settings.deduplication.similarity_threshold,
            )

            # Vector DB (depends on embeddings and tavily)
            self._services["vector_db"] = VectorDBService(
                embedding_service=self._services["embedding"],
                tavily_service=self._services["tavily"],
            )

            # Schedule extractor (independent)
            self._services["schedule_extractor"] = ScheduleExtractor()

            # Structured storage (independent)
            structured_db_path = Path(ROOT_DIR) / "data" / "schedules.db"
            self._services["structured_storage"] = StructuredDataStorage(
                structured_db_path
            )

            # Intent recognizer (independent)
            self._services["intent_recognizer"] = IntentRecognizer()

            # Memory processor (independent)
            self._services["memory_processor"] = MemoryProcessor()

            # Query router (depends on structured_storage, vector_db, intent_recognizer)
            self._services["query_router"] = SmartQueryRouter(
                structured_storage=self._services["structured_storage"],
                vector_db_service=self._services["vector_db"],
                intent_recognizer=self._services["intent_recognizer"],
            )

            # Response generator (depends on vector_db, embeddings, tavily)
            self._services["response_generator"] = ResponseGenerator(
                vector_db_service=self._services["vector_db"],
                embedding_service=self._services["embedding"],
                tavily_service=self._services["tavily"],
            )

            # Evaluator (depends on vector_db, intent_recognizer, response_generator)
            self._services["evaluator"] = RAGEvaluator(
                vector_db_service=self._services["vector_db"],
                intent_recognizer=self._services["intent_recognizer"],
                response_generator=self._services["response_generator"],
            )

            self._initialized = True
            logger.info("API service container initialized successfully")

        except Exception as e:
            logger.error(f"Service container initialization failed: {e}")
            raise

    def get(self, service_name: str):
        """Get a service by name."""
        if not self._initialized:
            self.initialize()
        return self._services.get(service_name)

    def health_check(self) -> Dict[str, Any]:
        """Check health of all services."""
        if not self._initialized:
            self.initialize()

        health = {}
        for name, service in self._services.items():
            if service is None:
                health[name] = {
                    "status": "disabled",
                    "reason": "Not configured",
                }
            else:
                try:
                    if hasattr(service, "get_collection_stats"):
                        stats = service.get_collection_stats()
                        health[name] = {"status": "healthy", "stats": stats}
                    elif hasattr(service, "get_embedding_stats"):
                        stats = service.get_embedding_stats()
                        health[name] = {"status": "healthy", "stats": stats}
                    elif hasattr(service, "get_statistics"):
                        stats = service.get_statistics()
                        health[name] = {"status": "healthy", "stats": stats}
                    elif hasattr(service, "get_response_stats"):
                        stats = service.get_response_stats()
                        health[name] = {"status": "healthy", "stats": stats}
                    else:
                        health[name] = {
                            "status": "healthy",
                            "info": "No stats available",
                        }
                except Exception as e:
                    health[name] = {"status": "error", "error": str(e)}

        return health


# Initialize service container
container = ServiceContainer()

# Create temp directory for file uploads if it doesn't exist
UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


def get_service(name: str):
    """Helper to get service from container."""
    return container.get(name)


# Dependency for API key validation
async def get_api_key(api_key_header: str = Security(api_key_header)):
    if not api_key_header:
        raise HTTPException(
            status_code=401,
            detail="Missing API Key",
        )

    if api_key_header.startswith("Bearer "):
        api_key_header = api_key_header.split(" ")[1]

    if api_key_header != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API Key",
        )
    return api_key_header


# Pydantic models for API requests and responses - MAINTAINING ALL EXISTING PAYLOADS
class ConversationMessage(BaseModel):
    content: str
    isUserMessage: bool = True
    topic: Optional[str] = "general"
    timestamp: Optional[str] = None


class QueryRequest(BaseModel):
    query: str
    top_k: int = Field(default=15, ge=1, le=50)
    doc_id: Optional[str] = None
    use_multi_query: bool = False
    use_memory: bool = False
    conversation_history: Optional[List[ConversationMessage]] = None
    max_history_messages: int = Field(default=15, ge=1, le=50)


class ContextInfo(BaseModel):
    has_context: bool
    context_is_relevant: bool
    needs_clarification: bool = False
    clarification_prompt: Optional[str] = None
    relevance_score: float
    relevance_reasons: List[str]
    resolved_references: List[Dict[str, Any]] = []
    context_instructions: str
    memory_available: bool = False
    comprehensive_context_used: bool = False


class QueryResponse(BaseModel):
    response: str
    intent_type: str
    topic: str
    confidence: float
    token_usage: Optional[Dict[str, int]] = None
    retrieved_chunks: Optional[List[Dict[str, Any]]] = None
    context_info: Optional[ContextInfo] = None
    current_time: Optional[str] = None
    needs_clarification: bool = False
    clarification_prompt: Optional[str] = None


class URLProcessRequest(BaseModel):
    url: str
    output_name: Optional[str] = None


class SitemapProcessRequest(BaseModel):
    sitemap_url: str
    max_pages: int = Field(default=50, ge=1, le=200)


class DocumentResponse(BaseModel):
    doc_id: str
    file_name: str
    doc_type: str
    num_chunks: int
    success: bool = True
    message: str = "Document processed successfully"
    file_size: Optional[int] = None
    processing_time: Optional[float] = None


class DocumentListResponse(BaseModel):
    documents: List[Dict[str, Any]]
    count: int
    total_chunks: int = 0
    supported_formats: List[str] = []


class DocumentInfoResponse(BaseModel):
    doc_id: str
    file_name: str
    doc_type: str
    num_chunks: int
    file_path: str
    processed_path: str
    chunks_path: str
    file_size: Optional[int] = None
    last_modified: Optional[float] = None
    processed_date: Optional[float] = None
    chunk_settings: Optional[Dict[str, Any]] = None
    success: bool = True


class SystemStatusResponse(BaseModel):
    status: str
    components: Dict[str, Dict[str, Any]]
    configuration: Dict[str, Any]
    validation: Dict[str, Any]


class DeduplicationStatusResponse(BaseModel):
    enabled: bool
    similarity_threshold: float
    deduplicated_chunks_found: bool
    chunk_count: Optional[int] = None
    stats: Optional[Dict[str, Any]] = None
    quality_metrics: Optional[Dict[str, Any]] = None


class EvaluationRequest(BaseModel):
    create_template: bool = False
    questions: Optional[List[Dict[str, Any]]] = None


class EvaluationResponse(BaseModel):
    status: str
    num_questions: int
    avg_score: float
    intent_accuracy: float
    topic_accuracy: float
    content_accuracy: float
    total_tokens: int
    avg_tokens_per_query: float
    results: List[Dict[str, Any]]


class MemoryTestRequest(BaseModel):
    scenario: str
    messages: List[ConversationMessage]
    query: str


class MemoryTestResponse(BaseModel):
    context_result: Dict[str, Any]
    intent_result: Dict[str, Any]
    success: bool
    message: str


# Health check endpoint with enhanced information
@app.get("/health")
async def health_check():
    try:
        # Initialize services if needed
        container.initialize()

        response_generator = get_service("response_generator")

        health_status = {
            "status": "healthy",
            "timestamp": (
                response_generator.get_current_kenya_time()[0]
                if response_generator
                else None
            ),
            "api_version": "2.1.0",
        }

        # Check critical services
        vector_db_service = get_service("vector_db")
        if vector_db_service:
            try:
                vector_stats = vector_db_service.get_collection_stats()
                health_status["vector_db"] = {
                    "status": "healthy",
                    "count": vector_stats["count"],
                }
            except Exception as e:
                health_status["vector_db"] = {
                    "status": "error",
                    "error": str(e),
                }

        embedding_service = get_service("embedding")
        if embedding_service:
            try:
                embedding_stats = embedding_service.get_embedding_stats()
                health_status["embeddings"] = {
                    "status": "healthy",
                    "total_embeddings": embedding_stats["total_embeddings"],
                }
            except Exception as e:
                health_status["embeddings"] = {
                    "status": "error",
                    "error": str(e),
                }

        health_status["openai_api"] = {
            "status": "configured" if os.getenv("OPENAI_API_KEY") else "missing"
        }

        health_status["tavily_api"] = {
            "status": "configured" if os.getenv("TAVILY_API_KEY") else "missing"
        }

        return health_status

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {"status": "unhealthy", "error": str(e)}


@app.get("/system/status", response_model=SystemStatusResponse)
async def get_system_status(api_key: str = Depends(get_api_key)):
    """Get comprehensive system status and configuration."""
    try:
        container.initialize()

        document_processor = get_service("document_processor")
        validation = validate_document_processor_setup(document_processor)

        # Component status
        components = {}

        # Check vector database
        vector_db_service = get_service("vector_db")
        try:
            vdb_stats = vector_db_service.get_collection_stats()
            components["vector_database"] = {
                "status": "healthy",
                "collection_name": vdb_stats["name"],
                "vector_count": vdb_stats["count"],
                "dimension": vdb_stats["dimension"],
                "real_time_integration": vdb_stats.get(
                    "real_time_integration", False
                ),
            }
        except Exception as e:
            components["vector_database"] = {"status": "error", "error": str(e)}

        # Check embedding service
        embedding_service = get_service("embedding")
        try:
            embedding_stats = embedding_service.get_embedding_stats()
            components["embedding_service"] = {
                "status": "healthy",
                "model": embedding_stats["model"],
                "total_embeddings": embedding_stats["total_embeddings"],
                "intent_recognition": embedding_stats.get(
                    "intent_recognition_enabled", False
                ),
            }
        except Exception as e:
            components["embedding_service"] = {
                "status": "error",
                "error": str(e),
            }

        # Check document processor
        documents = document_processor.list_documents()
        components["document_processor"] = {
            "status": "healthy",
            "processed_documents": len(documents),
            "supported_formats": len(
                document_processor.get_supported_extensions()
            ),
        }

        # Check schedule system
        structured_storage = get_service("structured_storage")
        try:
            if structured_storage:
                struct_stats = structured_storage.get_statistics()
                components["schedule_system"] = {
                    "status": "healthy",
                    "total_entries": struct_stats["total_entries"],
                    "class_groups": struct_stats["unique_class_groups"],
                }
            else:
                components["schedule_system"] = {"status": "disabled"}
        except Exception as e:
            components["schedule_system"] = {"status": "error", "error": str(e)}

        # Check query router
        query_router = get_service("query_router")
        try:
            if query_router:
                router_stats = query_router.get_routing_statistics()
                components["query_router"] = {
                    "status": "healthy",
                    "patterns": router_stats["total_structured_patterns"],
                    "hybrid_support": router_stats["supports_hybrid_queries"],
                }
            else:
                components["query_router"] = {"status": "disabled"}
        except Exception as e:
            components["query_router"] = {"status": "error", "error": str(e)}

        # Configuration
        response_generator = get_service("response_generator")
        configuration = {
            "deduplication_enabled": settings.deduplication.enabled,
            "similarity_threshold": settings.deduplication.similarity_threshold,
            "chunk_size": document_processor.chunk_size,
            "chunk_overlap": document_processor.chunk_overlap,
            "embedding_model": embedding_service.model_name,
            "llm_model": (
                response_generator.model if response_generator else "gpt-4o"
            ),
            "real_time_enabled": get_service("tavily") is not None,
            "schedule_detection": get_service("schedule_extractor") is not None,
        }

        return SystemStatusResponse(
            status="healthy" if validation["valid"] else "issues_detected",
            components=components,
            configuration=configuration,
            validation=validation,
        )

    except Exception as e:
        logger.error(f"Error getting system status: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error getting system status: {str(e)}"
        )


@app.post("/query", response_model=QueryResponse)
async def enhanced_query(
    request: QueryRequest, api_key: str = Depends(get_api_key)
):
    """Enhanced query endpoint with optimized service orchestration."""
    try:
        container.initialize()
        logger.info(f"Processing enhanced query: {request.query}")

        memory_processor = get_service("memory_processor")
        intent_recognizer = get_service("intent_recognizer")
        query_router = get_service("query_router")
        response_generator = get_service("response_generator")
        vector_db_service = get_service("vector_db")

        # Process conversation context if using memory
        context_info = {}
        if request.use_memory and request.conversation_history:
            messages = [msg.dict() for msg in request.conversation_history]
            context_info = memory_processor.process_conversation_context(
                request.query, messages, request.max_history_messages
            )

            if context_info.get("needs_clarification"):
                return QueryResponse(
                    response=context_info.get(
                        "clarification_prompt",
                        "Could you please clarify your question?",
                    ),
                    intent_type="clarification",
                    topic="clarification",
                    confidence=0.8,
                    needs_clarification=True,
                    clarification_prompt=context_info.get(
                        "clarification_prompt"
                    ),
                    context_info=ContextInfo(**context_info),
                )

        # Use smart query routing if available
        if query_router:
            try:
                route_result = query_router.route_query(
                    request.query, context_info
                )

                if route_result.get("answer") or route_result.get(
                    "formatted_response"
                ):
                    response_text = route_result.get(
                        "answer", route_result.get("formatted_response", "")
                    )

                    # Safe timestamp retrieval
                    try:
                        if response_generator and hasattr(
                            response_generator, "get_current_kenya_time"
                        ):
                            current_time = (
                                response_generator.get_current_kenya_time()[0]
                            )
                        else:
                            import pytz
                            from datetime import datetime

                            kenya_tz = pytz.timezone("Africa/Nairobi")
                            current_time = datetime.now(kenya_tz).strftime(
                                "%Y-%m-%d %H:%M:%S EAT"
                            )
                    except Exception:
                        from datetime import datetime

                        current_time = datetime.utcnow().strftime(
                            "%Y-%m-%d %H:%M:%S UTC"
                        )

                    return QueryResponse(
                        response=response_text,
                        intent_type="factual_query",
                        topic="general",
                        confidence=0.8,
                        token_usage={},
                        current_time=current_time,
                        retrieved_chunks=(
                            route_result.get("search_results", [])[:5]
                            if settings.api.debug
                            else None
                        ),
                        context_info=(
                            ContextInfo(**context_info)
                            if context_info
                            else None
                        ),
                    )
            except Exception as e:
                logger.warning(
                    f"Query routing failed, falling back to standard flow: {e}"
                )

        # Fallback to standard processing flow
        intent_info = intent_recognizer.recognize_intent(
            request.query, context_info
        )
        logger.info(f"Recognized intent: {intent_info}")

        # Retrieve relevant context with corrected parameters
        retrieved_chunks = []
        if intent_info["intent_type"] != "off_topic":
            if request.use_multi_query:
                retrieved_chunks = vector_db_service.multi_query_search(
                    query=request.query,
                    top_k=request.top_k,
                    filter_doc_id=request.doc_id,
                )
            else:
                # Use corrected parameter names
                retrieved_chunks = vector_db_service.search(
                    query=request.query,
                    top_k=request.top_k,
                    filter_doc_id=request.doc_id,
                    include_real_time=True,
                )

        # Generate response using optimized response generator
        response_data = response_generator.generate_response(
            query=request.query,
            context_info=context_info,
            use_real_time=True,
        )

        # Prepare response maintaining exact payload structure
        query_response = QueryResponse(
            response=response_data["response"],
            intent_type=intent_info["intent_type"],
            topic=intent_info["topic"],
            confidence=intent_info["confidence"],
            token_usage=response_data.get("token_usage"),
            current_time=response_data.get("timestamp"),
        )

        # Include debug information if enabled
        if settings.api.debug:
            query_response.retrieved_chunks = retrieved_chunks
            if context_info:
                query_response.context_info = ContextInfo(**context_info)

        return query_response

    except Exception as e:
        logger.error(f"Error processing enhanced query: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error processing query: {str(e)}"
        )


@app.post("/multi-query-search")
async def enhanced_multi_query_search(
    request: QueryRequest, api_key: str = Depends(get_api_key)
):
    """Enhanced multi-query search with improved ranking."""
    try:
        container.initialize()
        vector_db_service = get_service("vector_db")

        retrieved_chunks = vector_db_service.multi_query_search(
            query=request.query,
            top_k=request.top_k,
            filter_doc_id=request.doc_id,
        )

        return {
            "query": request.query,
            "chunks": retrieved_chunks,
            "count": len(retrieved_chunks),
            "search_type": "multi_query",
            "top_k": request.top_k,
        }
    except Exception as e:
        logger.error(f"Error in enhanced multi-query search: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error in multi-query search: {str(e)}"
        )


@app.post("/memory/test", response_model=MemoryTestResponse)
async def test_memory_system(
    request: MemoryTestRequest, api_key: str = Depends(get_api_key)
):
    """Test the memory system with conversation scenarios."""
    try:
        container.initialize()
        memory_processor = get_service("memory_processor")
        intent_recognizer = get_service("intent_recognizer")

        messages = [msg.dict() for msg in request.messages]
        context_result = memory_processor.process_conversation_context(
            request.query, messages
        )
        intent_result = intent_recognizer.recognize_intent(
            request.query, context_result
        )

        return MemoryTestResponse(
            context_result=context_result,
            intent_result=intent_result,
            success=True,
            message=f"Memory test completed for scenario: {request.scenario}",
        )

    except Exception as e:
        logger.error(f"Error testing memory system: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error testing memory system: {str(e)}"
        )


@app.post("/documents/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...), api_key: str = Depends(get_api_key)
):
    """Upload and process a document with enhanced features."""
    try:
        import time

        container.initialize()
        start_time = time.time()

        document_processor = get_service("document_processor")
        schedule_extractor = get_service("schedule_extractor")
        structured_storage = get_service("structured_storage")
        embedding_service = get_service("embedding")
        vector_db_service = get_service("vector_db")

        if not document_processor.is_file_supported(Path(file.filename)):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Supported formats: {', '.join(document_processor.get_supported_extensions())}",
            )

        temp_file_path = UPLOAD_DIR / file.filename

        with open(temp_file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        file_size = temp_file_path.stat().st_size

        # Check if it's a schedule document
        is_schedule, schedule_confidence = (
            schedule_extractor.detect_schedule_document(temp_file_path)
        )

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

        # Handle schedule documents
        if is_schedule:
            schedule_entries = schedule_extractor.extract_from_file(
                temp_file_path
            )
            if schedule_entries:
                schedule_dicts = [entry.to_dict() for entry in schedule_entries]
                structured_storage.store_schedules(
                    schedule_dicts,
                    metadata["doc_id"],
                    metadata["file_name"],
                    str(temp_file_path),
                )

        embedding_service.embed_chunks(metadata["chunks_path"])
        vector_db_service.index_chunks(metadata["chunks_path"])

        os.remove(temp_file_path)
        processing_time = time.time() - start_time

        return DocumentResponse(
            doc_id=metadata["doc_id"],
            file_name=metadata["file_name"],
            doc_type=metadata["doc_type"],
            num_chunks=metadata["num_chunks"],
            file_size=file_size,
            processing_time=processing_time,
            success=True,
            message="Document successfully processed and indexed",
        )

    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        if "temp_file_path" in locals() and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(
            status_code=500, detail=f"Error uploading document: {str(e)}"
        )


@app.post("/documents/process-url", response_model=DocumentResponse)
async def process_url(
    request: URLProcessRequest, api_key: str = Depends(get_api_key)
):
    """Process content from a URL."""
    try:
        import time

        container.initialize()
        start_time = time.time()

        document_processor = get_service("document_processor")
        embedding_service = get_service("embedding")
        vector_db_service = get_service("vector_db")

        logger.info(f"Processing URL: {request.url}")
        metadata = document_processor.process_url(
            request.url, request.output_name
        )

        if not metadata:
            return JSONResponse(
                status_code=500,
                content={"success": False, "message": "Failed to process URL"},
            )

        embedding_service.embed_chunks(metadata["chunks_path"])
        vector_db_service.index_chunks(metadata["chunks_path"])

        processing_time = time.time() - start_time

        return DocumentResponse(
            doc_id=metadata["doc_id"],
            file_name=metadata["file_name"],
            doc_type=metadata["doc_type"],
            num_chunks=metadata["num_chunks"],
            processing_time=processing_time,
            success=True,
            message="URL successfully processed and indexed",
        )

    except Exception as e:
        logger.error(f"Error processing URL: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error processing URL: {str(e)}"
        )


@app.post("/documents/process-sitemap")
async def process_sitemap(
    request: SitemapProcessRequest, api_key: str = Depends(get_api_key)
):
    """Process multiple pages from a sitemap."""
    try:
        import time

        container.initialize()
        start_time = time.time()

        document_processor = get_service("document_processor")
        embedding_service = get_service("embedding")
        vector_db_service = get_service("vector_db")

        logger.info(f"Processing sitemap: {request.sitemap_url}")
        results = document_processor.process_sitemap(
            request.sitemap_url, request.max_pages
        )

        if not results:
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "message": "Failed to process sitemap",
                },
            )

        for metadata in results:
            embedding_service.embed_chunks(metadata["chunks_path"])
            vector_db_service.index_chunks(metadata["chunks_path"])

        processing_time = time.time() - start_time

        return {
            "success": True,
            "message": f"Sitemap successfully processed with {len(results)} pages",
            "pages_processed": len(results),
            "processing_time": processing_time,
            "results": [
                {
                    "doc_id": r["doc_id"],
                    "file_name": r["file_name"],
                    "num_chunks": r["num_chunks"],
                }
                for r in results
            ],
        }

    except Exception as e:
        logger.error(f"Error processing sitemap: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error processing sitemap: {str(e)}"
        )


@app.delete("/documents/{doc_id}", response_model=DocumentResponse)
async def delete_document(doc_id: str, api_key: str = Depends(get_api_key)):
    """Delete a document and its associated chunks."""
    try:
        container.initialize()
        document_processor = get_service("document_processor")
        vector_db_service = get_service("vector_db")

        doc_info = document_processor.get_document_info(doc_id)
        if not doc_info:
            return JSONResponse(
                status_code=404,
                content={
                    "success": False,
                    "message": f"Document {doc_id} not found",
                },
            )

        success = document_processor.delete_document(doc_id)
        if success:
            vector_db_service.delete_document(doc_id)

        if not success:
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "message": f"Failed to delete document {doc_id}",
                },
            )

        return DocumentResponse(
            doc_id=doc_id,
            file_name=doc_info["file_name"],
            doc_type=doc_info["doc_type"],
            num_chunks=doc_info["num_chunks"],
            success=True,
            message="Document successfully deleted",
        )

    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error deleting document: {str(e)}"
        )


@app.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    doc_type: Optional[str] = Query(
        None, description="Filter by document type"
    ),
    search_query: Optional[str] = Query(
        None, description="Search documents by name or type"
    ),
    api_key: str = Depends(get_api_key),
):
    """List all processed documents with enhanced filtering."""
    try:
        container.initialize()
        document_processor = get_service("document_processor")

        if search_query:
            documents = document_processor.search_documents(search_query)
        elif doc_type:
            documents = document_processor.get_documents_by_type(doc_type)
        else:
            documents = document_processor.list_documents()

        total_chunks = sum(doc.get("num_chunks", 0) for doc in documents)
        supported_formats = document_processor.get_supported_extensions()

        return DocumentListResponse(
            documents=documents,
            count=len(documents),
            total_chunks=total_chunks,
            supported_formats=supported_formats,
        )

    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error listing documents: {str(e)}"
        )


@app.get("/documents/{doc_id}", response_model=DocumentInfoResponse)
async def get_document_info(doc_id: str, api_key: str = Depends(get_api_key)):
    """Get detailed information about a specific document."""
    try:
        container.initialize()
        document_processor = get_service("document_processor")

        doc_info = document_processor.get_document_info(doc_id)

        if not doc_info:
            return JSONResponse(
                status_code=404,
                content={
                    "success": False,
                    "message": f"Document {doc_id} not found",
                },
            )

        return DocumentInfoResponse(**doc_info, success=True)

    except Exception as e:
        logger.error(f"Error getting document info: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error getting document info: {str(e)}"
        )


@app.post("/documents/update", response_model=DocumentResponse)
async def update_document(
    file: UploadFile = File(...), api_key: str = Depends(get_api_key)
):
    """Update an existing document with enhanced features."""
    try:
        import time

        container.initialize()
        start_time = time.time()

        document_processor = get_service("document_processor")
        embedding_service = get_service("embedding")
        vector_db_service = get_service("vector_db")

        temp_file_path = UPLOAD_DIR / file.filename

        with open(temp_file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        file_size = temp_file_path.stat().st_size

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

        embedding_service.embed_chunks(metadata["chunks_path"])
        vector_db_service.index_chunks(metadata["chunks_path"])

        os.remove(temp_file_path)
        processing_time = time.time() - start_time

        return DocumentResponse(
            doc_id=metadata["doc_id"],
            file_name=metadata["file_name"],
            doc_type=metadata["doc_type"],
            num_chunks=metadata["num_chunks"],
            file_size=file_size,
            processing_time=processing_time,
            success=True,
            message="Document successfully updated and indexed",
        )

    except Exception as e:
        logger.error(f"Error updating document: {str(e)}")
        if "temp_file_path" in locals() and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(
            status_code=500, detail=f"Error updating document: {str(e)}"
        )


@app.post("/system/rebuild-index")
async def rebuild_index(api_key: str = Depends(get_api_key)):
    """Rebuild the vector database index with all processed documents."""
    try:
        container.initialize()
        document_processor = get_service("document_processor")
        vector_db_service = get_service("vector_db")

        documents = document_processor.list_documents()

        if not documents:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "message": "No documents found to index",
                },
            )

        vector_db_service.initialize_collection(recreate=True)

        indexed_count = 0
        for doc in documents:
            vector_db_service.index_chunks(doc["chunks_path"])
            indexed_count += 1

        final_stats = vector_db_service.get_collection_stats()

        return {
            "success": True,
            "message": f"Vector database index rebuilt successfully with {indexed_count} documents",
            "indexed_documents": indexed_count,
            "total_documents": len(documents),
            "final_vector_count": final_stats["count"],
        }

    except Exception as e:
        logger.error(f"Error rebuilding index: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error rebuilding index: {str(e)}"
        )


@app.get(
    "/system/deduplication-status", response_model=DeduplicationStatusResponse
)
async def get_deduplication_status(api_key: str = Depends(get_api_key)):
    """Get enhanced deduplication status and statistics."""
    try:
        container.initialize()
        document_processor = get_service("document_processor")

        dedup_dir = document_processor.dedup_dir
        dedup_file = dedup_dir / "deduplicated_chunks.jsonl"
        report_file = dedup_dir / "deduplication_report.json"

        response_data = DeduplicationStatusResponse(
            enabled=settings.deduplication.enabled,
            similarity_threshold=settings.deduplication.similarity_threshold,
            deduplicated_chunks_found=dedup_file.exists(),
        )

        if dedup_file.exists():
            chunk_count = 0
            with open(dedup_file, "r") as f:
                for _ in f:
                    chunk_count += 1
            response_data.chunk_count = chunk_count

            if report_file.exists():
                with open(report_file, "r") as f:
                    report = json.load(f)
                response_data.stats = report.get("stats", {})
                response_data.quality_metrics = report.get(
                    "quality_metrics", {}
                )

        return response_data

    except Exception as e:
        logger.error(f"Error getting deduplication status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting deduplication status: {str(e)}",
        )


@app.post("/system/run-deduplication")
async def run_deduplication(api_key: str = Depends(get_api_key)):
    """Run the enhanced deduplication process on existing chunks."""
    try:
        container.initialize()
        document_processor = get_service("document_processor")

        if not settings.deduplication.enabled:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "message": "Deduplication is disabled in settings",
                },
            )

        import time

        start_time = time.time()

        chunks_dir = document_processor.chunk_dir
        chunk_files = list(chunks_dir.glob("*_chunks.jsonl"))

        if not chunk_files:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "message": "No chunk files found to deduplicate",
                },
            )

        all_chunks = []
        for chunk_file in chunk_files:
            with open(chunk_file, "r") as f:
                for line in f:
                    try:
                        chunk_data = json.loads(line)
                        from services.document_processor import Chunk

                        chunk = Chunk(
                            chunk_id=chunk_data["chunk_id"],
                            doc_id=chunk_data["doc_id"],
                            chunk_index=chunk_data["chunk_index"],
                            text=chunk_data["text"],
                            metadata=chunk_data.get("metadata", {}),
                            information_score=chunk_data.get(
                                "information_score", 0.0
                            ),
                            source_file=str(chunk_file),
                        )
                        all_chunks.append(chunk)
                    except json.JSONDecodeError:
                        continue

        if not all_chunks:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "message": "No valid chunks found to deduplicate",
                },
            )

        document_processor.all_chunks = all_chunks
        document_processor._deduplicate_chunks()

        deduplicated_path = (
            document_processor.dedup_dir / "deduplicated_chunks.jsonl"
        )
        document_processor._save_deduplicated_chunks(deduplicated_path)
        document_processor._generate_deduplication_report()

        processing_time = time.time() - start_time

        original_count = len(all_chunks)
        final_count = len(document_processor.deduplicated_chunks)
        reduction_percentage = (
            ((original_count - final_count) / original_count * 100)
            if original_count > 0
            else 0
        )

        return {
            "success": True,
            "message": "Enhanced deduplication completed successfully",
            "original_chunks": original_count,
            "final_chunks": final_count,
            "reduction_percentage": reduction_percentage,
            "processing_time": processing_time,
            "files_processed": len(chunk_files),
        }

    except Exception as e:
        logger.error(f"Error running deduplication: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error running deduplication: {str(e)}"
        )


@app.post("/system/embed-deduplicated")
async def embed_deduplicated_chunks(api_key: str = Depends(get_api_key)):
    """Generate embeddings for deduplicated chunks."""
    try:
        container.initialize()
        document_processor = get_service("document_processor")
        embedding_service = get_service("embedding")

        dedup_file = document_processor.dedup_dir / "deduplicated_chunks.jsonl"
        if not dedup_file.exists():
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "message": "Deduplicated chunks not found. Run deduplication first.",
                },
            )

        import time

        start_time = time.time()

        chunk_count = 0
        with open(dedup_file, "r") as f:
            for _ in f:
                chunk_count += 1

        embeddings_dict = embedding_service.embed_chunks(dedup_file)
        processing_time = time.time() - start_time

        return {
            "success": True,
            "message": "Embeddings generated for deduplicated chunks",
            "chunk_count": chunk_count,
            "embeddings_generated": len(embeddings_dict),
            "processing_time": processing_time,
        }

    except Exception as e:
        logger.error(f"Error embedding deduplicated chunks: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error embedding deduplicated chunks: {str(e)}",
        )


@app.post("/system/index-deduplicated")
async def index_deduplicated_chunks(api_key: str = Depends(get_api_key)):
    """Index deduplicated chunks in vector database."""
    try:
        container.initialize()
        document_processor = get_service("document_processor")
        vector_db_service = get_service("vector_db")

        dedup_file = document_processor.dedup_dir / "deduplicated_chunks.jsonl"
        embeddings_file = (
            document_processor.dedup_dir.parent
            / "embeddings"
            / "deduplicated_embeddings.npz"
        )

        if not dedup_file.exists():
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "message": "Deduplicated chunks not found. Run deduplication first.",
                },
            )

        if not embeddings_file.exists():
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "message": "Deduplicated embeddings not found. Generate embeddings first.",
                },
            )

        initial_stats = vector_db_service.get_collection_stats()
        vector_db_service.index_chunks(dedup_file)
        final_stats = vector_db_service.get_collection_stats()
        added_vectors = final_stats["count"] - initial_stats["count"]

        return {
            "success": True,
            "message": "Deduplicated chunks indexed successfully",
            "initial_vectors": initial_stats["count"],
            "final_vectors": final_stats["count"],
            "added_vectors": added_vectors,
        }

    except Exception as e:
        logger.error(f"Error indexing deduplicated chunks: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error indexing deduplicated chunks: {str(e)}",
        )


@app.post("/evaluation/run", response_model=EvaluationResponse)
async def run_evaluation(
    request: EvaluationRequest, api_key: str = Depends(get_api_key)
):
    """Run comprehensive system evaluation."""
    try:
        import time

        container.initialize()
        evaluator = get_service("evaluator")
        start_time = time.time()

        eval_file = None
        if request.create_template:
            eval_file = evaluator.create_eval_set()
            logger.info(f"Created evaluation template at: {eval_file}")

        if request.questions:
            import tempfile
            import pandas as pd

            temp_file = tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False
            )
            df = pd.DataFrame(request.questions)
            df.to_csv(temp_file.name, index=False)
            eval_file = Path(temp_file.name)

        eval_results = evaluator.run_evaluation(eval_file)

        if eval_results["status"] == "error":
            return JSONResponse(
                status_code=500,
                content={"success": False, "message": eval_results["message"]},
            )

        report_path = evaluator.generate_report(eval_results)

        processing_time = time.time() - start_time
        eval_results["processing_time"] = processing_time
        eval_results["report_path"] = str(report_path)

        return EvaluationResponse(**eval_results)

    except Exception as e:
        logger.error(f"Error running evaluation: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error running evaluation: {str(e)}"
        )


@app.get("/evaluation/create-template")
async def create_evaluation_template(api_key: str = Depends(get_api_key)):
    """Create a sample evaluation template."""
    try:
        container.initialize()
        evaluator = get_service("evaluator")

        template_path = evaluator.create_eval_set()

        return {
            "success": True,
            "message": "Evaluation template created successfully",
            "template_path": str(template_path),
            "instructions": "Edit the CSV file to add your evaluation questions, then use the run evaluation endpoint",
        }

    except Exception as e:
        logger.error(f"Error creating evaluation template: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error creating evaluation template: {str(e)}",
        )


@app.get("/stats/embeddings")
async def get_embedding_stats(api_key: str = Depends(get_api_key)):
    """Get comprehensive embedding statistics."""
    try:
        container.initialize()
        embedding_service = get_service("embedding")

        stats = embedding_service.get_embedding_stats()

        embeddings_dir = embedding_service.embeddings_dir
        if embeddings_dir.exists():
            embedding_files = list(embeddings_dir.glob("*.npz"))
            total_size = sum(f.stat().st_size for f in embedding_files)
            stats["total_files"] = len(embedding_files)
            stats["total_size_mb"] = total_size / (1024 * 1024)

        return {"success": True, "stats": stats}

    except Exception as e:
        logger.error(f"Error getting embedding stats: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error getting embedding stats: {str(e)}"
        )


@app.get("/stats/vector-db")
async def get_vector_db_stats(api_key: str = Depends(get_api_key)):
    """Get comprehensive vector database statistics."""
    try:
        container.initialize()
        vector_db_service = get_service("vector_db")

        stats = vector_db_service.get_collection_stats()

        test_status = "healthy"
        try:
            test_results = vector_db_service.search("test query", top_k=1)
            if test_results:
                test_status = "responsive"
            else:
                test_status = "empty"
        except Exception:
            test_status = "error"

        db_size_mb = 0
        try:
            db_files = list(vector_db_service.db_path.rglob("*"))
            total_size = sum(f.stat().st_size for f in db_files if f.is_file())
            db_size_mb = total_size / (1024 * 1024)
        except Exception:
            pass

        stats.update(
            {
                "test_status": test_status,
                "db_path": str(vector_db_service.db_path),
                "db_size_mb": db_size_mb,
            }
        )

        return {"success": True, "stats": stats}

    except Exception as e:
        logger.error(f"Error getting vector database stats: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting vector database stats: {str(e)}",
        )


@app.get("/stats/processing")
async def get_processing_stats(api_key: str = Depends(get_api_key)):
    """Get comprehensive document processing statistics."""
    try:
        container.initialize()
        document_processor = get_service("document_processor")

        stats = document_processor.get_processing_stats()

        documents = document_processor.list_documents()
        doc_types = {}
        total_chunks = 0
        total_size = 0

        for doc in documents:
            doc_type = doc.get("doc_type", "unknown")
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
            total_chunks += doc.get("num_chunks", 0)
            total_size += doc.get("file_size", 0)

        stats.update(
            {
                "document_types": doc_types,
                "total_chunks": total_chunks,
                "total_size_mb": total_size / (1024 * 1024),
            }
        )

        return {"success": True, "stats": stats}

    except Exception as e:
        logger.error(f"Error getting processing stats: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error getting processing stats: {str(e)}"
        )


@app.get("/supported-formats")
async def get_supported_formats(api_key: str = Depends(get_api_key)):
    """Get all supported document formats and their capabilities."""
    try:
        container.initialize()
        document_processor = get_service("document_processor")

        extensions = document_processor.get_supported_extensions()

        formats_info = []
        for ext in sorted(extensions):
            loader_info = document_processor.loader_mapping.get(ext, {})
            doc_type = loader_info.get("doc_type", "unknown")

            loader_class = loader_info.get("loader_class")
            if callable(loader_class):
                loader_name = (
                    loader_class.__name__
                    if hasattr(loader_class, "__name__")
                    else "Dynamic"
                )
            else:
                loader_name = (
                    loader_class.__name__ if loader_class else "Unknown"
                )

            has_fallback = bool(loader_info.get("fallback_loader"))

            formats_info.append(
                {
                    "extension": ext,
                    "doc_type": doc_type,
                    "primary_loader": loader_name,
                    "has_fallback": has_fallback,
                }
            )

        capabilities = {
            "web_url_processing": True,
            "sitemap_processing": True,
            "deduplication": document_processor.enable_deduplication,
            "information_scoring": True,
            "batch_processing": True,
            "conversation_memory": True,
        }

        return {
            "success": True,
            "total_formats": len(extensions),
            "formats": formats_info,
            "pdf_strategy": document_processor.pdf_loader_strategy,
            "capabilities": capabilities,
        }

    except Exception as e:
        logger.error(f"Error getting supported formats: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error getting supported formats: {str(e)}"
        )


@app.post("/system/cleanup")
async def cleanup_system(api_key: str = Depends(get_api_key)):
    """Clean up temporary files and optimize storage."""
    try:
        container.initialize()
        document_processor = get_service("document_processor")
        embedding_service = get_service("embedding")
        vector_db_service = get_service("vector_db")

        cleaned_files = 0
        freed_space = 0

        temp_patterns = [
            "*.tmp",
            "*.temp",
            "*_temp.*",
            "*.log",
            ".DS_Store",
            "Thumbs.db",
        ]

        directories_to_clean = [
            document_processor.processed_dir,
            document_processor.chunk_dir,
            document_processor.dedup_dir,
            embedding_service.embeddings_dir,
            vector_db_service.db_path,
            UPLOAD_DIR,
        ]

        for directory in directories_to_clean:
            if directory.exists():
                for pattern in temp_patterns:
                    for temp_file in directory.rglob(pattern):
                        if temp_file.is_file():
                            try:
                                file_size = temp_file.stat().st_size
                                temp_file.unlink()
                                cleaned_files += 1
                                freed_space += file_size
                            except Exception:
                                continue

        freed_mb = freed_space / (1024 * 1024)

        return {
            "success": True,
            "message": "System cleanup completed",
            "files_removed": cleaned_files,
            "space_freed_mb": freed_mb,
        }

    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error during cleanup: {str(e)}"
        )


@app.get("/system/export-config")
async def export_system_config(api_key: str = Depends(get_api_key)):
    """Export current system configuration."""
    try:
        from datetime import datetime

        container.initialize()
        document_processor = get_service("document_processor")
        embedding_service = get_service("embedding")
        vector_db_service = get_service("vector_db")
        response_generator = get_service("response_generator")
        memory_processor = get_service("memory_processor")

        config_data = {
            "system_info": {
                "timestamp": datetime.now().isoformat(),
                "api_version": "2.1.0",
                "system_type": "enhanced_rag_system",
            },
            "document_processor": {
                "chunk_size": document_processor.chunk_size,
                "chunk_overlap": document_processor.chunk_overlap,
                "similarity_threshold": document_processor.similarity_threshold,
                "deduplication_enabled": document_processor.enable_deduplication,
                "pdf_strategy": document_processor.pdf_loader_strategy,
                "supported_formats": document_processor.get_supported_extensions(),
                "directories": {
                    "raw_dir": str(document_processor.raw_dir),
                    "processed_dir": str(document_processor.processed_dir),
                    "chunk_dir": str(document_processor.chunk_dir),
                    "dedup_dir": str(document_processor.dedup_dir),
                },
            },
            "embedding_service": {
                "model": embedding_service.model_name,
                "dimension": embedding_service.dimension,
                "stats": embedding_service.get_embedding_stats(),
            },
            "vector_database": {
                "collection_name": vector_db_service.collection_name,
                "db_path": str(vector_db_service.db_path),
                "stats": vector_db_service.get_collection_stats(),
            },
            "llm_settings": {
                "model": response_generator.model,
                "temperature": response_generator.temperature,
                "max_tokens": response_generator.max_tokens,
            },
            "memory_processor": {
                "max_context_tokens": memory_processor.max_context_tokens,
                "memory_window": memory_processor.memory_window,
                "ambiguity_threshold": memory_processor.ambiguity_threshold,
            },
            "api_settings": {"debug": settings.api.debug, "cors_enabled": True},
        }

        return {
            "success": True,
            "message": "System configuration exported",
            "config": config_data,
            "export_timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error exporting configuration: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error exporting configuration: {str(e)}"
        )


@app.get("/schedule/stats")
async def get_schedule_stats(api_key: str = Depends(get_api_key)):
    """Get comprehensive schedule system statistics."""
    try:
        container.initialize()
        structured_storage = get_service("structured_storage")
        query_router = get_service("query_router")

        if not structured_storage:
            return JSONResponse(
                status_code=503,
                content={
                    "success": False,
                    "message": "Schedule system not available",
                },
            )

        stats = structured_storage.get_statistics()

        response_data = {
            "success": True,
            "schedule_system": {
                "total_documents": stats.get("documents_processed", 0),
                "total_entries": stats["total_entries"],
                "unique_class_groups": stats["unique_class_groups"],
                "unique_subjects": stats["unique_subjects"],
                "unique_rooms": stats["unique_rooms"],
                "unique_instructors": stats["unique_instructors"],
            },
        }

        if query_router:
            router_stats = query_router.get_routing_statistics()
            response_data["query_router"] = {
                "structured_patterns": router_stats[
                    "total_structured_patterns"
                ],
                "hybrid_support": router_stats["supports_hybrid_queries"],
                "fallback_available": router_stats["fallback_available"],
            }

        return response_data

    except Exception as e:
        logger.error(f"Error getting schedule stats: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error getting schedule stats: {str(e)}"
        )


@app.post("/schedule/query")
async def query_schedule(
    query_text: str = Query(..., description="Natural language schedule query"),
    show_sql: bool = Query(False, description="Show generated SQL query"),
    api_key: str = Depends(get_api_key),
):
    """Query schedule data using natural language with smart routing."""
    try:
        container.initialize()
        query_router = get_service("query_router")
        response_generator = get_service("response_generator")

        if not query_router:
            if not response_generator:
                return JSONResponse(
                    status_code=503,
                    content={
                        "success": False,
                        "message": "Query processing services not available",
                    },
                )

            response_data = response_generator.generate_response(query_text)
            return {
                "success": True,
                "response": response_data.get(
                    "response", "No response generated"
                ),
                "approach": "fallback",
                "result_count": 0,
            }

        analysis = query_router.analyze_query(query_text)
        route_result = query_router.route_query(query_text)

        response_data = {
            "success": True,
            "query": query_text,
            "analysis": {
                "type": analysis.query_type.value,
                "confidence": analysis.confidence,
                "explanation": analysis.explanation,
            },
            "response": route_result.get(
                "answer",
                route_result.get("formatted_response", "No response available"),
            ),
            "approach": route_result.get("routing_info", {}).get(
                "primary_approach", "unknown"
            ),
            "result_count": route_result.get("result_count", 0),
        }

        if show_sql and route_result.get("sql_query"):
            response_data["sql_query"] = route_result["sql_query"]

        return response_data

    except Exception as e:
        logger.error(f"Error processing schedule query: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error processing schedule query: {str(e)}"
        )


@app.post("/real-time/test")
async def test_real_time_integration(
    query: str = Query(
        ..., description="Query to test with real-time information"
    ),
    api_key: str = Depends(get_api_key),
):
    """Test real-time information retrieval using Tavily."""
    try:
        container.initialize()
        tavily_service = get_service("tavily")
        vector_db_service = get_service("vector_db")

        if not tavily_service:
            return JSONResponse(
                status_code=503,
                content={
                    "success": False,
                    "message": "Real-time service not available - check TAVILY_API_KEY",
                },
            )

        direct_result = tavily_service.search(
            query=query, max_results=2, topic="general"
        )

        test_result = {
            "success": True,
            "query": query,
            "direct_search_results": len(direct_result.get("results", [])),
            "direct_results": direct_result.get("results", []),
        }

        if vector_db_service:
            integration_test = vector_db_service.test_real_time_integration(
                query
            )
            test_result["integration_test"] = integration_test

        return test_result

    except Exception as e:
        logger.error(f"Error testing real-time features: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error testing real-time features: {str(e)}",
        )


@app.post("/schedule/upload")
async def upload_schedule_file(
    file: UploadFile = File(...),
    force_schedule: bool = Query(
        False, description="Force processing as schedule"
    ),
    api_key: str = Depends(get_api_key),
):
    """Upload and process a schedule document with enhanced extraction."""
    try:
        import time

        container.initialize()
        start_time = time.time()

        document_processor = get_service("document_processor")
        schedule_extractor = get_service("schedule_extractor")
        structured_storage = get_service("structured_storage")
        embedding_service = get_service("embedding")
        vector_db_service = get_service("vector_db")

        temp_file_path = UPLOAD_DIR / file.filename

        with open(temp_file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        file_size = temp_file_path.stat().st_size

        if force_schedule:
            original_detect = schedule_extractor.detect_schedule_document
            schedule_extractor.detect_schedule_document = lambda x, y=None: (
                True,
                0.9,
            )

        try:
            schedule_entries = schedule_extractor.extract_from_file(
                temp_file_path
            )

            if schedule_entries:
                import hashlib

                doc_id = hashlib.md5(str(temp_file_path).encode()).hexdigest()

                schedule_dicts = [entry.to_dict() for entry in schedule_entries]
                stored_count = structured_storage.store_schedules(
                    schedule_dicts, doc_id, file.filename, str(temp_file_path)
                )

                semantic_chunks = schedule_extractor.convert_to_semantic_chunks(
                    schedule_entries
                )

                metadata = document_processor.process_document(temp_file_path)
                if metadata:
                    embedding_service.embed_chunks(metadata["chunks_path"])
                    vector_db_service.index_chunks(metadata["chunks_path"])

                processing_time = time.time() - start_time

                return {
                    "success": True,
                    "message": "Schedule file processed successfully",
                    "doc_id": doc_id,
                    "file_name": file.filename,
                    "schedule_entries": len(schedule_entries),
                    "structured_entries": stored_count,
                    "semantic_chunks": len(semantic_chunks),
                    "file_size": file_size,
                    "processing_time": processing_time,
                }
            else:
                return JSONResponse(
                    status_code=400,
                    content={
                        "success": False,
                        "message": "No schedule data found in file",
                    },
                )

        finally:
            if force_schedule:
                schedule_extractor.detect_schedule_document = original_detect
            if temp_file_path.exists():
                os.remove(temp_file_path)

    except Exception as e:
        logger.error(f"Error processing schedule file: {str(e)}")
        if "temp_file_path" in locals() and temp_file_path.exists():
            os.remove(temp_file_path)
        raise HTTPException(
            status_code=500, detail=f"Error processing schedule file: {str(e)}"
        )


@app.get("/schedule/class/{class_group}")
async def get_class_schedule(
    class_group: str,
    day: Optional[str] = Query(None, description="Filter by specific day"),
    api_key: str = Depends(get_api_key),
):
    """Get schedule for a specific class group."""
    try:
        container.initialize()
        structured_storage = get_service("structured_storage")

        if not structured_storage:
            return JSONResponse(
                status_code=503,
                content={
                    "success": False,
                    "message": "Schedule system not available",
                },
            )

        schedule_data = structured_storage.get_class_schedule(class_group, day)

        return {
            "success": True,
            "class_group": class_group,
            "day_filter": day,
            "schedule_count": len(schedule_data),
            "schedule": schedule_data,
        }

    except Exception as e:
        logger.error(f"Error getting class schedule: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error getting class schedule: {str(e)}"
        )


@app.get("/schedule/instructor/{instructor}")
async def get_instructor_schedule(
    instructor: str, api_key: str = Depends(get_api_key)
):
    """Get all classes taught by a specific instructor."""
    try:
        container.initialize()
        structured_storage = get_service("structured_storage")

        if not structured_storage:
            return JSONResponse(
                status_code=503,
                content={
                    "success": False,
                    "message": "Schedule system not available",
                },
            )

        schedule_data = structured_storage.get_instructor_schedule(instructor)

        return {
            "success": True,
            "instructor": instructor,
            "classes_count": len(schedule_data),
            "schedule": schedule_data,
        }

    except Exception as e:
        logger.error(f"Error getting instructor schedule: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting instructor schedule: {str(e)}",
        )


@app.get("/schedule/room/{room}")
async def get_room_schedule(room: str, api_key: str = Depends(get_api_key)):
    """Get all classes scheduled in a specific room."""
    try:
        container.initialize()
        structured_storage = get_service("structured_storage")

        if not structured_storage:
            return JSONResponse(
                status_code=503,
                content={
                    "success": False,
                    "message": "Schedule system not available",
                },
            )

        schedule_data = structured_storage.get_room_schedule(room)

        return {
            "success": True,
            "room": room,
            "classes_count": len(schedule_data),
            "schedule": schedule_data,
        }

    except Exception as e:
        logger.error(f"Error getting room schedule: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error getting room schedule: {str(e)}"
        )


@app.get("/schedule/available-rooms")
async def get_available_rooms(
    day: str = Query(..., description="Day of the week"),
    start_time: str = Query(..., description="Start time in HH:MM format"),
    end_time: str = Query(..., description="End time in HH:MM format"),
    api_key: str = Depends(get_api_key),
):
    """Get rooms available during a specific time slot."""
    try:
        container.initialize()
        structured_storage = get_service("structured_storage")

        if not structured_storage:
            return JSONResponse(
                status_code=503,
                content={
                    "success": False,
                    "message": "Schedule system not available",
                },
            )

        available_rooms = structured_storage.get_available_rooms(
            day, start_time, end_time
        )

        return {
            "success": True,
            "day": day,
            "time_slot": f"{start_time}-{end_time}",
            "available_rooms_count": len(available_rooms),
            "available_rooms": available_rooms,
        }

    except Exception as e:
        logger.error(f"Error getting available rooms: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error getting available rooms: {str(e)}"
        )


@app.delete("/schedule/document/{doc_id}")
async def delete_schedule_document(
    doc_id: str, api_key: str = Depends(get_api_key)
):
    """Delete all schedule entries for a specific document."""
    try:
        container.initialize()
        structured_storage = get_service("structured_storage")

        if not structured_storage:
            return JSONResponse(
                status_code=503,
                content={
                    "success": False,
                    "message": "Schedule system not available",
                },
            )

        deleted_count = structured_storage.delete_document_schedules(doc_id)

        return {
            "success": True,
            "message": f"Deleted {deleted_count} schedule entries for document {doc_id}",
            "doc_id": doc_id,
            "deleted_entries": deleted_count,
        }

    except Exception as e:
        logger.error(f"Error deleting schedule document: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting schedule document: {str(e)}",
        )


@app.post("/vector-db/optimize")
async def optimize_vector_database(api_key: str = Depends(get_api_key)):
    """Optimize vector database performance and storage."""
    try:
        container.initialize()
        vector_db_service = get_service("vector_db")

        optimization_report = vector_db_service.optimize_collection()

        return {
            "success": True,
            "message": "Vector database optimization completed",
            "optimization_report": optimization_report,
        }

    except Exception as e:
        logger.error(f"Error optimizing vector database: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error optimizing vector database: {str(e)}",
        )


@app.get("/vector-db/similar-chunks/{chunk_id}")
async def get_similar_chunks(
    chunk_id: str,
    top_k: int = Query(5, description="Number of similar chunks to return"),
    api_key: str = Depends(get_api_key),
):
    """Find chunks similar to a given chunk."""
    try:
        container.initialize()
        vector_db_service = get_service("vector_db")

        similar_chunks = vector_db_service.get_similar_chunks(chunk_id, top_k)

        return {
            "success": True,
            "source_chunk_id": chunk_id,
            "similar_chunks_count": len(similar_chunks),
            "similar_chunks": similar_chunks,
        }

    except Exception as e:
        logger.error(f"Error finding similar chunks: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error finding similar chunks: {str(e)}"
        )


@app.get("/embeddings/validate")
async def validate_embeddings_integrity(api_key: str = Depends(get_api_key)):
    """Validate embeddings integrity and consistency."""
    try:
        container.initialize()
        embedding_service = get_service("embedding")

        validation_results = embedding_service.validate_embeddings_integrity()

        return {
            "success": True,
            "validation_results": validation_results,
        }

    except Exception as e:
        logger.error(f"Error validating embeddings: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error validating embeddings: {str(e)}"
        )


@app.post("/embeddings/optimize")
async def optimize_embedding_storage(api_key: str = Depends(get_api_key)):
    """Optimize embedding storage and clean up unused files."""
    try:
        container.initialize()
        embedding_service = get_service("embedding")

        optimization_results = embedding_service.optimize_storage()

        return {
            "success": True,
            "message": "Embedding storage optimization completed",
            "optimization_results": optimization_results,
        }

    except Exception as e:
        logger.error(f"Error optimizing embedding storage: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error optimizing embedding storage: {str(e)}",
        )


@app.get("/system/health-detailed")
async def detailed_health_check(api_key: str = Depends(get_api_key)):
    """Get detailed health check of all system components."""
    try:
        container.initialize()

        health_report = container.health_check()
        vector_db_service = get_service("vector_db")

        if vector_db_service:
            try:
                service_health = vector_db_service.get_service_health()
                health_report["service_health"] = service_health
            except Exception as e:
                health_report["service_health_error"] = str(e)

        api_health = {
            "api_version": "2.1.0",
            "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
            "tavily_configured": bool(os.getenv("TAVILY_API_KEY")),
            "debug_mode": settings.api.debug,
            "cors_enabled": True,
        }

        return {
            "success": True,
            "timestamp": "2024-11-25T12:00:00Z",
            "api_health": api_health,
            "service_health": health_report,
        }

    except Exception as e:
        logger.error(f"Error in detailed health check: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error in detailed health check: {str(e)}"
        )


@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "success": False,
            "message": "Endpoint not found",
            "path": str(request.url.path),
        },
    )


@app.exception_handler(422)
async def validation_error_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "message": "Validation error",
            "details": exc.errors() if hasattr(exc, "errors") else str(exc),
        },
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "Internal server error",
            "error": str(exc) if settings.api.debug else "An error occurred",
        },
    )


if __name__ == "__main__":
    import uvicorn

    if not os.getenv("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY environment variable not set")

    if not os.getenv("TAVILY_API_KEY"):
        logger.warning("TAVILY_API_KEY not set - real-time features disabled")

    try:
        container.initialize()
        document_processor = get_service("document_processor")
        validation = validate_document_processor_setup(document_processor)

        if validation["errors"]:
            logger.error("System validation errors detected!")
            for error in validation["errors"]:
                logger.error(f"  - {error}")
        if validation["warnings"]:
            logger.warning("System validation warnings:")
            for warning in validation["warnings"]:
                logger.warning(f"  - {warning}")

        features = []
        if get_service("tavily"):
            features.append("real-time information")
        if get_service("schedule_extractor"):
            features.append("schedule detection")
        if get_service("structured_storage"):
            features.append("structured queries")
        if get_service("query_router"):
            features.append("smart routing")

        feature_str = ", ".join(features) if features else "basic features"
        logger.info(f"Enhanced RAG system ready with {feature_str}")

    except Exception as e:
        logger.warning(f"Could not validate system setup: {str(e)}")

    uvicorn.run(
        "api.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info"
    )
