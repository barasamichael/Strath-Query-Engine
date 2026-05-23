import os
import json
import tqdm
import logging
from typing import Any
from typing import Dict
from typing import List
from pathlib import Path
from typing import Optional
from datetime import datetime

import typer
from rich.table import Table
from rich.panel import Panel
from rich.console import Console
from rich.logging import RichHandler

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

# Setup rich console for pretty output
console = Console()

# Setup logging with rich handler
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

log = logging.getLogger("strathmore-rag-cli")

# Create Typer app
app = typer.Typer(
    help="Enhanced Strathmore University RAG System CLI with Optimized Service Management"
)


class ServiceContainer:
    """Centralized service container with proper dependency injection."""

    def __init__(self):
        self._services = {}
        self._initialized = False

    def initialize(self):
        """Initialize all services with proper dependency management."""
        if self._initialized:
            return

        try:
            console.print("[dim]Initializing service container...[/dim]")

            # Core services first (no dependencies)
            self._services["embedding"] = EmbeddingService()

            # Tavily service (independent)
            if os.getenv("TAVILY_API_KEY"):
                self._services["tavily"] = TavilyService()
                log.info("Tavily service initialized")
            else:
                self._services["tavily"] = None
                log.warning("Tavily service disabled - no API key")

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
            console.print(
                "[dim green]✓ Service container initialized successfully[/dim green]"
            )

        except Exception as e:
            log.error(f"Service container initialization failed: {e}")
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
                    # Basic health checks
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

    def get_all_services(self) -> Dict[str, Any]:
        """Get all services for compatibility."""
        if not self._initialized:
            self.initialize()
        return self._services.copy()


# Initialize service container
container = ServiceContainer()

# Conversation memory for interactive sessions
conversation_memory: List[Dict[str, Any]] = []


def get_service(name: str):
    """Helper to get service from container."""
    return container.get(name)


@app.command()
def system_status():
    """Display comprehensive system status and configuration."""
    console.print(
        "[bold blue]Enhanced Strathmore RAG System Status[/bold blue]"
    )

    # Initialize services if needed
    container.initialize()

    # Validate setup
    document_processor = get_service("document_processor")
    validation = validate_document_processor_setup(document_processor)

    # System health
    health_table = Table(title="System Health")
    health_table.add_column("Component", style="cyan")
    health_table.add_column("Status", style="green")
    health_table.add_column("Details", style="dim")

    # Check API keys
    openai_status = (
        "✓ Available" if os.getenv("OPENAI_API_KEY") else "✗ Missing"
    )
    health_table.add_row(
        "OpenAI API Key",
        openai_status,
        "Required for embeddings and responses",
    )

    tavily_status = (
        "✓ Available" if os.getenv("TAVILY_API_KEY") else "✗ Missing"
    )
    health_table.add_row(
        "Tavily API Key",
        tavily_status,
        "Required for real-time information",
    )

    # Check document processor
    dp_status = "✓ Valid" if validation["valid"] else "✗ Issues"
    health_table.add_row(
        "Document Processor",
        dp_status,
        f"{len(validation['errors'])} errors, {len(validation['warnings'])} warnings",
    )

    # Check vector database
    vector_db_service = get_service("vector_db")
    try:
        vdb_stats = vector_db_service.get_collection_stats()
        vdb_status = f"✓ Ready ({vdb_stats['count']} vectors)"
    except Exception as e:
        vdb_status = f"✗ Error: {str(e)[:30]}..."
    health_table.add_row(
        "Vector Database",
        vdb_status,
        f"Collection: {settings.vector_db.collection_name}",
    )

    # Check processed documents
    documents = document_processor.list_documents()
    doc_status = f"✓ {len(documents)} documents"
    health_table.add_row(
        "Processed Documents",
        doc_status,
        f"In {document_processor.processed_dir}",
    )

    # Check schedule system
    schedule_extractor = get_service("schedule_extractor")
    structured_storage = get_service("structured_storage")
    try:
        if schedule_extractor and structured_storage:
            struct_stats = structured_storage.get_statistics()
            schedule_status = (
                f"✓ Active ({struct_stats['total_entries']} entries)"
            )
            schedule_details = (
                f"{struct_stats['unique_class_groups']} class groups"
            )
        else:
            schedule_status = "✗ Not Available"
            schedule_details = "Services not initialized"
    except Exception as e:
        schedule_status = f"✗ Error: {str(e)[:30]}..."
        schedule_details = "Failed to check schedule system"

    health_table.add_row(
        "Schedule Detection", schedule_status, schedule_details
    )

    # Check structured storage
    try:
        if structured_storage:
            struct_stats = structured_storage.get_statistics()
            struct_status = f"✓ Ready ({struct_stats['total_entries']} entries)"
            struct_details = (
                f"{struct_stats['unique_class_groups']} class groups"
            )
        else:
            struct_status = "✗ Not Available"
            struct_details = "Structured storage not initialized"
    except Exception as e:
        struct_status = f"✗ Error: {str(e)[:30]}..."
        struct_details = "Failed to check structured storage"

    health_table.add_row("Structured Storage", struct_status, struct_details)

    # Check real-time services
    tavily_service = get_service("tavily")
    realtime_status = "✓ Available" if tavily_service else "✗ Disabled"
    realtime_details = (
        "Domain-filtered real-time search"
        if tavily_service
        else "Tavily API key missing"
    )
    health_table.add_row(
        "Real-time Information", realtime_status, realtime_details
    )

    # Check intent recognition
    embedding_service = get_service("embedding")
    try:
        intent_status = (
            "✓ Available"
            if hasattr(embedding_service, "intent_embeddings")
            and embedding_service.intent_embeddings
            else "✗ Not Initialized"
        )
        intent_details = (
            "Embedding-based intent classification"
            if intent_status == "✓ Available"
            else "Intent templates not loaded"
        )
    except Exception:
        intent_status = "✗ Error"
        intent_details = "Failed to check intent recognition"

    health_table.add_row("Intent Recognition", intent_status, intent_details)

    # Check query router
    query_router = get_service("query_router")
    if query_router:
        router_status = "✓ Available"
        router_details = "Smart routing enabled"
    else:
        router_status = "✗ Not Available"
        router_details = "Router not initialized"

    health_table.add_row("Query Router", router_status, router_details)

    console.print(health_table)

    # Configuration details
    config_table = Table(title="Configuration")
    config_table.add_column("Setting", style="cyan")
    config_table.add_column("Value", style="green")

    response_generator = get_service("response_generator")
    config_table.add_row("Environment", settings.environment)
    config_table.add_row(
        "LLM Model",
        (
            response_generator.model
            if hasattr(response_generator, "model")
            else "gpt-4o"
        ),
    )
    config_table.add_row("Embedding Model", embedding_service.model_name)
    config_table.add_row(
        "Chunk Size", str(getattr(document_processor, "chunk_size", 400))
    )
    config_table.add_row(
        "Deduplication Enabled", str(settings.deduplication.enabled)
    )
    config_table.add_row(
        "Similarity Threshold", str(settings.deduplication.similarity_threshold)
    )
    config_table.add_row(
        "Real-time Integration", "✓ Enabled" if tavily_service else "✗ Disabled"
    )
    config_table.add_row(
        "Intent Recognition",
        (
            "✓ Enabled"
            if hasattr(embedding_service, "intent_embeddings")
            else "✗ Disabled"
        ),
    )

    console.print(config_table)

    # Show validation issues if any
    if validation["errors"]:
        console.print("[bold red]Errors:[/bold red]")
        for error in validation["errors"]:
            console.print(f"  • {error}")

    if validation["warnings"]:
        console.print("[bold yellow]Warnings:[/bold yellow]")
        for warning in validation["warnings"]:
            console.print(f"  • {warning}")


@app.command()
def process_document(
    file_path: str = typer.Argument(..., help="Path to the document file")
):
    """Process a single document with enhanced semantic chunking and schedule detection."""
    console.print(f"Processing document: [bold blue]{file_path}[/bold blue]")

    try:
        # Check if file exists
        if not Path(file_path).exists():
            console.print(f"[bold red]File not found:[/bold red] {file_path}")
            return

        document_processor = get_service("document_processor")
        schedule_extractor = get_service("schedule_extractor")
        structured_storage = get_service("structured_storage")
        embedding_service = get_service("embedding")
        vector_db_service = get_service("vector_db")

        # Check if it's a schedule document
        is_schedule, schedule_confidence = (
            schedule_extractor.detect_schedule_document(file_path)
        )

        metadata = document_processor.process_document(file_path)

        if not metadata:
            console.print("[bold red]Failed to process document[/bold red]")
            return

        console.print(
            f"Document processed: [bold green]{metadata['doc_id']}[/bold green]"
        )
        console.print(
            f"Created [bold green]{metadata['num_chunks']}[/bold green] chunks"
        )

        # Handle schedule documents
        if is_schedule:
            console.print(
                "[bold yellow]📋 Schedule Document Detected![/bold yellow]"
            )
            console.print(f"  📈 Confidence: {schedule_confidence:.2f}")

            # Extract schedule data
            schedule_entries = schedule_extractor.extract_from_file(file_path)

            if schedule_entries:
                # Store in structured database
                schedule_dicts = [entry.to_dict() for entry in schedule_entries]
                stored_count = structured_storage.store_schedules(
                    schedule_dicts,
                    metadata["doc_id"],
                    metadata["file_name"],
                    str(file_path),
                )

                console.print(
                    f"  📊 {len(schedule_entries)} schedule entries extracted"
                )
                console.print(f"  💾 {stored_count} entries stored in database")

                # Create semantic chunks for hybrid search
                semantic_chunks = schedule_extractor.convert_to_semantic_chunks(
                    schedule_entries
                )
                console.print(
                    f"  🔍 {len(semantic_chunks)} semantic chunks created"
                )

        # Generate embeddings
        console.print("Generating embeddings...")
        embedding_service.embed_chunks(metadata["chunks_path"])

        # Index chunks
        console.print("Indexing chunks in vector database...")
        vector_db_service.index_chunks(metadata["chunks_path"])

        console.print(
            "[bold green]Document successfully processed and indexed![/bold green]"
        )

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")


@app.command()
def process_schedule_file(
    file_path: str = typer.Argument(
        ..., help="Path to the schedule file (Excel or PDF)"
    ),
    force_schedule: bool = typer.Option(
        False, "--force", "-f", help="Force processing as schedule"
    ),
):
    """Process a specific file as a schedule document."""
    console.print(
        f"Processing schedule file: [bold blue]{file_path}[/bold blue]"
    )

    try:
        file_path = Path(file_path)

        if not file_path.exists():
            console.print(f"[bold red]File not found:[/bold red] {file_path}")
            return

        document_processor = get_service("document_processor")
        schedule_extractor = get_service("schedule_extractor")
        structured_storage = get_service("structured_storage")
        embedding_service = get_service("embedding")
        vector_db_service = get_service("vector_db")

        if force_schedule:
            console.print("[yellow]Forcing schedule processing[/yellow]")
            original_detect = schedule_extractor.detect_schedule_document
            schedule_extractor.detect_schedule_document = lambda x, y=None: (
                True,
                0.9,
            )

        # Extract schedule entries directly
        schedule_entries = schedule_extractor.extract_from_file(file_path)

        if force_schedule:
            schedule_extractor.detect_schedule_document = original_detect

        if schedule_entries:
            # Generate doc_id
            import hashlib

            doc_id = hashlib.md5(str(file_path).encode()).hexdigest()

            # Store in structured database
            schedule_dicts = [entry.to_dict() for entry in schedule_entries]
            stored_count = structured_storage.store_schedules(
                schedule_dicts, doc_id, file_path.name, str(file_path)
            )

            # Create semantic chunks
            semantic_chunks = schedule_extractor.convert_to_semantic_chunks(
                schedule_entries
            )

            console.print(
                f"[bold green]Successfully processed:[/bold green] {doc_id}"
            )
            console.print(
                f"✓ {len(schedule_entries)} schedule entries extracted"
            )
            console.print(f"✓ {stored_count} structured entries stored")
            console.print(f"✓ {len(semantic_chunks)} semantic chunks created")

            # Also process as regular document for vector search
            metadata = document_processor.process_document(file_path)
            if metadata:
                embedding_service.embed_chunks(metadata["chunks_path"])
                vector_db_service.index_chunks(metadata["chunks_path"])

            console.print("[bold green]Processing complete![/bold green]")
        else:
            console.print("[bold red]No schedule data found[/bold red]")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")


@app.command()
def schedule_stats():
    """Display comprehensive schedule processing statistics."""
    console.print("[bold blue]Schedule Processing Statistics[/bold blue]")

    try:
        structured_storage = get_service("structured_storage")
        query_router = get_service("query_router")

        if structured_storage:
            stats = structured_storage.get_statistics()

            # Create statistics table
            stats_table = Table(title="Schedule System Overview")
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", style="green")

            stats_table.add_row(
                "Total Documents", str(stats.get("documents_processed", 0))
            )
            stats_table.add_row(
                "Total Schedule Entries", str(stats["total_entries"])
            )
            stats_table.add_row(
                "Unique Class Groups", str(stats["unique_class_groups"])
            )
            stats_table.add_row(
                "Unique Subjects", str(stats["unique_subjects"])
            )
            stats_table.add_row("Unique Rooms", str(stats["unique_rooms"]))
            stats_table.add_row(
                "Unique Instructors", str(stats["unique_instructors"])
            )

            console.print(stats_table)

            # Show router statistics if available
            if query_router:
                router_stats = query_router.get_routing_statistics()
                console.print(
                    "\n[bold green]Smart Routing Enabled[/bold green]"
                )
                console.print(
                    f"Structured patterns: {router_stats['total_structured_patterns']}"
                )
                console.print(
                    f"Hybrid queries: {'✓' if router_stats['supports_hybrid_queries'] else '✗'}"
                )
        else:
            console.print(
                "[bold yellow]Schedule system not available[/bold yellow]"
            )

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")


@app.command()
def query_schedule(
    query_text: str = typer.Argument(
        ..., help="Natural language schedule query"
    ),
    show_sql: bool = typer.Option(
        False, "--show-sql", "-s", help="Show generated SQL query"
    ),
    show_routing: bool = typer.Option(
        False, "--show-routing", "-r", help="Show routing analysis"
    ),
):
    """Query schedule data using natural language."""
    console.print(
        f"Processing schedule query: [bold blue]{query_text}[/bold blue]"
    )

    try:
        query_router = get_service("query_router")
        response_generator = get_service("response_generator")

        if query_router:
            # Analyze query routing
            analysis = query_router.analyze_query(query_text)

            if show_routing:
                console.print("\n[bold yellow]Query Analysis:[/bold yellow]")
                console.print(f"Type: {analysis.query_type.value}")
                console.print(f"Confidence: {analysis.confidence:.2f}")
                console.print(f"Explanation: {analysis.explanation}")

            # Route query
            route_result = query_router.route_query(query_text)

            if show_sql and route_result.get("sql_query"):
                console.print("\n[bold yellow]SQL Query:[/bold yellow]")
                console.print(route_result["sql_query"])

            # Display response
            console.print("\n[bold green]Response:[/bold green]")
            console.print(route_result.get("answer", "No response generated"))

            # Show results count
            result_count = route_result.get("result_count", 0)
            approach = route_result.get("routing_info", {}).get(
                "primary_approach", "unknown"
            )
            console.print(
                f"\n[dim]Found {result_count} results using {approach} approach[/dim]"
            )

        else:
            # Fallback to response generator
            response_data = response_generator.generate_response(query_text)
            console.print("\n[bold green]Response:[/bold green]")
            console.print(
                response_data.get("response", "No response generated")
            )

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")


@app.command()
def validate_schedule_system():
    """Validate the schedule detection and processing system."""
    console.print("[bold blue]Validating Schedule System[/bold blue]")

    try:
        schedule_extractor = get_service("schedule_extractor")
        structured_storage = get_service("structured_storage")
        query_router = get_service("query_router")

        validation = {
            "valid": True,
            "features": [],
            "errors": [],
            "warnings": [],
        }

        if schedule_extractor:
            validation["features"].append("Schedule extraction")
        else:
            validation["errors"].append("Schedule extractor not available")
            validation["valid"] = False

        if structured_storage:
            validation["features"].append("Structured storage")
            try:
                stats = structured_storage.get_statistics()
                validation["features"].append(
                    f"Database with {stats['total_entries']} entries"
                )
            except Exception as e:
                validation["warnings"].append(
                    f"Database access issue: {str(e)}"
                )
        else:
            validation["errors"].append("Structured storage not available")
            validation["valid"] = False

        if query_router:
            validation["features"].append("Smart query routing")
        else:
            validation["warnings"].append("Query router not available")

        # Display validation results
        if validation["valid"]:
            console.print(
                "[bold green]✓ Schedule system is properly configured[/bold green]"
            )
        else:
            console.print("[bold red]✗ Schedule system has issues[/bold red]")

        # Show features
        if validation["features"]:
            console.print("\n[bold blue]Available Features:[/bold blue]")
            for feature in validation["features"]:
                console.print(f"  ✓ {feature}")

        # Show errors
        if validation["errors"]:
            console.print("\n[bold red]Errors:[/bold red]")
            for error in validation["errors"]:
                console.print(f"  ✗ {error}")

        # Show warnings
        if validation["warnings"]:
            console.print("\n[bold yellow]Warnings:[/bold yellow]")
            for warning in validation["warnings"]:
                console.print(f"  ⚠ {warning}")

        return validation["valid"]

    except Exception as e:
        console.print(f"[bold red]Validation failed:[/bold red] {str(e)}")
        return False


@app.command()
def process_folder(
    folder_path: str = typer.Argument(
        ..., help="Path to the folder containing documents"
    ),
    recursive: bool = typer.Option(
        False, "--recursive", "-r", help="Process subfolders recursively"
    ),
    rebuild_index: bool = typer.Option(
        False, "--rebuild-index", help="Rebuild vector index after processing"
    ),
):
    """Process multiple documents from a folder with enhanced detection."""
    console.print(f"Processing folder: [bold blue]{folder_path}[/bold blue]")

    try:
        folder_path_obj = Path(folder_path)

        if not folder_path_obj.exists():
            console.print(
                f"[bold red]Folder not found:[/bold red] {folder_path}"
            )
            return

        if not folder_path_obj.is_dir():
            console.print(
                f"[bold red]Path is not a directory:[/bold red] {folder_path}"
            )
            return

        document_processor = get_service("document_processor")
        embedding_service = get_service("embedding")
        vector_db_service = get_service("vector_db")

        # Process documents
        if recursive:
            all_files = list(folder_path_obj.rglob("*"))
            supported_files = [
                f
                for f in all_files
                if f.is_file() and document_processor.is_file_supported(f)
            ]

            results = []
            for file_path in supported_files:
                try:
                    result = document_processor.process_document(file_path)
                    if result:
                        results.append(result)
                except Exception as e:
                    console.print(
                        f"[bold red]Failed to process {file_path.name}: {str(e)}[/bold red]"
                    )
        else:
            results = document_processor.process_folder(folder_path_obj)

        if not results:
            console.print(
                "[bold yellow]No documents found or processed[/bold yellow]"
            )
            return

        console.print(
            f"Processed [bold green]{len(results)}[/bold green] documents"
        )

        # Generate embeddings for all chunks
        console.print("Generating embeddings for all processed documents...")
        for result in results:
            try:
                embedding_service.embed_chunks(result["chunks_path"])
            except Exception as e:
                console.print(
                    f"[bold yellow]Warning: Failed to embed {result['file_name']}: {str(e)}[/bold yellow]"
                )

        # Rebuild index if requested
        if rebuild_index:
            console.print("Rebuilding vector database index...")
            vector_db_service.initialize_collection(recreate=True)

        # Index all chunks
        console.print("Indexing all chunks in vector database...")
        for result in results:
            try:
                vector_db_service.index_chunks(result["chunks_path"])
            except Exception as e:
                console.print(
                    f"[bold yellow]Warning: Failed to index {result['file_name']}: {str(e)}[/bold yellow]"
                )

        console.print(
            "[bold green]All documents successfully processed and indexed![/bold green]"
        )

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")


@app.command()
def list_documents(
    show_details: bool = typer.Option(
        False, "--details", "-d", help="Show detailed information"
    ),
    doc_type: Optional[str] = typer.Option(
        None, "--type", "-t", help="Filter by document type"
    ),
):
    """List all processed documents with enhanced information."""
    try:
        document_processor = get_service("document_processor")

        if doc_type and hasattr(document_processor, "get_documents_by_type"):
            documents = document_processor.get_documents_by_type(doc_type)
            console.print(
                f"Showing documents of type: [bold blue]{doc_type}[/bold blue]"
            )
        else:
            documents = document_processor.list_documents()

        if not documents:
            console.print("[yellow]No documents found.[/yellow]")
            return

        # Create and display table
        table = Table(title=f"Documents ({len(documents)})")
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Filename", style="green")
        table.add_column("Type", style="blue")
        table.add_column("Chunks", style="magenta", justify="right")
        table.add_column("Size (MB)", style="yellow", justify="right")

        if show_details:
            table.add_column("Processed Path", style="dim")
            table.add_column("Chunks Path", style="dim")

        for doc in documents:
            size_mb = (
                doc.get("file_size", 0) / (1024 * 1024)
                if doc.get("file_size")
                else 0
            )

            row = [
                doc["doc_id"][:8] + "...",
                doc["file_name"],
                doc["doc_type"],
                str(doc["num_chunks"]),
                f"{size_mb:.2f}",
            ]

            if show_details:
                row.extend([doc["processed_path"], doc["chunks_path"]])

            table.add_row(*row)

        console.print(table)

        # Show processing statistics
        # Show processing statistics
        stats = document_processor.get_processing_stats()
        console.print(f"\nSupported formats: {stats['supported_formats']}")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")


@app.command()
def document_info(
    doc_id: str = typer.Argument(
        ..., help="Document ID to get information about"
    )
):
    """Get detailed information about a specific document."""
    try:
        document_processor = get_service("document_processor")
        doc_info = document_processor.get_document_info(doc_id)

        if not doc_info:
            console.print(
                f"[bold yellow]Document with ID {doc_id} not found[/bold yellow]"
            )
            return

        # Calculate file size in MB
        file_size = doc_info.get("file_size", 0)
        size_mb = file_size / (1024 * 1024) if file_size else 0

        # Display detailed information
        info_text = (
            f"[bold blue]Document ID:[/bold blue] {doc_info['doc_id']}\n"
            f"[bold blue]Filename:[/bold blue] {doc_info['file_name']}\n"
            f"[bold blue]Document Type:[/bold blue] {doc_info['doc_type']}\n"
            f"[bold blue]Number of Chunks:[/bold blue] {doc_info['num_chunks']}\n"
            f"[bold blue]File Size:[/bold blue] {size_mb:.2f} MB\n"
            f"[bold blue]Source Path:[/bold blue] {doc_info['file_path']}\n"
            f"[bold blue]Processed Path:[/bold blue] {doc_info['processed_path']}\n"
            f"[bold blue]Chunks Path:[/bold blue] {doc_info['chunks_path']}\n"
        )

        # Add chunk settings if available
        if "chunk_settings" in doc_info:
            settings_info = doc_info["chunk_settings"]
            info_text += (
                f"[bold blue]Chunk Size:[/bold blue] {settings_info.get('chunk_size', 'N/A')}\n"
                f"[bold blue]Chunk Overlap:[/bold blue] {settings_info.get('chunk_overlap', 'N/A')}\n"
            )

        console.print(
            Panel.fit(
                info_text,
                title=f"Document Information: {doc_info['file_name']}",
                border_style="green",
            )
        )

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")


@app.command()
def initialize_collection(
    recreate: bool = typer.Option(
        False, "--recreate", "-r", help="Recreate collection if it exists"
    )
):
    """Initialize or recreate the vector database collection."""
    action = "Recreating" if recreate else "Initializing"
    console.print(
        f"{action} vector database collection: [bold blue]{settings.vector_db.collection_name}[/bold blue]"
    )

    try:
        vector_db_service = get_service("vector_db")
        vector_db_service.initialize_collection(recreate=recreate)
        console.print(
            "[bold green]Collection initialized successfully![/bold green]"
        )

        # Show statistics
        stats = vector_db_service.get_collection_stats()
        console.print(f"Collection now contains: {stats['count']} vectors")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")


@app.command()
def rebuild_index():
    """Rebuild the vector database index with all processed documents."""
    console.print("[bold blue]Rebuilding vector database index[/bold blue]")

    try:
        if typer.confirm(
            "Do you want to recreate the collection?", default=True
        ):
            vector_db_service = get_service("vector_db")
            vector_db_service.initialize_collection(recreate=True)
            console.print(
                "[bold green]Collection recreated successfully![/bold green]"
            )

        document_processor = get_service("document_processor")
        documents = document_processor.list_documents()

        if not documents:
            console.print("[yellow]No documents found to index.[/yellow]")
            return

        console.print(
            f"Found [bold green]{len(documents)}[/bold green] documents to index"
        )

        # Index each document's chunks
        for doc in documents:
            console.print(f"Indexing {doc['file_name']}...")
            try:
                vector_db_service.index_chunks(doc["chunks_path"])
            except Exception as e:
                console.print(
                    f"[bold yellow]Warning: Failed to index {doc['file_name']}: {str(e)}[/bold yellow]"
                )

        console.print(
            "[bold green]Vector database index rebuilt successfully![/bold green]"
        )

        # Show final statistics
        stats = vector_db_service.get_collection_stats()
        console.print(f"Final collection size: {stats['count']} vectors")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")


@app.command()
def query(
    query_text: str = typer.Argument(..., help="Query text to search for"),
    top_k: int = typer.Option(
        15, "--top-k", "-k", help="Number of chunks to retrieve"
    ),
    use_multi_query: bool = typer.Option(
        True, "--multi-query", "-m", help="Use multi-query approach"
    ),
    show_chunks: bool = typer.Option(
        False, "--show-chunks", "-c", help="Show retrieved chunks"
    ),
    use_memory: bool = typer.Option(
        False, "--use-memory", help="Use conversation memory"
    ),
    use_real_time: bool = typer.Option(
        True, "--real-time", help="Include real-time information"
    ),
):
    """Test a query against the enhanced RAG system with all features."""
    console.print(f"Processing query: [bold blue]{query_text}[/bold blue]")

    try:
        response_generator = get_service("response_generator")
        memory_processor = get_service("memory_processor")

        # Process conversation context if using memory
        context_info = {}
        if use_memory and conversation_memory:
            context_info = memory_processor.process_conversation_context(
                query_text, conversation_memory
            )

            if context_info.get("needs_clarification"):
                console.print(
                    f"[bold yellow]Clarification needed:[/bold yellow] {context_info.get('clarification_prompt', '')}"
                )
                return

        # Use enhanced response generator with all features
        response_data = response_generator.generate_response(
            query=query_text,
            context_info=context_info,
            use_real_time=use_real_time,
        )

        # Display enhanced information
        info_table = Table(title="Query Processing Information")
        info_table.add_column("Aspect", style="cyan")
        info_table.add_column("Value", style="green")

        info_table.add_row(
            "Intent Type", response_data.get("intent_type", "unknown")
        )
        info_table.add_row(
            "Intent Confidence",
            f"{response_data.get('intent_confidence', 0.0):.2f}",
        )
        info_table.add_row(
            "Context Sources", str(response_data.get("context_sources", 0))
        )
        info_table.add_row(
            "Real-time Info",
            "Yes" if response_data.get("real_time_used") else "No",
        )
        info_table.add_row(
            "Processing Approach",
            response_data.get("approach", "standard"),
        )

        console.print(info_table)

        # Display response
        console.print("\n[bold green]Response:[/bold green]")
        console.print(response_data.get("response", "No response generated"))

        # Show chunks if requested and available
        if show_chunks:
            search_results = response_data.get("search_results", [])
            if search_results:
                console.print(
                    f"\n[bold yellow]Retrieved Chunks ({len(search_results)}):[/bold yellow]"
                )
                for i, result in enumerate(search_results[:5], 1):
                    chunk_text = result.get("text", "")[:200] + "..."
                    score = result.get("score", 0.0)
                    search_type = result.get("search_type", "semantic")
                    console.print(
                        f"\n{i}. [dim]Score: {score:.3f} | Type: {search_type}[/dim]"
                    )
                    console.print(f"   {chunk_text}")

        # Update conversation memory if using memory
        if use_memory:
            conversation_memory.append(
                {
                    "query": query_text,
                    "response": response_data.get("response", ""),
                    "timestamp": datetime.now().isoformat(),
                    "intent_type": response_data.get("intent_type", "unknown"),
                }
            )

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")


@app.command()
def interactive():
    """Start an interactive enhanced RAG session with full memory support."""
    console.print("[bold blue]Enhanced Interactive RAG Session[/bold blue]")
    console.print(
        "Type 'quit' to exit, 'clear' to clear memory, 'stats' for session stats, 'help' for commands"
    )
    console.print()

    session_memory = []
    response_generator = get_service("response_generator")
    memory_processor = get_service("memory_processor")

    while True:
        try:
            query = typer.prompt("\nYour question")

            if query.lower() in ["quit", "exit", "q"]:
                console.print("[bold blue]Session ended. Goodbye![/bold blue]")
                break
            elif query.lower() == "clear":
                session_memory.clear()
                console.print("[yellow]Memory cleared[/yellow]")
                continue
            elif query.lower() == "help":
                console.print("[bold cyan]Available commands:[/bold cyan]")
                console.print("  quit/exit/q - End session")
                console.print("  clear - Clear conversation memory")
                console.print("  stats - Show session statistics")
                console.print("  help - Show this help message")
                continue
            elif query.lower() == "stats":
                console.print(
                    f"[yellow]Session stats: {len(session_memory)} exchanges[/yellow]"
                )

                structured_storage = get_service("structured_storage")
                if structured_storage:
                    try:
                        stats = structured_storage.get_statistics()
                        console.print(
                            f"Schedule entries available: {stats.get('total_entries', 0)}"
                        )
                    except Exception:
                        pass

                # Show system stats
                embedding_service = get_service("embedding")
                try:
                    embedding_stats = embedding_service.get_embedding_stats()
                    console.print(
                        f"Total embeddings: {embedding_stats.get('total_embeddings', 0)}"
                    )
                    console.print(
                        f"Intent recognition: {'enabled' if embedding_stats.get('intent_recognition_enabled') else 'disabled'}"
                    )
                except Exception:
                    pass
                continue

            # Process conversation context
            context_info = {}
            if session_memory:
                context_info = memory_processor.process_conversation_context(
                    query, session_memory
                )

            # Use enhanced response generator
            response_data = response_generator.generate_response(
                query=query,
                context_info=context_info,
                use_real_time=True,
            )

            # Display enhanced approach information
            approach = response_data.get("approach", "standard")
            context_sources = response_data.get("context_sources", 0)
            intent_type = response_data.get("intent_type", "unknown")

            console.print(
                f"[dim]({approach} | {intent_type} | {context_sources} sources)[/dim]"
            )

            # Display response
            console.print(
                f"[bold green]Assistant:[/bold green] {response_data.get('response', 'No response generated')}"
            )

            # Update memory
            session_memory.append(
                {
                    "content": query,
                    "isUserMessage": True,
                    "timestamp": datetime.now().isoformat(),
                }
            )
            session_memory.append(
                {
                    "content": response_data.get("response", ""),
                    "isUserMessage": False,
                    "timestamp": datetime.now().isoformat(),
                    "intent_type": intent_type,
                }
            )

        except KeyboardInterrupt:
            console.print("\n[bold blue]Session ended. Goodbye![/bold blue]")
            break
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")


@app.command()
def evaluate(
    questions_file: str = typer.Option(
        "tests/eval_data/eval_questions.csv",
        help="Path to evaluation questions file",
    ),
    output_file: str = typer.Option(
        None, help="Output file for results (optional)"
    ),
):
    """Evaluate the enhanced RAG system performance with comprehensive metrics."""
    console.print("[bold blue]Evaluating Enhanced RAG System[/bold blue]")

    try:
        evaluator = get_service("evaluator")
        results = evaluator.run_evaluation(
            Path(questions_file), Path(output_file) if output_file else None
        )

        if not results or results.get("status") != "success":
            console.print(
                "[bold red]Evaluation failed or no results[/bold red]"
            )
            return

        # Display results
        console.print("\n[bold green]Evaluation Results:[/bold green]")
        console.print(f"Questions evaluated: {results['num_questions']}")
        console.print(f"Average score: {results['avg_score']:.2f}")
        console.print(f"Intent accuracy: {results['intent_accuracy']:.2f}")
        console.print(f"Topic accuracy: {results['topic_accuracy']:.2f}")
        console.print(f"Content accuracy: {results['content_accuracy']:.2f}")

        # Save results if output file specified
        if output_file:
            console.print(f"Results saved to: {output_file}")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")


@app.command()
def test_real_time(
    query: str = typer.Argument(
        ..., help="Query to test with real-time information"
    ),
    max_results: int = typer.Option(2, help="Maximum results from Tavily"),
):
    """Test real-time information retrieval using Tavily."""
    console.print(
        f"Testing real-time search for: [bold blue]{query}[/bold blue]"
    )

    tavily_service = get_service("tavily")
    if not tavily_service:
        console.print(
            "[bold red]Tavily service not available - check API key[/bold red]"
        )
        return

    try:
        # Test direct Tavily search
        console.print("Fetching real-time information...")
        result = tavily_service.search(
            query=query, max_results=max_results, topic="general"
        )

        console.print(f"Found {len(result.get('results', []))} results")

        for i, res in enumerate(result.get("results", []), 1):
            console.print(f"\n[bold cyan]Result {i}:[/bold cyan]")
            console.print(f"Title: {res.get('title', 'No title')}")
            console.print(f"URL: {res.get('url', 'No URL')}")
            console.print(
                f"Content: {res.get('content', 'No content')[:200]}..."
            )
            console.print(f"Relevance: {res.get('relevance_score', 0.0):.2f}")

        # Test integration with response generator
        console.print(
            "\n[bold yellow]Testing integration with response generator...[/bold yellow]"
        )

        response_generator = get_service("response_generator")
        response_data = response_generator.generate_response(
            query=query, use_real_time=True
        )

        console.print("\nIntegrated response:")
        console.print(
            f"Real-time info included: {response_data.get('real_time_used', False)}"
        )
        console.print(
            f"Response: {response_data.get('response', 'No response')[:300]}..."
        )

    except Exception as e:
        console.print(
            f"[bold red]Error testing real-time features:[/bold red] {str(e)}"
        )


@app.command()
def backup_system(
    compress: bool = typer.Option(
        True, "--compress", "-c", help="Compress backup"
    ),
    include_embeddings: bool = typer.Option(
        False, "--embeddings", "-e", help="Include embeddings in backup"
    ),
    include_vector_db: bool = typer.Option(
        False, "--vector-db", "-v", help="Include vector database in backup"
    ),
):
    """Create a backup of the enhanced RAG system."""
    import shutil
    import tarfile

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"enhanced_rag_system_backup_{timestamp}"

    try:
        document_processor = get_service("document_processor")
        embedding_service = get_service("embedding")
        vector_db_service = get_service("vector_db")
        structured_storage = get_service("structured_storage")

        # Create backup directory
        backup_dir = Path("backups")
        backup_dir.mkdir(exist_ok=True)

        if compress:
            backup_file = backup_dir / f"{backup_name}.tar.gz"
        else:
            backup_path = backup_dir / backup_name
            backup_path.mkdir(exist_ok=True)

        temp_backup = backup_dir / f"temp_{backup_name}"
        temp_backup.mkdir(exist_ok=True)

        try:
            console.print(
                f"Creating backup: [bold blue]{backup_name}[/bold blue]"
            )

            # Backup processed documents
            console.print("Backing up processed documents...")
            if document_processor.processed_dir.exists():
                shutil.copytree(
                    document_processor.processed_dir,
                    temp_backup / "processed",
                    dirs_exist_ok=True,
                )

            # Backup chunks
            console.print("Backing up chunks...")
            if document_processor.chunk_dir.exists():
                shutil.copytree(
                    document_processor.chunk_dir,
                    temp_backup / "chunks",
                    dirs_exist_ok=True,
                )

            # Backup structured storage (schedules database)
            console.print("Backing up schedule database...")
            if structured_storage and hasattr(structured_storage, "db_path"):
                db_path = structured_storage.db_path
                if db_path.exists():
                    shutil.copy2(db_path, temp_backup / "schedules.db")

            # Backup embeddings if requested
            if include_embeddings:
                console.print("Backing up embeddings...")
                if embedding_service.embeddings_dir.exists():
                    shutil.copytree(
                        embedding_service.embeddings_dir,
                        temp_backup / "embeddings",
                        dirs_exist_ok=True,
                    )

            # Backup vector database if requested
            if include_vector_db:
                console.print("Backing up vector database...")
                if vector_db_service.db_path.exists():
                    shutil.copytree(
                        vector_db_service.db_path,
                        temp_backup / "vector_db",
                        dirs_exist_ok=True,
                    )

            # Create enhanced configuration snapshot
            console.print("Creating configuration snapshot...")
            config_data = {
                "backup_info": {
                    "timestamp": timestamp,
                    "include_embeddings": include_embeddings,
                    "include_vector_db": include_vector_db,
                },
                "system_config": {
                    "chunk_size": getattr(
                        document_processor, "chunk_size", 400
                    ),
                    "embedding_model": embedding_service.model_name,
                },
                "statistics": {
                    "total_documents": len(document_processor.list_documents()),
                    "vector_count": vector_db_service.get_collection_stats().get(
                        "count", 0
                    ),
                    "embedding_stats": embedding_service.get_embedding_stats(),
                },
            }

            with open(temp_backup / "backup_config.json", "w") as f:
                json.dump(config_data, f, indent=2)

            # Compress if requested
            if compress:
                console.print("Compressing backup...")
                with tarfile.open(backup_file, "w:gz") as tar:
                    tar.add(temp_backup, arcname=backup_name)

                # Remove temporary directory
                shutil.rmtree(temp_backup)

                backup_size = backup_file.stat().st_size / (1024 * 1024)
                console.print(
                    f"[bold green]Backup created:[/bold green] {backup_file}"
                )
                console.print(f"Backup size: {backup_size:.2f} MB")
            else:
                # Move temp backup to final location
                shutil.move(temp_backup, backup_path)

                # Calculate size
                total_size = sum(
                    f.stat().st_size
                    for f in backup_path.rglob("*")
                    if f.is_file()
                )
                backup_size = total_size / (1024 * 1024)
                console.print(
                    f"[bold green]Backup created:[/bold green] {backup_path}"
                )
                console.print(f"Backup size: {backup_size:.2f} MB")

        except Exception as e:
            # Cleanup on error
            if temp_backup.exists():
                shutil.rmtree(temp_backup)
            raise e

    except Exception as e:
        console.print(f"[bold red]Error creating backup:[/bold red] {str(e)}")


@app.command()
def process_url(
    url: str = typer.Argument(..., help="URL to process"),
    output_name: Optional[str] = typer.Option(
        None, "--name", "-n", help="Custom name for the processed document"
    ),
    verify_ssl: bool = typer.Option(
        False, "--verify-ssl", help="Verify SSL certificates"
    ),
    index_immediately: bool = typer.Option(
        True, "--index", "-i", help="Generate embeddings and index immediately"
    ),
):
    """Process content from a single URL."""
    console.print(f"Processing URL: [bold blue]{url}[/bold blue]")

    try:
        # Validate URL format
        from urllib.parse import urlparse

        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            console.print(f"[bold red]Invalid URL format:[/bold red] {url}")
            console.print("URL must include protocol (http:// or https://)")
            return

        document_processor = get_service("document_processor")
        schedule_extractor = get_service("schedule_extractor")
        structured_storage = get_service("structured_storage")
        embedding_service = get_service("embedding_service")
        vector_db_service = get_service("vector_db")

        try:
            # Process the URL
            metadata = document_processor.process_url(url, output_name)

            if not metadata:
                console.print(
                    "[bold red]Failed to process URL - no content extracted[/bold red]"
                )
                return

            console.print(
                f"Successfully processed URL: [bold green]{metadata['doc_id']}[/bold green]"
            )
            console.print(
                f"Document name: [green]{metadata['file_name']}[/green]"
            )
            console.print(
                f"Created [bold green]{metadata['num_chunks']}[/bold green] chunks"
            )

            # Check if content might be schedule-related
            if schedule_extractor:
                try:
                    # Read the processed content to check for schedule patterns
                    processed_path = Path(metadata["processed_path"])
                    if processed_path.exists():
                        with open(processed_path, "r", encoding="utf-8") as f:
                            content = f.read()

                        # Use a simple text-based detection since we don't have a file
                        is_schedule = any(
                            keyword in content.lower()
                            for keyword in [
                                "schedule",
                                "timetable",
                                "class",
                                "course",
                                "monday",
                                "tuesday",
                                "wednesday",
                                "thursday",
                                "friday",
                                "time",
                                "room",
                                "professor",
                            ]
                        )

                        if is_schedule:
                            console.print(
                                "[bold yellow]📋 Potential schedule content detected![/bold yellow]"
                            )

                            # Try to extract schedule data from the text content
                            schedule_entries = (
                                schedule_extractor.extract_from_text(content)
                            )
                            if schedule_entries:
                                schedule_dicts = [
                                    entry.to_dict()
                                    for entry in schedule_entries
                                ]
                                stored_count = (
                                    structured_storage.store_schedules(
                                        schedule_dicts,
                                        metadata["doc_id"],
                                        metadata["file_name"],
                                        url,
                                    )
                                )
                                console.print(
                                    f"  📊 {len(schedule_entries)} schedule entries extracted"
                                )
                                console.print(
                                    f"  💾 {stored_count} entries stored in database"
                                )

                except Exception as e:
                    console.print(
                        f"[bold yellow]Warning: Schedule extraction failed: {str(e)}[/bold yellow]"
                    )

            # Generate embeddings and index if requested
            if index_immediately:
                console.print("Generating embeddings...")
                try:
                    embedding_service.embed_chunks(metadata["chunks_path"])
                    console.print("Indexing in vector database...")
                    vector_db_service.index_chunks(metadata["chunks_path"])
                    console.print(
                        "[bold green]✓ Document indexed successfully![/bold green]"
                    )
                except Exception as e:
                    console.print(
                        f"[bold yellow]Warning: Indexing failed: {str(e)}[/bold yellow]"
                    )
                    console.print(
                        "Document processed but not indexed. You can index it later."
                    )

            # Display final summary
            console.print("\n[bold green]URL Processing Complete![/bold green]")
            console.print(f"  • Document ID: {metadata['doc_id']}")
            console.print(f"  • Source URL: {url}")
            console.print(f"  • Chunks created: {metadata['num_chunks']}")
            console.print(
                f"  • Indexed: {'Yes' if index_immediately else 'No'}"
            )

        finally:
            pass

    except Exception as e:
        console.print(f"[bold red]Error processing URL:[/bold red] {str(e)}")


@app.command()
def process_sitemap(
    sitemap_url: str = typer.Argument(..., help="Sitemap URL to process"),
    max_pages: int = typer.Option(
        50, "--max-pages", "-m", help="Maximum number of pages to process"
    ),
    verify_ssl: bool = typer.Option(
        False, "--verify-ssl", help="Verify SSL certificates"
    ),
    index_immediately: bool = typer.Option(
        True, "--index", "-i", help="Generate embeddings and index immediately"
    ),
    filter_pattern: Optional[str] = typer.Option(
        None, "--filter", "-f", help="Filter URLs by pattern (regex supported)"
    ),
):
    """Process multiple pages from a sitemap URL."""
    console.print(f"Processing sitemap: [bold blue]{sitemap_url}[/bold blue]")
    console.print(f"Maximum pages: {max_pages}")

    try:
        # Validate sitemap URL
        from urllib.parse import urlparse

        parsed = urlparse(sitemap_url)
        if not parsed.scheme or not parsed.netloc:
            console.print(
                f"[bold red]Invalid sitemap URL format:[/bold red] {sitemap_url}"
            )
            return

        document_processor = get_service("document_processor")
        schedule_extractor = get_service("schedule_extractor")
        structured_storage = get_service("structured_storage")
        embedding_service = get_service("embedding_service")
        vector_db_service = get_service("vector_db")

        try:
            # Process the sitemap
            results = document_processor.process_sitemap(sitemap_url, max_pages)

            if not results:
                console.print(
                    "[bold red]No pages processed from sitemap[/bold red]"
                )
                return

            console.print(
                f"Successfully processed [bold green]{len(results)}[/bold green] pages from sitemap"
            )

            # Filter results if pattern provided
            if filter_pattern:
                import re

                try:
                    pattern = re.compile(filter_pattern, re.IGNORECASE)
                    filtered_results = [
                        result
                        for result in results
                        if pattern.search(result["file_path"])
                        or pattern.search(result["file_name"])
                    ]
                    console.print(
                        f"Filtered to [yellow]{len(filtered_results)}[/yellow] pages matching pattern: {filter_pattern}"
                    )
                    results = filtered_results
                except re.error as e:
                    console.print(
                        f"[bold yellow]Warning: Invalid regex pattern: {str(e)}[/bold yellow]"
                    )

            # Process schedule detection for all pages
            schedule_pages = 0
            total_schedule_entries = 0

            if schedule_extractor and structured_storage:
                console.print("Analyzing pages for schedule content...")
                for result in tqdm(results, desc="Schedule detection"):
                    try:
                        processed_path = Path(result["processed_path"])
                        if processed_path.exists():
                            with open(
                                processed_path, "r", encoding="utf-8"
                            ) as f:
                                content = f.read()

                            # Check for schedule indicators
                            is_schedule = any(
                                keyword in content.lower()
                                for keyword in [
                                    "schedule",
                                    "timetable",
                                    "class",
                                    "course",
                                    "monday",
                                    "tuesday",
                                    "wednesday",
                                    "thursday",
                                    "friday",
                                    "time",
                                    "room",
                                ]
                            )

                            if is_schedule:
                                schedule_pages += 1
                                schedule_entries = (
                                    schedule_extractor.extract_from_text(
                                        content
                                    )
                                )
                                if schedule_entries:
                                    schedule_dicts = [
                                        entry.to_dict()
                                        for entry in schedule_entries
                                    ]
                                    stored_count = (
                                        structured_storage.store_schedules(
                                            schedule_dicts,
                                            result["doc_id"],
                                            result["file_name"],
                                            result["file_path"],
                                        )
                                    )
                                    total_schedule_entries += stored_count

                    except Exception:
                        continue

                if schedule_pages > 0:
                    console.print(
                        f"[bold yellow]📋 Found {schedule_pages} pages with schedule content[/bold yellow]"
                    )
                    console.print(
                        f"  📊 {total_schedule_entries} schedule entries extracted total"
                    )

            # Generate embeddings and index if requested
            if index_immediately:
                console.print("Processing embeddings and indexing...")
                successful_indexes = 0
                failed_indexes = 0

                for result in tqdm(results, desc="Indexing"):
                    try:
                        embedding_service.embed_chunks(result["chunks_path"])
                        vector_db_service.index_chunks(result["chunks_path"])
                        successful_indexes += 1
                    except Exception:
                        failed_indexes += 1

                console.print(
                    f"[bold green]✓ {successful_indexes} pages indexed successfully[/bold green]"
                )
                if failed_indexes > 0:
                    console.print(
                        f"[bold yellow]⚠ {failed_indexes} pages failed indexing[/bold yellow]"
                    )

            # Display comprehensive summary
            total_chunks = sum(result["num_chunks"] for result in results)

            console.print(
                "\n[bold green]Sitemap Processing Complete![/bold green]"
            )
            console.print(f"  • Pages processed: {len(results)}")
            console.print(f"  • Total chunks created: {total_chunks:,}")
            console.print(f"  • Schedule pages found: {schedule_pages}")
            console.print(
                f"  • Schedule entries extracted: {total_schedule_entries}"
            )
            console.print(
                f"  • Successfully indexed: {successful_indexes if index_immediately else 'Not indexed'}"
            )
            console.print(f"  • Source sitemap: {sitemap_url}")

        finally:
            pass

    except Exception as e:
        console.print(
            f"[bold red]Error processing sitemap:[/bold red] {str(e)}"
        )


@app.command()
def process_url_list(
    file_path: str = typer.Argument(
        ..., help="Path to text file containing URLs (one per line)"
    ),
    max_urls: int = typer.Option(
        100, "--max-urls", "-m", help="Maximum number of URLs to process"
    ),
    verify_ssl: bool = typer.Option(
        True, "--verify-ssl", help="Verify SSL certificates"
    ),
    index_immediately: bool = typer.Option(
        True, "--index", "-i", help="Generate embeddings and index immediately"
    ),
    skip_duplicates: bool = typer.Option(
        True,
        "--skip-duplicates",
        help="Skip URLs that have already been processed",
    ),
    max_workers: int = typer.Option(
        10, "--threads", "-t", help="Maximum number of concurrent threads"
    ),
    timeout_seconds: int = typer.Option(
        30, "--timeout", help="Timeout per URL in seconds"
    ),
    retry_failed: bool = typer.Option(
        True, "--retry", help="Retry failed URLs once"
    ),
):
    """Process multiple URLs from a text file with concurrent threading for high performance."""
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from urllib.parse import urlparse
    import hashlib
    from dataclasses import dataclass
    from typing import Optional

    console.print(
        f"Processing URL list from: [bold blue]{file_path}[/bold blue]"
    )
    console.print(
        f"Using [bold cyan]{max_workers}[/bold cyan] concurrent threads"
    )

    @dataclass
    class ProcessingResult:
        line_num: int
        url: str
        success: bool
        metadata: Optional[Dict[str, Any]] = None
        error: Optional[str] = None
        processing_time: float = 0.0
        thread_id: str = ""
        retry_attempt: int = 0

    # Thread-safe counters and collections
    lock = threading.Lock()
    processed_count = 0
    failed_count = 0
    schedule_pages_count = 0
    total_schedule_entries_count = 0

    # Thread-safe collections
    all_results = []
    successful_results = []
    failed_urls = []

    def update_counters(result: ProcessingResult):
        """Thread-safe counter updates."""
        nonlocal processed_count, failed_count

        with lock:
            all_results.append(result)
            if result.success:
                processed_count += 1
                successful_results.append(result.metadata)
            else:
                failed_count += 1
                failed_urls.append((result.line_num, result.url, result.error))

    def process_single_url(url_data: tuple) -> ProcessingResult:
        """Process a single URL in a thread-safe manner."""
        line_num, url = url_data
        thread_id = threading.current_thread().name
        start_time = time.time()

        try:
            # Get services (these should be thread-safe)
            document_processor = get_service("document_processor")
            schedule_extractor = get_service("schedule_extractor")
            structured_storage = get_service("structured_storage")
            embedding_service = get_service("embedding")
            vector_db_service = get_service("vector_db")

            # Generate meaningful name from URL
            parsed = urlparse(url)
            url_name = (
                f"{parsed.netloc}_{parsed.path.replace('/', '_').strip('_')}"
            )
            if not url_name or url_name == parsed.netloc + "_":
                url_name = f"url_{line_num}_{int(time.time())}"

            # Process the URL with timeout handling
            metadata = None
            try:
                metadata = document_processor.process_url(url, url_name)
            except Exception as e:
                if "timeout" in str(e).lower():
                    raise Exception(f"Timeout after {timeout_seconds}s")
                raise

            if not metadata:
                return ProcessingResult(
                    line_num=line_num,
                    url=url,
                    success=False,
                    error="No content extracted",
                    processing_time=time.time() - start_time,
                    thread_id=thread_id,
                )

            # Check for schedule content (thread-safe)
            local_schedule_entries = 0
            is_schedule_page = False
            if schedule_extractor and structured_storage:
                try:
                    processed_path = Path(metadata["processed_path"])
                    if processed_path.exists():
                        with open(processed_path, "r", encoding="utf-8") as f:
                            content = f.read()

                        is_schedule = any(
                            keyword in content.lower()
                            for keyword in [
                                "schedule",
                                "timetable",
                                "class",
                                "course",
                                "monday",
                                "tuesday",
                                "wednesday",
                                "thursday",
                                "friday",
                            ]
                        )

                        if is_schedule:
                            is_schedule_page = True

                            schedule_entries = (
                                schedule_extractor.extract_from_text(content)
                            )
                            if schedule_entries:
                                schedule_dicts = [
                                    entry.to_dict()
                                    for entry in schedule_entries
                                ]
                                # Thread-safe database operation
                                with lock:
                                    stored_count = (
                                        structured_storage.store_schedules(
                                            schedule_dicts,
                                            metadata["doc_id"],
                                            metadata["file_name"],
                                            url,
                                        )
                                    )
                                    local_schedule_entries = stored_count
                                    schedule_pages_count += 1
                                    total_schedule_entries_count += stored_count

                except Exception:
                    pass  # Don't fail the entire URL for schedule extraction issues

            # Add schedule info to metadata for tracking
            metadata["local_schedule_entries"] = local_schedule_entries
            metadata["is_schedule_page"] = is_schedule_page

            # Index immediately if requested (thread-safe)
            if index_immediately:
                try:
                    embedding_service.embed_chunks(metadata["chunks_path"])
                    vector_db_service.index_chunks(metadata["chunks_path"])
                except Exception:
                    pass  # Don't fail for indexing issues

            # Add schedule info to metadata for tracking
            metadata["local_schedule_entries"] = local_schedule_entries

            return ProcessingResult(
                line_num=line_num,
                url=url,
                success=True,
                metadata=metadata,
                processing_time=time.time() - start_time,
                thread_id=thread_id,
            )

        except Exception as e:
            return ProcessingResult(
                line_num=line_num,
                url=url,
                success=False,
                error=str(e)[:200],  # Limit error message length
                processing_time=time.time() - start_time,
                thread_id=thread_id,
            )

    try:
        # Validate input file
        url_file = Path(file_path)
        if not url_file.exists():
            console.print(f"[bold red]File not found:[/bold red] {file_path}")
            return

        if not url_file.is_file():
            console.print(
                f"[bold red]Path is not a file:[/bold red] {file_path}"
            )
            return

        # Read and validate URLs
        console.print("Reading and validating URLs...")
        urls = []
        invalid_urls = []
        duplicate_urls = set()

        with open(url_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        document_processor = get_service("document_processor")
        existing_urls = set()

        # Check for existing URLs if skip_duplicates is enabled
        if skip_duplicates:
            for doc_info in document_processor.list_documents():
                if doc_info.get("doc_type") in ["web", "web_sitemap"]:
                    existing_urls.add(doc_info.get("file_path", ""))

        for line_num, line in enumerate(lines, 1):
            url = line.strip()

            # Skip empty lines and comments
            if not url or url.startswith("#"):
                continue

            # Validate URL format
            try:
                parsed = urlparse(url)
                if not parsed.scheme or not parsed.netloc:
                    invalid_urls.append(
                        (line_num, url, "Missing protocol or domain")
                    )
                    continue

                if parsed.scheme not in ["http", "https"]:
                    invalid_urls.append(
                        (line_num, url, "Only HTTP/HTTPS supported")
                    )
                    continue

                # Check for duplicates in file
                url_hash = hashlib.md5(url.encode()).hexdigest()
                if url_hash in duplicate_urls:
                    console.print(
                        f"[dim yellow]Line {line_num}: Duplicate URL skipped[/dim yellow]"
                    )
                    continue
                duplicate_urls.add(url_hash)

                # Check if already processed
                if skip_duplicates and url in existing_urls:
                    console.print(
                        f"[dim yellow]Line {line_num}: Already processed, skipping[/dim yellow]"
                    )
                    continue

                urls.append((line_num, url))

            except Exception as e:
                invalid_urls.append((line_num, url, f"Parse error: {str(e)}"))

        # Limit URLs if specified
        if len(urls) > max_urls:
            console.print(f"[yellow]Limiting to first {max_urls} URLs[/yellow]")
            urls = urls[:max_urls]

        # Report validation results
        console.print("URL validation complete:")
        console.print(f"  • Valid URLs to process: [green]{len(urls)}[/green]")
        console.print(
            f"  • Invalid URLs skipped: [red]{len(invalid_urls)}[/red]"
        )
        console.print(f"  • Total lines processed: {len(lines)}")

        if invalid_urls and len(invalid_urls) <= 10:
            console.print("\n[bold red]Invalid URLs:[/bold red]")
            for line_num, url, reason in invalid_urls:
                console.print(f"  Line {line_num}: {url[:50]}... - {reason}")

        if not urls:
            console.print("[bold red]No valid URLs to process![/bold red]")
            return

        try:
            # Process URLs with ThreadPoolExecutor
            console.print(
                f"\nStarting concurrent processing of {len(urls)} URLs..."
            )
            console.print(
                f"Max workers: {max_workers} | Timeout: {timeout_seconds}s"
            )

            start_time = time.time()

            with ThreadPoolExecutor(
                max_workers=max_workers, thread_name_prefix="URLProcessor"
            ) as executor:
                # Submit all URLs for processing
                future_to_url = {
                    executor.submit(process_single_url, url_data): url_data
                    for url_data in urls
                }

                # Process completed futures with progress tracking
                with tqdm(
                    total=len(urls),
                    desc="Processing URLs",
                    bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                ) as pbar:

                    for future in as_completed(
                        future_to_url, timeout=timeout_seconds * len(urls)
                    ):
                        try:
                            result = future.result(timeout=timeout_seconds)
                            update_counters(result)

                            # Update progress bar with dynamic description
                            current_schedule_pages = sum(
                                1
                                for r in all_results
                                if r.success
                                and r.metadata
                                and r.metadata.get("is_schedule_page", False)
                            )
                            current_schedule_entries = sum(
                                r.metadata.get("local_schedule_entries", 0)
                                for r in all_results
                                if r.success and r.metadata
                            )

                            pbar.set_description(
                                f"Processing URLs (✓{processed_count} ✗{failed_count} 📋{current_schedule_pages})"
                            )
                            pbar.update(1)

                        except Exception as e:
                            url_data = future_to_url[future]
                            error_result = ProcessingResult(
                                line_num=url_data[0],
                                url=url_data[1],
                                success=False,
                                error=f"Future exception: {str(e)[:200]}",
                                processing_time=timeout_seconds,
                                thread_id="timeout",
                            )
                            update_counters(error_result)
                            pbar.update(1)

            processing_time = time.time() - start_time

            # Retry failed URLs if requested
            if retry_failed and failed_urls:
                retry_urls = [
                    (line_num, url)
                    for line_num, url, error in failed_urls
                    if "timeout" not in error.lower()
                ]

                if retry_urls:
                    console.print(
                        f"\n[yellow]Retrying {len(retry_urls)} failed URLs...[/yellow]"
                    )

                    # Clear failed_urls for retry
                    with lock:
                        failed_urls.clear()
                        failed_count = 0

                    with ThreadPoolExecutor(
                        max_workers=max_workers // 2,
                        thread_name_prefix="URLRetry",
                    ) as retry_executor:
                        retry_futures = {
                            retry_executor.submit(
                                process_single_url, url_data
                            ): url_data
                            for url_data in retry_urls
                        }

                        with tqdm(
                            total=len(retry_urls), desc="Retrying failed URLs"
                        ) as retry_pbar:
                            for future in as_completed(
                                retry_futures,
                                timeout=timeout_seconds * len(retry_urls),
                            ):
                                try:
                                    result = future.result(
                                        timeout=timeout_seconds
                                    )
                                    result.retry_attempt = 1
                                    update_counters(result)
                                    retry_pbar.update(1)
                                except Exception as e:
                                    url_data = retry_futures[future]
                                    error_result = ProcessingResult(
                                        line_num=url_data[0],
                                        url=url_data[1],
                                        success=False,
                                        error=f"Retry failed: {str(e)[:200]}",
                                        processing_time=timeout_seconds,
                                        thread_id="retry_timeout",
                                        retry_attempt=1,
                                    )
                                    update_counters(error_result)
                                    retry_pbar.update(1)

            # Calculate comprehensive statistics
            total_chunks = sum(
                result["num_chunks"] for result in successful_results
            )
            final_schedule_pages = sum(
                1
                for r in all_results
                if r.success
                and r.metadata
                and r.metadata.get("is_schedule_page", False)
            )
            final_schedule_entries = sum(
                r.metadata.get("local_schedule_entries", 0)
                for r in all_results
                if r.success and r.metadata
            )
            success_rate = (processed_count / len(urls) * 100) if urls else 0
            avg_processing_time = (
                sum(r.processing_time for r in all_results) / len(all_results)
                if all_results
                else 0
            )
            urls_per_second = (
                len(urls) / processing_time if processing_time > 0 else 0
            )

            # Display comprehensive results
            console.print(
                f"\n[bold green]Concurrent URL Processing Complete![/bold green]"
            )
            console.print(f"[bold cyan]Performance Metrics:[/bold cyan]")
            console.print(f"  • Total processing time: {processing_time:.2f}s")
            console.print(f"  • URLs per second: {urls_per_second:.2f}")
            console.print(
                f"  • Average time per URL: {avg_processing_time:.2f}s"
            )
            console.print(f"  • Threads used: {max_workers}")

            console.print(f"\n[bold cyan]Processing Results:[/bold cyan]")
            console.print(f"  • URLs attempted: {len(urls)}")
            console.print(
                f"  • Successfully processed: [green]{processed_count}[/green]"
            )
            console.print(f"  • Failed: [red]{failed_count}[/red]")
            console.print(f"  • Success rate: {success_rate:.1f}%")
            console.print(f"  • Total chunks created: {total_chunks:,}")
            console.print(f"  • Schedule pages found: {final_schedule_pages}")
            console.print(
                f"  • Schedule entries extracted: {final_schedule_entries}"
            )
            console.print(
                f"  • Indexed: {'Yes' if index_immediately else 'No'}"
            )

            # Thread performance analysis
            thread_stats = {}
            for result in all_results:
                thread_id = result.thread_id
                if thread_id not in thread_stats:
                    thread_stats[thread_id] = {
                        "count": 0,
                        "success": 0,
                        "total_time": 0.0,
                    }
                thread_stats[thread_id]["count"] += 1
                thread_stats[thread_id]["total_time"] += result.processing_time
                if result.success:
                    thread_stats[thread_id]["success"] += 1

            if len(thread_stats) > 1:
                console.print(f"\n[bold cyan]Thread Performance:[/bold cyan]")
                for thread_id, stats in sorted(thread_stats.items()):
                    success_rate = (
                        (stats["success"] / stats["count"] * 100)
                        if stats["count"] > 0
                        else 0
                    )
                    avg_time = (
                        stats["total_time"] / stats["count"]
                        if stats["count"] > 0
                        else 0
                    )
                    console.print(
                        f"  • {thread_id}: {stats['count']} URLs, {success_rate:.1f}% success, {avg_time:.2f}s avg"
                    )

            # Show failed URLs (limited)
            if failed_urls:
                console.print(
                    f"\n[bold red]Failed URLs ({len(failed_urls)}):[/bold red]"
                )
                for line_num, url, error in failed_urls[:10]:
                    console.print(f"  Line {line_num}: {url[:60]}... - {error}")
                if len(failed_urls) > 10:
                    console.print(
                        f"  ... and {len(failed_urls) - 10} more failures"
                    )

            # Save detailed processing report
            report_path = (
                Path("data") / f"concurrent_url_report_{int(time.time())}.json"
            )
            report_path.parent.mkdir(exist_ok=True)

            # Prepare thread-safe report data
            report_data = {
                "timestamp": datetime.now().isoformat(),
                "source_file": str(url_file),
                "settings": {
                    "max_urls": max_urls,
                    "max_workers": max_workers,
                    "timeout_seconds": timeout_seconds,
                    "verify_ssl": False,
                    "skip_duplicates": skip_duplicates,
                    "index_immediately": index_immediately,
                    "retry_failed": retry_failed,
                },
                "performance": {
                    "total_processing_time": processing_time,
                    "urls_per_second": urls_per_second,
                    "average_time_per_url": avg_processing_time,
                    "threads_used": max_workers,
                },
                "results": {
                    "total_urls_attempted": len(urls),
                    "successful": processed_count,
                    "failed": failed_count,
                    "success_rate_percent": success_rate,
                    "total_chunks": total_chunks,
                    "schedule_pages": final_schedule_pages,
                    "schedule_entries": final_schedule_entries,
                },
                "thread_performance": {
                    thread_id: {
                        "urls_processed": stats["count"],
                        "success_count": stats["success"],
                        "success_rate": (
                            (stats["success"] / stats["count"] * 100)
                            if stats["count"] > 0
                            else 0
                        ),
                        "total_time": stats["total_time"],
                        "average_time": (
                            stats["total_time"] / stats["count"]
                            if stats["count"] > 0
                            else 0
                        ),
                    }
                    for thread_id, stats in thread_stats.items()
                },
                "successful_documents": [
                    {
                        "doc_id": result["doc_id"],
                        "url": result["file_path"],
                        "chunks": result["num_chunks"],
                        "name": result["file_name"],
                        "schedule_entries": result.get(
                            "local_schedule_entries", 0
                        ),
                    }
                    for result in successful_results
                ],
                "failed_urls": [
                    {"line": line_num, "url": url, "error": error}
                    for line_num, url, error in failed_urls
                ],
            }

            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)

            console.print(
                f"\n[dim]Detailed report saved to: {report_path}[/dim]"
            )
            console.print(
                f"[dim]Peak memory efficiency with {max_workers} concurrent threads[/dim]"
            )

        finally:
            pass

    except Exception as e:
        console.print(
            f"[bold red]Error processing URL list:[/bold red] {str(e)}"
        )
        # Make sure we don't leave hanging threads
        import sys

        sys.exit(1)


if __name__ == "__main__":
    # Check for required API keys
    if not os.getenv("OPENAI_API_KEY"):
        console.print(
            "[bold yellow]Warning:[/bold yellow] OPENAI_API_KEY environment variable not set"
        )
        console.print("Set it by running: export OPENAI_API_KEY=your_key_here")
        console.print()

    if not os.getenv("TAVILY_API_KEY"):
        console.print(
            "[bold yellow]Warning:[/bold yellow] TAVILY_API_KEY environment variable not set"
        )
        console.print(
            "Real-time features will be disabled. Set it by running: export TAVILY_API_KEY=your_key_here"
        )
        console.print()

    # Validate system setup on startup
    try:
        container.initialize()
        document_processor = get_service("document_processor")
        validation = validate_document_processor_setup(document_processor)

        if validation["errors"]:
            console.print(
                "[bold red]System validation errors detected![/bold red]"
            )
            console.print("Run 'system_status' for details.")
            console.print()
        else:
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
            console.print(
                f"[bold green]✓ Enhanced RAG system ready with {feature_str}![/bold green]"
            )
            console.print()
    except Exception as e:
        console.print(
            f"[bold yellow]Could not validate system setup: {str(e)}[/bold yellow]"
        )
        console.print()

    app()
