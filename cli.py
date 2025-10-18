import os
import json
import typer
import logging
from rich.table import Table
from rich.panel import Panel
from rich.console import Console
from rich.logging import RichHandler

from services.vector_db import VectorDBService
from services.embeddings import EmbeddingService
from services.intent_recognizer import IntentRecognizer
from services.response_generator import ResponseGenerator
from services.document_processor import DocumentProcessor

from config.settings import settings

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
app = typer.Typer(help="Strathmore University RAG System CLI")

# Initialize services
document_processor = DocumentProcessor(
    enable_deduplication=settings.deduplication.enabled,
    similarity_threshold=settings.deduplication.similarity_threshold,
)
embedding_service = EmbeddingService()
vector_db_service = VectorDBService(embedding_service=embedding_service)
intent_recognizer = IntentRecognizer()
response_generator = ResponseGenerator()


@app.command()
def process_document(
    file_path: str = typer.Argument(..., help="Path to the document file")
):
    """Process a single document: chunk, embed, and index."""
    console.print(f"Processing document: [bold blue]{file_path}[/bold blue]")

    try:
        metadata = document_processor.process_file(file_path)
        if not metadata:
            console.print("[bold red]Failed to process document[/bold red]")
            return

        console.print(
            f"Document processed: [bold green]{metadata['doc_id']}[/bold green]"
        )
        console.print(
            f"Created [bold green]{metadata['num_chunks']}[/bold green] chunks"
        )

        # Embed chunks
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
def process_folder(
    folder_path: str = typer.Argument(
        ..., help="Path to the folder containing documents to process"
    )
):
    """Process all documents in a specified folder."""
    console.print(
        f"Processing all documents in: [bold blue]{folder_path}[/bold blue]"
    )

    try:
        documents_metadata = document_processor.process_folder(folder_path)

        if not documents_metadata:
            console.print(
                "[yellow]No documents found to process in the folder[/yellow]"
            )
            return

        console.print(
            f"Processed [bold green]{len(documents_metadata)}[/bold green] documents"
        )

        # Embed and index all processed chunks
        console.print("Generating embeddings for all new chunks...")
        for metadata in documents_metadata:
            embedding_service.embed_chunks(metadata["chunks_path"])
            vector_db_service.index_chunks(metadata["chunks_path"])

        console.print(
            "[bold green]All documents in the folder successfully processed and indexed![/bold green]"
        )

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")


@app.command()
def process_all_documents():
    """Process all documents in the raw directory."""
    console.print("Processing all documents in the raw directory...")

    try:
        documents_metadata = document_processor.process_all_documents()

        if not documents_metadata:
            console.print("[yellow]No documents found to process[/yellow]")
            return

        console.print(
            f"Processed [bold green]{len(documents_metadata)}[/bold green] documents"
        )

        # Embed all chunks (will use deduplicated chunks if available)
        console.print("Generating embeddings for all chunks...")
        embedding_service.embed_chunks()

        # Index all chunks (will use deduplicated chunks if available)
        console.print("Indexing all chunks in vector database...")
        vector_db_service.index_chunks()

        console.print(
            "[bold green]All documents successfully processed and indexed![/bold green]"
        )

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")


@app.command()
def delete_document(
    doc_id: str = typer.Argument(..., help="Document ID to delete")
):
    """Delete a document and its associated chunks."""
    console.print(f"Deleting document: [bold blue]{doc_id}[/bold blue]")

    try:
        # Get document info before deletion
        doc_info = document_processor.get_document_info(doc_id)
        if not doc_info:
            console.print(
                f"[bold yellow]Document with ID {doc_id} not found[/bold yellow]"
            )
            return

        # Display document details before deletion
        console.print(f"Found document: {doc_info['file_name']}")

        # Confirm deletion
        if not typer.confirm("Are you sure you want to delete this document?"):
            console.print("Deletion canceled.")
            return

        # Delete document
        success = document_processor.delete_document(doc_id)

        if success:
            console.print(
                f"[bold green]Document {doc_id} successfully deleted[/bold green]"
            )

            # Inform about vector DB
            console.print(
                "[bold yellow]Note:[/bold yellow] You may need to rebuild your vector database"
            )
            console.print(
                "Run 'initialize_collection --recreate' to rebuild the vector database with the updated documents"
            )
        else:
            console.print(
                f"[bold red]Failed to delete document {doc_id}[/bold red]"
            )

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")


@app.command()
def update_document(
    file_path: str = typer.Argument(
        ..., help="Path to the updated document file"
    )
):
    """Update an existing document with a new version."""
    console.print(f"Updating document from: [bold blue]{file_path}[/bold blue]")

    try:
        # Generate document ID from file path
        temp_doc_id = document_processor._generate_document_id(file_path)

        # Check if document exists
        existing_doc = document_processor.get_document_info(temp_doc_id)
        if existing_doc:
            console.print(
                f"Updating existing document: {existing_doc['file_name']}"
            )
        else:
            console.print(
                "[yellow]Document doesn't exist yet. Will be added as new.[/yellow]"
            )

        # Update/process document
        metadata = document_processor.update_document(file_path)

        console.print(
            f"Document updated: [bold green]{metadata['doc_id']}[/bold green]"
        )
        console.print(
            f"Created [bold green]{metadata['num_chunks']}[/bold green] chunks"
        )

        # Embed chunks
        console.print("Generating embeddings...")
        embedding_service.embed_chunks(metadata["chunks_path"])

        # Index chunks
        console.print("Indexing chunks in vector database...")
        vector_db_service.index_chunks(metadata["chunks_path"])

        console.print(
            "[bold green]Document successfully updated and indexed![/bold green]"
        )

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")


@app.command()
def list_documents(
    show_details: bool = typer.Option(
        False,
        "--details",
        "-d",
        help="Show detailed information about each document",
    )
):
    """List all processed documents."""
    console.print("[bold blue]Processed Documents[/bold blue]")

    try:
        documents = document_processor.list_documents()

        if not documents:
            console.print(
                "[yellow]No documents have been processed yet.[/yellow]"
            )
            return

        # Create and display table
        table = Table(title=f"Processed Documents ({len(documents)})")
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Filename", style="green")
        table.add_column("Type", style="blue")
        table.add_column("Chunks", style="magenta", justify="right")

        if show_details:
            table.add_column("Processed Path", style="dim")
            table.add_column("Chunks Path", style="dim")

        for doc in documents:
            row = [
                doc["doc_id"],
                doc["file_name"],
                doc["doc_type"],
                str(doc["num_chunks"]),
            ]

            if show_details:
                row.extend([doc["processed_path"], doc["chunks_path"]])

            table.add_row(*row)

        console.print(table)

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
        doc_info = document_processor.get_document_info(doc_id)

        if not doc_info:
            console.print(
                f"[bold yellow]Document with ID {doc_id} not found[/bold yellow]"
            )
            return

        # Display detailed information
        console.print(
            Panel.fit(
                f"[bold blue]Document ID:[/bold blue] {doc_info['doc_id']}\n"
                f"[bold blue]Filename:[/bold blue] {doc_info['file_name']}\n"
                f"[bold blue]Document Type:[/bold blue] {doc_info['doc_type']}\n"
                f"[bold blue]Number of Chunks:[/bold blue] {doc_info['num_chunks']}\n"
                f"[bold blue]Source Path:[/bold blue] {doc_info['file_path']}\n"
                f"[bold blue]Processed Path:[/bold blue] {doc_info['processed_path']}\n"
                f"[bold blue]Chunks Path:[/bold blue] {doc_info['chunks_path']}\n",
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
        vector_db_service.initialize_collection(recreate=recreate)
        console.print(
            "[bold green]Collection initialized successfully![/bold green]"
        )

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")


@app.command()
def rebuild_index():
    """Rebuild the vector database index with all processed documents."""
    console.print("[bold blue]Rebuilding vector database index[/bold blue]")

    try:
        # First check if we need to recreate the collection
        if typer.confirm(
            "Do you want to recreate the collection?", default=True
        ):
            vector_db_service.initialize_collection(recreate=True)
            console.print(
                "[bold green]Collection recreated successfully![/bold green]"
            )

        # Get all documents
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
            vector_db_service.index_chunks(doc["chunks_path"])

        console.print(
            "[bold green]Vector database index rebuilt successfully![/bold green]"
        )

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")


@app.command()
def query(
    query_text: str = typer.Argument(..., help="Query text to search for"),
    top_k: int = typer.Option(
        15, "--top-k", "-k", help="Number of chunks to retrieve"
    ),
    use_multi_query: bool = typer.Option(
        True,
        "--multi-query",
        "-m",
        help="Use multi-query approach for better retrieval",
    ),
):
    """Test a query against the RAG system."""
    console.print(f"Processing query: [bold blue]{query_text}[/bold blue]")

    try:
        # Recognize intent
        intent_info = intent_recognizer.recognize_intent(query_text)
        console.print(
            f"Detected intent: [bold green]{intent_info['intent_type']}[/bold green]"
        )
        console.print(
            f"Detected topic: [bold green]{intent_info['topic']}[/bold green]"
        )
        console.print(
            f"Confidence: [bold green]{intent_info['confidence']:.2f}[/bold green]"
        )

        # Retrieve relevant context
        console.print("Retrieving relevant context...")
        retrieved_chunks = []
        if intent_info["intent_type"] != "off_topic":
            if use_multi_query:
                retrieved_chunks = vector_db_service.multi_query_search(
                    query=query_text, top_k=top_k
                )
            else:
                retrieved_chunks = vector_db_service.search(
                    query=query_text, top_k=top_k
                )

            if retrieved_chunks:
                console.print(
                    f"Retrieved [bold green]{len(retrieved_chunks)}[/bold green] chunks"
                )
                for i, chunk in enumerate(
                    retrieved_chunks[:3]
                ):  # Show only first 3 for brevity
                    console.print(
                        f"[bold blue]Chunk {i+1}[/bold blue] (Score: {chunk['score']:.2f})"
                    )
                    # Show merged status if available
                    if chunk.get("is_merged", False):
                        console.print(
                            "  [bold yellow]MERGED CHUNK[/bold yellow]"
                        )
                    # Show information score if available
                    if "information_score" in chunk:
                        console.print(
                            f"  Information Score: {chunk['information_score']:.2f}"
                        )
                    console.print(f"  {chunk['text'][:100]}...")
                if len(retrieved_chunks) > 3:
                    console.print(
                        f"... and {len(retrieved_chunks) - 3} more chunks"
                    )
            else:
                console.print("[yellow]No relevant chunks found[/yellow]")

        # Generate response
        console.print("Generating response...")
        response_data = response_generator.generate_response(
            query=query_text,
            retrieved_context=retrieved_chunks,
            intent_info=intent_info,
        )

        console.print("\n[bold green]Response:[/bold green]")
        console.print(response_data["response"])

        if "token_usage" in response_data:
            console.print(f"\nToken usage: {response_data['token_usage']}")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")


@app.command()
def interactive(
    top_k: int = typer.Option(
        25, "--top-k", "-k", help="Number of chunks to retrieve for each query"
    ),
    use_multi_query: bool = typer.Option(
        True,
        "--multi-query",
        "-m",
        help="Use multi-query approach for better retrieval",
    ),
):
    """Start an interactive query session."""
    console.print("[bold green]Starting interactive query session[/bold green]")
    console.print("Type 'exit' or 'quit' to end the session")

    while True:
        query_text = console.input("[bold blue]Query:[/bold blue] ")

        if query_text.lower() in ["exit", "quit"]:
            console.print("[bold green]Ending session. Goodbye![/bold green]")
            break

        try:
            # Recognize intent
            intent_info = intent_recognizer.recognize_intent(query_text)
            console.print(f"Intent info:\n{intent_info}\n")

            # Retrieve relevant context
            retrieved_chunks = []
            if intent_info["intent_type"] != "off_topic":
                if use_multi_query:
                    retrieved_chunks = vector_db_service.multi_query_search(
                        query=query_text, top_k=top_k
                    )
                    console.print(f"Retrieved chunks: \n{retrieved_chunks}\n")
                else:
                    retrieved_chunks = vector_db_service.search(
                        query=query_text, top_k=top_k
                    )
                    console.print(f"Retrieved chunks: \n{retrieved_chunks}\n")

            # Generate response
            response_data = response_generator.generate_response(
                query=query_text,
                retrieved_context=retrieved_chunks,
                intent_info=intent_info,
            )
            console.print(f"Response Data\n{response_data}\n")

            console.print("\n[bold green]Response:[/bold green]")
            console.print(response_data["response"])
            console.print()

        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")


@app.command()
def deduplication_status():
    """Show the status of deduplication in the system."""
    console.print("[bold blue]Deduplication Status[/bold blue]")

    # Check settings - using simpler approach without complex formatting
    if settings.deduplication.enabled:
        console.print("Deduplication Enabled: [bold green]True[/bold green]")
    else:
        console.print("Deduplication Enabled: [bold red]False[/bold red]")

    console.print(
        f"Similarity Threshold: [bold blue]{settings.deduplication.similarity_threshold}[/bold blue]"
    )

    # Check if deduplicated files exist
    dedup_dir = document_processor.dedup_dir
    dedup_file = dedup_dir / "deduplicated_chunks.jsonl"
    report_file = dedup_dir / "deduplication_report.json"

    if dedup_file.exists():
        # Count chunks
        chunk_count = 0
        with open(dedup_file, "r") as f:
            for _ in f:
                chunk_count += 1

        console.print(
            f"Deduplicated Chunks: [bold green]{chunk_count}[/bold green]"
        )

        # Load and display report summary if available
        if report_file.exists():
            import json

            try:
                with open(report_file, "r") as f:
                    report = json.load(f)

                # Create stats table
                table = Table(title="Deduplication Statistics")
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="green")

                stats = report.get("stats", {})
                table.add_row(
                    "Original Chunks",
                    str(stats.get("total_original_chunks", "N/A")),
                )
                table.add_row(
                    "Deduplicated Chunks",
                    str(stats.get("total_deduplicated_chunks", "N/A")),
                )
                table.add_row(
                    "Merged Chunks", str(stats.get("merged_chunks", "N/A"))
                )
                table.add_row(
                    "Unchanged Chunks",
                    str(stats.get("unchanged_chunks", "N/A")),
                )
                table.add_row(
                    "Text Reduction",
                    f"{stats.get('text_reduction', 'N/A')} characters",
                )

                # Handle the percentage formatting separately to avoid errors
                percentage = stats.get("reduction_percentage", "N/A")
                if isinstance(percentage, (int, float)):
                    percentage_str = f"{percentage:.2f}%"
                else:
                    percentage_str = "N/A"
                table.add_row("Reduction Percentage", percentage_str)

                console.print(table)

                # Display some of the merged chunks
                merged_chunks = report.get("merged_chunks", [])
                if merged_chunks:
                    console.print(
                        f"\nTop merged chunks: [bold blue]{min(5, len(merged_chunks))} of {len(merged_chunks)}[/bold blue]"
                    )
                    for i, chunk in enumerate(merged_chunks[:5]):
                        console.print(
                            f"  [bold]{i+1}.[/bold] Chunk ID: {chunk.get('id', 'N/A')}"
                        )
                        console.print(
                            f"     Merged from {len(chunk.get('merged_from', []))} sources"
                        )

                        # Handle the score formatting separately to avoid errors
                        score = chunk.get("information_score", "N/A")
                        if isinstance(score, (int, float)):
                            score_str = f"{score:.4f}"
                        else:
                            score_str = "N/A"
                        console.print(f"     Information score: {score_str}")

            except Exception as e:
                console.print(
                    f"[bold red]Error reading deduplication report:[/bold red] {str(e)}"
                )
    else:
        console.print("[yellow]No deduplicated chunks found.[/yellow]")
        console.print(f"Expected location: {dedup_file}")
        console.print(
            "Run 'process_all_documents' to generate deduplicated chunks."
        )


@app.command()
def run_deduplication():
    """Run the deduplication process on existing chunks."""
    if not settings.deduplication.enabled:
        console.print(
            "[bold yellow]Deduplication is disabled in settings.[/bold yellow]"
        )
        console.print(
            "Enable it by setting deduplication.enabled=true in config.yaml"
        )
        return

    console.print("[bold blue]Running deduplication process...[/bold blue]")

    try:
        # Create a document processor with deduplication enabled
        processor = DocumentProcessor(enable_deduplication=True)

        # Load existing chunks
        console.print("Loading existing chunks...")
        chunks = []
        chunks_dir = processor.chunk_dir

        for chunk_file in chunks_dir.glob("*_chunks.jsonl"):
            with open(chunk_file, "r") as f:
                doc_chunks = [json.loads(line) for line in f]
                console.print(
                    f"Loaded {len(doc_chunks)} chunks from {chunk_file.name}"
                )

                # Convert to Chunk objects
                for chunk_data in doc_chunks:
                    chunk = processor.Chunk(
                        chunk_id=chunk_data["chunk_id"],
                        doc_id=chunk_data["doc_id"],
                        chunk_index=chunk_data["chunk_index"],
                        text=chunk_data["text"],
                        metadata=chunk_data.get("metadata", {}),
                        source_file=str(chunk_file),
                    )
                    chunks.append(chunk)

        if not chunks:
            console.print(
                "[bold yellow]No chunks found to deduplicate.[/bold yellow]"
            )
            return

        console.print(
            f"Loaded [bold green]{len(chunks)}[/bold green] total chunks"
        )

        # Store chunks for deduplication
        processor.all_chunks = chunks

        # Run deduplication
        console.print("Running deduplication process...")
        processor._deduplicate_chunks()

        # Save deduplicated chunks
        deduplicated_path = processor.dedup_dir / "deduplicated_chunks.jsonl"
        processor._save_deduplicated_chunks(deduplicated_path)
        console.print(
            f"Saved [bold green]{len(processor.deduplicated_chunks)}[/bold green] deduplicated chunks"
        )

        # Generate report
        processor._generate_deduplication_report()
        console.print("[bold green]Deduplication complete![/bold green]")
        console.print(
            f"Report saved to {processor.dedup_dir}/deduplication_report.json"
        )

        # Suggest next steps
        console.print("\n[bold blue]Next Steps:[/bold blue]")
        console.print(
            "1. Run 'embed_deduplicated' to generate embeddings for deduplicated chunks"
        )
        console.print(
            "2. Run 'index_deduplicated' to index deduplicated chunks in the vector database"
        )

    except Exception as e:
        console.print(
            f"[bold red]Error during deduplication:[/bold red] {str(e)}"
        )
        import traceback

        console.print(traceback.format_exc())


@app.command()
def embed_deduplicated():
    """Generate embeddings for deduplicated chunks."""
    console.print(
        "[bold blue]Generating embeddings for deduplicated chunks...[/bold blue]"
    )

    try:
        # Check if deduplicated chunks exist
        dedup_file = document_processor.dedup_dir / "deduplicated_chunks.jsonl"
        if not dedup_file.exists():
            console.print(
                "[bold yellow]Deduplicated chunks not found.[/bold yellow]"
            )
            console.print(
                "Run 'run_deduplication' first to generate deduplicated chunks."
            )
            return

        # Generate embeddings
        console.print("Generating embeddings...")
        embedding_service.embed_deduplicated_chunks()

        console.print(
            "[bold green]Embeddings generated successfully![/bold green]"
        )

    except Exception as e:
        console.print(
            f"[bold red]Error generating embeddings:[/bold red] {str(e)}"
        )


@app.command()
def index_deduplicated():
    """Index deduplicated chunks in the vector database."""
    console.print("[bold blue]Indexing deduplicated chunks...[/bold blue]")

    try:
        # Check if deduplicated chunks exist
        dedup_file = document_processor.dedup_dir / "deduplicated_chunks.jsonl"
        if not dedup_file.exists():
            console.print(
                "[bold yellow]Deduplicated chunks not found.[/bold yellow]"
            )
            console.print(
                "Run 'run_deduplication' first to generate deduplicated chunks."
            )
            return

        # Check if embeddings exist
        embeddings_file = (
            document_processor.dedup_dir.parent
            / "embeddings"
            / "deduplicated_embeddings.npz"
        )
        if not embeddings_file.exists():
            console.print(
                "[bold yellow]Deduplicated embeddings not found.[/bold yellow]"
            )
            console.print(
                "Run 'embed_deduplicated' first to generate embeddings."
            )
            return

        # Index chunks
        console.print("Indexing chunks...")
        vector_db_service.index_chunks(dedup_file)

        console.print("[bold green]Chunks indexed successfully![/bold green]")
        console.print(
            "The vector database will now use the deduplicated chunks for retrieval."
        )

    except Exception as e:
        console.print(f"[bold red]Error indexing chunks:[/bold red] {str(e)}")


if __name__ == "__main__":
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        console.print(
            "[bold yellow]Warning:[/bold yellow] OPENAI_API_KEY environment variable not set"
        )
        console.print("Set it by running: export OPENAI_API_KEY=your_key_here")

    app()
