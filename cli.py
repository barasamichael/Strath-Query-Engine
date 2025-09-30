import os
import typer
import logging
from rich.console import Console
from rich.logging import RichHandler

from services.chunking.document_processor import DocumentProcessor
from services.retrieval.embeddings import EmbeddingService
from services.retrieval.vector_db import VectorDBService
from services.generation.intent_recognizer import IntentRecognizer
from services.generation.response_generator import ResponseGenerator
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
document_processor = DocumentProcessor()
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

        # Embed all chunks
        console.print("Generating embeddings for all chunks...")
        embedding_service.embed_chunks()

        # Index all chunks
        console.print("Indexing all chunks in vector database...")
        vector_db_service.index_chunks()

        console.print(
            "[bold green]All documents successfully processed and indexed![/bold green]"
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

            # Retrieve relevant context
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

            # Generate response
            response_data = response_generator.generate_response(
                query=query_text,
                retrieved_context=retrieved_chunks,
                intent_info=intent_info,
            )

            console.print("\n[bold green]Response:[/bold green]")
            console.print(response_data["response"])
            console.print()

        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")


if __name__ == "__main__":
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        console.print(
            "[bold yellow]Warning:[/bold yellow] OPENAI_API_KEY environment variable not set"
        )
        console.print("Set it by running: export OPENAI_API_KEY=your_key_here")

    app()
