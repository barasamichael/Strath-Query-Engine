import sys
import pytest
import logging
from pathlib import Path

from services.generation.intent_recognizer import IntentType
from services.generation.intent_recognizer import TopicCategory
from services.generation.intent_recognizer import IntentRecognizer
from services.generation.response_generator import ResponseGenerator

from services.retrieval.vector_db import VectorDBService
from services.retrieval.embeddings import EmbeddingService
from services.chunking.document_processor import DocumentProcessor

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tests")

# Test data
TEST_QUERIES = [
    "What are the main academic departments at Strathmore University?",
    "How can I apply for a scholarship?",
    "What is the dress code for students?",
    "Tell me about the library services.",
    "What happens if I miss an exam?",
    "Hi, how are you doing today?",
    "What's the price of bitcoin right now?",
]


@pytest.fixture
def document_processor():
    return DocumentProcessor()


@pytest.fixture
def embedding_service():
    return EmbeddingService()


@pytest.fixture
def vector_db_service(embedding_service):
    return VectorDBService(embedding_service=embedding_service)


@pytest.fixture
def intent_recognizer():
    return IntentRecognizer()


@pytest.fixture
def response_generator():
    return ResponseGenerator()


def test_intent_recognition(intent_recognizer):
    """Test intent recognition on sample queries."""
    logger.info("Testing intent recognition...")

    for query in TEST_QUERIES:
        intent_info = intent_recognizer.recognize_intent(query)
        logger.info(f"Query: '{query}'")
        logger.info(f"  Intent: {intent_info['intent_type']}")
        logger.info(f"  Topic: {intent_info['topic']}")
        logger.info(f"  Confidence: {intent_info['confidence']:.2f}")

        # Basic assertions
        assert "intent_type" in intent_info
        assert "topic" in intent_info
        assert "confidence" in intent_info
        assert isinstance(intent_info["intent_type"], IntentType)
        assert isinstance(intent_info["topic"], TopicCategory)
        assert 0 <= intent_info["confidence"] <= 1

    # Specific intent assertions
    assert (
        intent_recognizer.recognize_intent("What is the dress code?")[
            "intent_type"
        ]
        == IntentType.FACTUAL_QUERY
    )
    assert (
        intent_recognizer.recognize_intent("How do I apply for a scholarship?")[
            "intent_type"
        ]
        == IntentType.PROCEDURAL_QUERY
    )
    assert (
        intent_recognizer.recognize_intent("Why is attendance mandatory?")[
            "intent_type"
        ]
        == IntentType.EXPLANATION_QUERY
    )
    assert (
        intent_recognizer.recognize_intent("Thanks for the help!")[
            "intent_type"
        ]
        == IntentType.FEEDBACK
    )

    # Off-topic detection
    assert (
        intent_recognizer.recognize_intent("What's the price of Bitcoin?")[
            "intent_type"
        ]
        == IntentType.OFF_TOPIC
    )
    assert (
        intent_recognizer.recognize_intent("Tell me a joke")["intent_type"]
        == IntentType.OFF_TOPIC
    )


def test_document_processor_setup(document_processor):
    """Test document processor directory setup."""
    assert document_processor.raw_dir.exists()
    assert document_processor.processed_dir.exists()
    assert document_processor.chunk_dir.exists()


def test_embedding_service_setup(embedding_service):
    """Test embedding service initialization."""
    assert embedding_service.model is not None
    assert embedding_service.dimension > 0

    # Test query embedding
    query_embedding = embedding_service.embed_query("Test query")
    assert query_embedding.shape[0] == embedding_service.dimension


def test_vector_db_service_setup(vector_db_service):
    """Test vector database service initialization."""
    assert vector_db_service.client is not None
    assert vector_db_service.dimension > 0

    # Initialize collection (if it doesn't exist)
    try:
        vector_db_service.initialize_collection()
    except Exception as e:
        logger.error(f"Error initializing collection: {str(e)}")
        pytest.fail(f"Vector DB initialization failed: {str(e)}")


def test_end_to_end_query(
    vector_db_service, intent_recognizer, response_generator
):
    """Test end-to-end query processing (if data is available)."""
    # Check if we have any data in the vector database
    collections = vector_db_service.client.get_collections().collections
    collection_names = [collection.name for collection in collections]

    if vector_db_service.collection_name not in collection_names:
        logger.warning(
            f"Collection {vector_db_service.collection_name} not found. "
            + "Skipping end-to-end test."
        )
        pytest.skip("No vector database collection found")

    # Use a simple query
    query = "What is the dress code at Strathmore?"

    # Recognize intent
    intent_info = intent_recognizer.recognize_intent(query)
    logger.info(f"Query intent: {intent_info['intent_type']}")

    # Retrieve relevant context
    retrieved_chunks = []
    try:
        if intent_info["intent_type"] != "off_topic":
            retrieved_chunks = vector_db_service.search(query=query, top_k=3)
            logger.info(f"Retrieved {len(retrieved_chunks)} chunks")
    except Exception as e:
        logger.warning(f"Retrieval failed: {str(e)}")

    # Generate response
    try:
        response_data = response_generator.generate_response(
            query=query,
            retrieved_context=retrieved_chunks,
            intent_info=intent_info,
        )
        logger.info(f"Generated response: {response_data['response'][:100]}...")

        # Basic assertions
        assert "response" in response_data
        assert len(response_data["response"]) > 0
    except Exception as e:
        logger.error(f"Response generation failed: {str(e)}")
        pytest.fail(f"End-to-end test failed: {str(e)}")
