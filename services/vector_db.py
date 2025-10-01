import os
import json
import logging
import pickle
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

from config.settings import settings, ROOT_DIR
from services.embeddings import EmbeddingService

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vector_db")


class SimpleVectorDB:
    """A simple in-memory vector database that stores embeddings and metadata."""

    def __init__(self):
        self.vectors = []
        self.metadata = []

    def add_vectors(self, vectors, metadata_list):
        """Add vectors and their metadata to the database."""
        self.vectors.extend(vectors)
        self.metadata.extend(metadata_list)

    def search(self, query_vector, top_k=20, filter_fn=None):
        """Search for the most similar vectors to the query vector."""
        if not self.vectors:
            return []

        # Convert vectors to numpy array if not already
        vectors_array = np.array(self.vectors)

        # Compute similarities
        similarities = cosine_similarity([query_vector], vectors_array)[0]

        # Get indices sorted by similarity (descending)
        sorted_indices = np.argsort(similarities)[::-1]

        # Apply filter if provided
        if filter_fn:
            filtered_indices = [
                idx for idx in sorted_indices if filter_fn(self.metadata[idx])
            ]
            sorted_indices = filtered_indices

        # Get top k results
        top_indices = sorted_indices[:top_k]

        # Format results
        results = []
        for idx in top_indices:
            results.append(
                {
                    **self.metadata[idx],
                    "score": float(
                        similarities[idx]
                    ),  # Convert to Python float for serialization
                }
            )

        return results

    def save(self, path):
        """Save the database to a file."""
        with open(path, "wb") as f:
            pickle.dump({"vectors": self.vectors,
                        "metadata": self.metadata}, f)

    def load(self, path):
        """Load the database from a file."""
        if os.path.exists(path):
            with open(path, "rb") as f:
                data = pickle.load(f)
                self.vectors = data["vectors"]
                self.metadata = data["metadata"]
            return True
        return False


class VectorDBService:
    def __init__(self, embedding_service: Optional[EmbeddingService] = None):
        self.embedding_service = embedding_service or EmbeddingService()
        self.dimension = self.embedding_service.dimension
        self.collection_name = settings.vector_db.collection_name

        # Initialize simple vector DB
        self.db = SimpleVectorDB()

        # Define paths
        self.db_path = ROOT_DIR / "database" / "vector_store"
        if not self.db_path.exists():
            self.db_path.mkdir(parents=True, exist_ok=True)

        self.db_file = self.db_path / f"{self.collection_name}.pkl"

        # Try to load existing database
        if os.path.exists(self.db_file):
            self.db.load(self.db_file)
            logger.info(
                f"Loaded vector database with {len(self.db.vectors)} vectors"
            )

    def initialize_collection(self, recreate: bool = False) -> None:
        """Initialize or recreate the vector collection."""
        if recreate or not os.path.exists(self.db_file):
            self.db = SimpleVectorDB()
            logger.info(
                f"{'Recreated' if recreate else 'Initialized'} vector database"
            )
        else:
            self.db.load(self.db_file)
            logger.info(
                f"Vector database already exists with {len(self.db.vectors)} vectors"
            )

    def index_chunks(self, chunks_dir: Optional[Path] = None) -> None:
        """Index all chunks from the chunks directory into the vector database."""
        chunks_dir = chunks_dir or ROOT_DIR / "data" / "chunks"

        if not chunks_dir.exists():
            logger.error(f"Chunks directory not found: {chunks_dir}")
            return

        # Reset the database to avoid duplicates
        self.db = SimpleVectorDB()
        logger.info("Reset vector database to avoid duplicates")

        # Process each chunks file
        for chunks_file in tqdm(
            list(chunks_dir.glob("*_chunks.jsonl")), desc="Indexing files"
        ):
            try:
                self._index_chunks_file(chunks_file)
            except Exception as e:
                logger.error(f"Error indexing {chunks_file}: {str(e)}")

        # Save the database
        self.db.save(self.db_file)
        logger.info(
            f"Saved vector database with {len(self.db.vectors)} vectors"
        )

    def _index_chunks_file(self, chunks_file: Path) -> None:
        """Index chunks from a single file into the vector database."""
        logger.info(f"Indexing chunks from: {chunks_file}")

        # Check if embeddings already exist
        doc_id = chunks_file.stem.replace("_chunks", "")
        embeddings_file = (
            ROOT_DIR / "data" / "embeddings" / f"{doc_id}_embeddings.npz"
        )

        if not embeddings_file.exists():
            # Generate embeddings if they don't exist
            logger.info(
                f"Embeddings not found for {doc_id}, generating them now"
            )
            self.embedding_service.embed_chunks(chunks_file)

        # Load embeddings
        embeddings_data = np.load(embeddings_file)
        embeddings = embeddings_data["embeddings"]
        chunk_ids = embeddings_data["chunk_ids"]

        # Load chunks to get full metadata
        chunks = []
        with open(chunks_file, "r", encoding="utf-8") as f:
            for line in f:
                chunks.append(json.loads(line))

        # Add vectors to the database
        vectors = []
        metadata_list = []

        for i, (chunk, chunk_id) in enumerate(zip(chunks, chunk_ids)):
            if i >= len(embeddings):
                logger.warning(
                    f"Mismatch between chunks and embeddings for {doc_id}"
                )
                break

            # Create metadata
            metadata = {
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"],
            }

            vectors.append(embeddings[i].tolist())
            metadata_list.append(metadata)

        # Add vectors to the database
        self.db.add_vectors(vectors, metadata_list)

        logger.info(f"Indexed {len(vectors)} chunks from {doc_id}")

    def search(
        self, query: str, top_k: int = 20, filter_doc_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search for chunks similar to the query."""
        # Generate query embedding
        query_embedding = self.embedding_service.embed_query(query)

        # Prepare filter if needed
        filter_fn = None
        if filter_doc_id:
            def filter_fn(metadata): return metadata.get(
                "doc_id") == filter_doc_id

        # Perform search
        results = self.db.search(
            query_vector=query_embedding.tolist(),
            top_k=top_k,
            filter_fn=filter_fn,
        )

        return results

    def multi_query_search(
        self, query: str, top_k: int = 20, filter_doc_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search using multiple query formulations to get more comprehensive results.
        This helps retrieve more diverse and relevant chunks for complex queries.
        """
        # Original query results (get more of the requested chunks from direct query)
        original_results = self.search(
            query=query, top_k=int(top_k * 0.7), filter_doc_id=filter_doc_id
        )

        # Generate alternative query formulations that are more focused
        alt_queries = [
            # More specific reformulations
            f"{query} at Strathmore University",
            f"{query} strathmore policy",
            f"{query} requirements strathmore",
        ]

        # Get results for alternative queries
        all_results = original_results.copy()
        seen_chunks = {result["chunk_id"] for result in original_results}

        # Try each alternative query until we have enough results
        for alt_query in alt_queries:
            if len(all_results) >= top_k:
                break

            # Get results for this alternative query
            alt_results = self.search(
                query=alt_query, top_k=3, filter_doc_id=filter_doc_id
            )

            # Add only new chunks
            for result in alt_results:
                if result["chunk_id"] not in seen_chunks:
                    all_results.append(result)
                    seen_chunks.add(result["chunk_id"])

        # Sort by relevance score
        all_results.sort(key=lambda x: x["score"], reverse=True)

        # Return top_k results
        return all_results[:top_k]
