import json
import logging
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Union
from typing import Optional

import chromadb
from chromadb.config import Settings

from config.settings import settings, ROOT_DIR
from services.embeddings import EmbeddingService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vector_db")


def ensure_path(path_input: Union[str, Path, None]) -> Optional[Path]:
    """
    Ensure input is converted to Path object with proper error handling.

    Args:
        path_input: String path, Path object, or None

    Returns:
        Path object or None if input was None

    Raises:
        ValueError: If path_input is not a valid path type
    """
    if path_input is None:
        return None

    if isinstance(path_input, Path):
        return path_input

    if isinstance(path_input, str):
        if not path_input.strip():
            return None
        return Path(path_input)

    raise ValueError(
        f"Invalid path type: {type(path_input)}. Expected str, Path, or None."
    )


def safe_path_operation(func):
    """
    Decorator to safely handle path operations by converting string arguments to Path objects.
    """

    def wrapper(*args, **kwargs):
        # Convert string paths to Path objects in args
        new_args = []
        for arg in args:
            if isinstance(arg, str) and (
                "/" in arg
                or "\\" in arg
                or arg.endswith(
                    (".txt", ".json", ".jsonl", ".npz", ".md", ".pdf", ".docx")
                )
            ):
                new_args.append(Path(arg))
            else:
                new_args.append(arg)

        # Convert string paths to Path objects in kwargs
        new_kwargs = {}
        for key, value in kwargs.items():
            if key.endswith(("_path", "_dir", "_file")) or key in (
                "path",
                "file_path",
                "dir_path",
                "output_path",
                "input_path",
                "chunks_file",
            ):
                if isinstance(value, str):
                    new_kwargs[key] = Path(value)
                else:
                    new_kwargs[key] = value
            else:
                new_kwargs[key] = value

        return func(*new_args, **new_kwargs)

    return wrapper


class VectorDBError(Exception):
    """Custom exception for vector database errors."""

    pass


class VectorDBService:
    """Production-ready vector database service using ChromaDB with bulletproof path handling."""

    def __init__(self, embedding_service: Optional[EmbeddingService] = None):
        try:
            self.embedding_service = embedding_service or EmbeddingService()
            self.dimension = self.embedding_service.dimension
            self.collection_name = settings.vector_db.collection_name

            # Define ChromaDB path with bulletproof handling
            self.db_path = ensure_path(ROOT_DIR) / "database" / "chroma_db"
            if not self.db_path:
                raise VectorDBError("Invalid database path configuration")

            try:
                self.db_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise VectorDBError(
                    f"Failed to create database directory {self.db_path}: {str(e)}"
                )

            # Initialize ChromaDB client with persistent storage
            self.client = chromadb.PersistentClient(
                path=str(self.db_path),
                settings=Settings(anonymized_telemetry=False, allow_reset=True),
            )

            # Get or create collection
            self.collection = self._get_or_create_collection()

            logger.info(f"ChromaDB initialized at {self.db_path}")
            logger.info(
                f"Collection '{self.collection_name}' has {self.collection.count()} vectors"
            )

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {str(e)}")
            raise VectorDBError(f"Database initialization failed: {str(e)}")

    def _get_or_create_collection(self):
        """Get existing collection or create new one."""
        try:
            # Try to get existing collection
            collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"Loaded existing collection: {self.collection_name}")
            return collection
        except Exception:
            # Create new collection if it doesn't exist
            logger.info(f"Creating new collection: {self.collection_name}")
            return self.client.create_collection(
                name=self.collection_name, metadata={"hnsw:space": "cosine"}
            )

    def initialize_collection(self, recreate: bool = False) -> None:
        """Initialize or recreate the vector collection."""
        try:
            if recreate:
                # Delete existing collection
                try:
                    self.client.delete_collection(name=self.collection_name)
                    logger.info(
                        f"Deleted existing collection: {self.collection_name}"
                    )
                except Exception as e:
                    logger.warning(
                        f"No existing collection to delete: {str(e)}"
                    )

                # Create new collection
                self.collection = self.client.create_collection(
                    name=self.collection_name, metadata={"hnsw:space": "cosine"}
                )
                logger.info(f"Created new collection: {self.collection_name}")
            else:
                logger.info(
                    f"Collection already exists with {self.collection.count()} vectors"
                )

        except Exception as e:
            logger.error(f"Error initializing collection: {str(e)}")
            raise VectorDBError(f"Collection initialization failed: {str(e)}")

    @safe_path_operation
    def index_chunks(
        self, chunks_file: Optional[Union[str, Path]] = None
    ) -> None:
        """Index chunks with deduplication preference and error recovery."""
        try:
            # Check for deduplicated chunks first
            dedup_dir = ensure_path(ROOT_DIR) / "data" / "deduplicated"
            if not dedup_dir:
                raise VectorDBError("Invalid dedup directory path")

            dedup_file = dedup_dir / "deduplicated_chunks.jsonl"

            if dedup_file.exists():
                logger.info("Found deduplicated chunks, indexing those")
                self._index_deduplicated_chunks(dedup_file)
                return

            # Process standard chunks
            if chunks_file:
                chunks_file = ensure_path(chunks_file)
                if not chunks_file:
                    raise VectorDBError("Invalid chunks file path provided")

                chunks_dir = chunks_file.parent
            else:
                chunks_dir = ensure_path(ROOT_DIR) / "data" / "chunks"
                if not chunks_dir:
                    raise VectorDBError("Invalid chunks directory path")

            if not chunks_dir.exists():
                raise VectorDBError(f"Chunks directory not found: {chunks_dir}")

            files_to_process = (
                [chunks_file]
                if chunks_file
                else list(chunks_dir.glob("*_chunks.jsonl"))
            )

            if not files_to_process:
                raise VectorDBError("No chunk files found to index")

            # Process each file
            successful = 0
            failed = 0

            for chunk_file in files_to_process:
                try:
                    self._index_chunks_file(chunk_file)
                    successful += 1
                except Exception as e:
                    logger.error(f"Failed to index {chunk_file.name}: {str(e)}")
                    failed += 1

            logger.info(
                f"Indexing complete: {successful} succeeded, {failed} failed"
            )
            logger.info(
                f"Total vectors in collection: {self.collection.count()}"
            )

        except Exception as e:
            logger.error(f"Error during indexing: {str(e)}")
            raise VectorDBError(f"Indexing failed: {str(e)}")

    def _index_chunks_file(self, chunks_file: Union[str, Path]) -> None:
        """Index chunks from a single file with embedding cost optimization."""
        # Ensure chunks_file is a Path object
        chunks_file = ensure_path(chunks_file)
        if not chunks_file:
            raise VectorDBError("Invalid chunks file path")

        try:
            logger.info(f"Indexing chunks from: {chunks_file.name}")

            # Get document ID
            doc_id = chunks_file.stem.replace("_chunks", "")

            # Check if embeddings exist
            embeddings_dir = ensure_path(ROOT_DIR) / "data" / "embeddings"
            if not embeddings_dir:
                raise VectorDBError("Invalid embeddings directory path")

            embeddings_file = embeddings_dir / f"{doc_id}_embeddings.npz"

            if not embeddings_file.exists():
                logger.info(f"Generating embeddings for {doc_id}")
                self.embedding_service.embed_chunks(chunks_file)

            # Load embeddings and chunks
            embeddings_data = self.embedding_service.load_embeddings(
                embeddings_file
            )
            if embeddings_data is None:
                raise VectorDBError(f"Failed to load embeddings for {doc_id}")

            embeddings = embeddings_data["embeddings"]
            chunk_ids = embeddings_data["chunk_ids"]

            # Load chunk metadata
            chunks = []
            with open(chunks_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        chunks.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(
                            f"Skipping invalid JSON line in {chunks_file.name}: {str(e)}"
                        )

            if len(chunks) != len(embeddings):
                logger.warning(
                    f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings"
                )
                min_len = min(len(chunks), len(embeddings))
                chunks = chunks[:min_len]
                embeddings = embeddings[:min_len]
                chunk_ids = chunk_ids[:min_len]

            # Prepare data for ChromaDB
            ids = [str(chunk["chunk_id"]) for chunk in chunks]
            documents = [chunk["text"] for chunk in chunks]
            metadatas = [
                {
                    "doc_id": chunk["doc_id"],
                    "chunk_index": chunk["chunk_index"],
                    "chunk_id": chunk["chunk_id"],
                }
                for chunk in chunks
            ]

            # Add to collection in batches
            batch_size = 100
            for i in range(0, len(ids), batch_size):
                batch_ids = ids[i : i + batch_size]
                batch_embeddings = embeddings[i : i + batch_size].tolist()
                batch_documents = documents[i : i + batch_size]
                batch_metadatas = metadatas[i : i + batch_size]

                self.collection.add(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    documents=batch_documents,
                    metadatas=batch_metadatas,
                )

            logger.info(f"Successfully indexed {len(ids)} chunks from {doc_id}")

        except Exception as e:
            logger.error(
                f"Error indexing chunks file {chunks_file.name}: {str(e)}"
            )
            raise VectorDBError(f"Failed to index {chunks_file.name}: {str(e)}")

    def _index_deduplicated_chunks(self, dedup_file: Union[str, Path]) -> None:
        """Index deduplicated chunks with error handling."""
        # Ensure dedup_file is a Path object
        dedup_file = ensure_path(dedup_file)
        if not dedup_file:
            raise VectorDBError("Invalid deduplicated file path")

        try:
            # Check for embeddings
            embeddings_dir = ensure_path(ROOT_DIR) / "data" / "embeddings"
            if not embeddings_dir:
                raise VectorDBError("Invalid embeddings directory path")

            embeddings_file = embeddings_dir / "deduplicated_embeddings.npz"

            if not embeddings_file.exists():
                logger.info("Generating embeddings for deduplicated chunks")
                self.embedding_service.embed_deduplicated_chunks()

            # Load embeddings and chunks
            embeddings_data = self.embedding_service.load_embeddings(
                embeddings_file
            )
            if embeddings_data is None:
                raise VectorDBError("Failed to load deduplicated embeddings")

            embeddings = embeddings_data["embeddings"]

            # Load chunks
            chunks = []
            with open(dedup_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        chunks.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping invalid JSON line: {str(e)}")

            if len(chunks) != len(embeddings):
                logger.warning(
                    f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings"
                )
                min_len = min(len(chunks), len(embeddings))
                chunks = chunks[:min_len]
                embeddings = embeddings[:min_len]

            # Prepare data
            ids = [str(chunk["chunk_id"]) for chunk in chunks]
            documents = [chunk["text"] for chunk in chunks]
            metadatas = []

            for chunk in chunks:
                metadata = {
                    "doc_id": chunk.get("doc_id", ""),
                    "chunk_index": chunk.get("chunk_index", 0),
                    "chunk_id": chunk["chunk_id"],
                }

                # Add deduplication metadata
                if "information_score" in chunk:
                    metadata["information_score"] = chunk["information_score"]
                if "merged_from" in chunk:
                    metadata["is_merged"] = True
                    metadata["merged_count"] = len(chunk["merged_from"])

                metadatas.append(metadata)

            # Add to collection in batches
            batch_size = 100
            for i in range(0, len(ids), batch_size):
                batch_ids = ids[i : i + batch_size]
                batch_embeddings = embeddings[i : i + batch_size].tolist()
                batch_documents = documents[i : i + batch_size]
                batch_metadatas = metadatas[i : i + batch_size]

                self.collection.add(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    documents=batch_documents,
                    metadatas=batch_metadatas,
                )

            logger.info(f"Successfully indexed {len(ids)} deduplicated chunks")

        except Exception as e:
            logger.error(f"Error indexing deduplicated chunks: {str(e)}")
            raise VectorDBError(
                f"Failed to index deduplicated chunks: {str(e)}"
            )

    def search(
        self, query: str, top_k: int = 20, filter_doc_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search with comprehensive error handling."""
        try:
            if self.collection.count() == 0:
                logger.warning("Collection is empty")
                return []

            # Generate query embedding
            query_embedding = self.embedding_service.embed_query(query)

            if query_embedding is None:
                raise VectorDBError("Failed to generate query embedding")

            # Prepare filter
            where_filter = {"doc_id": filter_doc_id} if filter_doc_id else None

            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                where=where_filter,
            )

            # Format results
            formatted_results = []
            if results and results["ids"] and len(results["ids"]) > 0:
                for i in range(len(results["ids"][0])):
                    result = {
                        "chunk_id": results["metadatas"][0][i].get(
                            "chunk_id", ""
                        ),
                        "doc_id": results["metadatas"][0][i].get("doc_id", ""),
                        "chunk_index": results["metadatas"][0][i].get(
                            "chunk_index", 0
                        ),
                        "text": results["documents"][0][i],
                        # Convert distance to similarity
                        "score": 1 - results["distances"][0][i],
                    }

                    # Add optional metadata
                    if "information_score" in results["metadatas"][0][i]:
                        result["information_score"] = results["metadatas"][0][
                            i
                        ]["information_score"]
                        # Boost score slightly for high-information chunks
                        result["score"] = result["score"] * (
                            1 + 0.1 * result["information_score"]
                        )

                    if results["metadatas"][0][i].get("is_merged"):
                        result["is_merged"] = True
                        result["merged_count"] = results["metadatas"][0][i].get(
                            "merged_count", 0
                        )

                    formatted_results.append(result)

            # Re-sort by adjusted score
            formatted_results.sort(key=lambda x: x["score"], reverse=True)
            print(formatted_results)

            return formatted_results

        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            raise VectorDBError(f"Search failed: {str(e)}")

    def multi_query_search(
        self, query: str, top_k: int = 20, filter_doc_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Multi-query search with error recovery."""
        try:
            # Get results from original query
            original_results = self.search(
                query=query, top_k=int(top_k * 0.7), filter_doc_id=filter_doc_id
            )

            # Generate alternative queries
            alt_queries = [
                f"{query} at Strathmore University",
                f"{query} strathmore policy",
                f"{query} requirements strathmore",
            ]

            all_results = original_results.copy()
            seen_chunks = {r["chunk_id"] for r in original_results}

            # Try alternative queries
            for alt_query in alt_queries:
                if len(all_results) >= top_k:
                    break

                try:
                    alt_results = self.search(
                        query=alt_query, top_k=3, filter_doc_id=filter_doc_id
                    )

                    for result in alt_results:
                        if result["chunk_id"] not in seen_chunks:
                            all_results.append(result)
                            seen_chunks.add(result["chunk_id"])

                except Exception as e:
                    logger.warning(f"Alternative query failed: {str(e)}")
                    continue

            # Sort and limit
            all_results.sort(key=lambda x: x["score"], reverse=True)
            return all_results[:top_k]

        except Exception as e:
            logger.error(f"Multi-query search error: {str(e)}")
            # Fallback to regular search
            try:
                return self.search(query, top_k, filter_doc_id)
            except:
                return []

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        try:
            return {
                "name": self.collection_name,
                "count": self.collection.count(),
                "dimension": self.dimension,
            }
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {"error": str(e)}
