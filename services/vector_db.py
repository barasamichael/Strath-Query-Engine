import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Union, Optional
from datetime import datetime

import chromadb
from chromadb.config import Settings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config.settings import settings, ROOT_DIR
from services.embeddings import EmbeddingService
from services.tavily_service import TavilyService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vector_db")


class VectorDBService:
    """
    Vector database service with hybrid search (semantic + TF-IDF keyword).

    Real-time web retrieval (Tavily) is handled exclusively by ResponseGenerator.
    The tavily_service parameter is kept for diagnostic endpoints only.
    """

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        tavily_service: Optional[TavilyService] = None,
    ):
        try:
            self.embedding_service = embedding_service or EmbeddingService()
            self.tavily_service = tavily_service
            self.dimension = self.embedding_service.dimension
            self.collection_name = settings.vector_db.collection_name

            self.db_path = Path(ROOT_DIR) / "database" / "chroma_db"
            self.db_path.mkdir(parents=True, exist_ok=True)

            self.client = chromadb.PersistentClient(
                path=str(self.db_path),
                settings=Settings(anonymized_telemetry=False, allow_reset=True),
            )

            self.collection = self._get_or_create_collection()

            # TF-IDF hybrid search state.
            # _tfidf_doc_cache is built alongside the matrix so that positional
            # indices from TF-IDF similarity always align with the right document.
            self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
            self.tfidf_matrix = None
            self.document_texts: List[str] = []
            self._tfidf_doc_cache: List[Dict[str, Any]] = []

            # Bug 3 fix: rebuild TF-IDF from existing data on startup so hybrid
            # search is active immediately without requiring index_chunks() first.
            if self.collection.count() > 0:
                self._prepare_hybrid_search()

            logger.info(
                f"VectorDBService initialised with {self.collection.count()} vectors"
            )

        except Exception as e:
            logger.error(f"Failed to initialise VectorDBService: {e}")
            raise

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def _get_or_create_collection(self):
        """Get existing collection or create a new one."""
        try:
            collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"Loaded existing collection: {self.collection_name}")
            return collection
        except Exception:
            logger.info(f"Creating new collection: {self.collection_name}")
            return self.client.create_collection(
                name=self.collection_name, metadata={"hnsw:space": "cosine"}
            )

    def initialize_collection(self, recreate: bool = False) -> None:
        """Initialize or recreate the vector collection."""
        try:
            if recreate:
                try:
                    self.client.delete_collection(name=self.collection_name)
                    logger.info(f"Deleted existing collection: {self.collection_name}")
                except Exception as e:
                    logger.warning(f"No existing collection to delete: {e}")

                self.collection = self.client.create_collection(
                    name=self.collection_name, metadata={"hnsw:space": "cosine"}
                )
                logger.info(f"Created new collection: {self.collection_name}")

            self.tfidf_vectorizer = None
            self.tfidf_matrix = None
            self.document_texts = []
            self._tfidf_doc_cache = []

        except Exception as e:
            logger.error(f"Error initialising collection: {e}")
            raise

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_chunks(self, chunks_file: Optional[Union[str, Path]] = None) -> None:
        """Index chunks and rebuild hybrid search components."""
        try:
            self._index_chunks_standard(chunks_file)
            self._prepare_hybrid_search()
        except Exception as e:
            logger.error(f"Error during indexing: {e}")
            raise

    def _index_chunks_standard(
        self, chunks_file: Optional[Union[str, Path]] = None
    ) -> None:
        chunks_dir = Path(ROOT_DIR) / "data" / "chunks"

        files_to_process = (
            [Path(chunks_file)]
            if chunks_file
            else list(chunks_dir.glob("*_chunks.jsonl"))
        )

        if not files_to_process:
            raise ValueError("No chunk files found to index")

        successful = 0
        for chunk_file in files_to_process:
            try:
                self._index_chunks_file(chunk_file)
                successful += 1
            except Exception as e:
                logger.error(f"Failed to index {chunk_file.name}: {e}")

        logger.info(f"Indexing complete: {successful} files processed")
        logger.info(f"Total vectors in collection: {self.collection.count()}")

    def _index_chunks_file(self, chunks_file: Path) -> None:
        logger.info(f"Indexing chunks from: {chunks_file.name}")

        doc_id = chunks_file.stem.replace("_chunks", "")

        embeddings_dir = Path(ROOT_DIR) / "data" / "embeddings"
        embeddings_file = embeddings_dir / f"{doc_id}_embeddings.npz"

        if not embeddings_file.exists():
            logger.info(f"Generating embeddings for {doc_id}")
            self.embedding_service.embed_chunks(chunks_file)

        embeddings_data = self.embedding_service._load_existing_embeddings(
            embeddings_file
        )
        if not embeddings_data:
            raise ValueError(f"Failed to load embeddings for {doc_id}")

        chunks = []
        with open(chunks_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    chunks.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON line: {e}")

        chunk_lookup = {chunk["chunk_id"]: chunk for chunk in chunks}

        ids, documents, embeddings, metadatas = [], [], [], []

        for chunk_id, embedding in embeddings_data.items():
            if chunk_id in chunk_lookup:
                chunk = chunk_lookup[chunk_id]

                ids.append(str(chunk_id))
                documents.append(chunk["text"])
                embeddings.append(embedding.tolist())

                metadata = {
                    "doc_id": chunk.get("doc_id", doc_id),
                    "chunk_index": chunk.get("chunk_index", 0),
                    "chunk_id": chunk_id,
                    "information_score": chunk.get("information_score", 0.5),
                    "semantic_boundary_score": chunk.get(
                        "semantic_boundary_score", 0.5
                    ),
                }

                if chunk.get("metadata", {}).get("semantic_chunking"):
                    metadata["semantic_chunking"] = True

                metadatas.append(metadata)

        batch_size = 100
        for i in range(0, len(ids), batch_size):
            self.collection.add(
                ids=ids[i: i + batch_size],
                embeddings=embeddings[i: i + batch_size],
                documents=documents[i: i + batch_size],
                metadatas=metadatas[i: i + batch_size],
            )

        logger.info(f"Indexed {len(ids)} chunks from {doc_id}")

    # ------------------------------------------------------------------
    # Hybrid search preparation
    # ------------------------------------------------------------------

    def _prepare_hybrid_search(self) -> None:
        """
        Build TF-IDF matrix and a doc cache aligned with it.

        The cache stores one dict per document in the same order as the matrix
        rows so that a TF-IDF similarity index is always a valid cache index.
        This avoids the re-fetch + positional alignment bug that existed before.
        """
        try:
            all_data = self.collection.get()

            if not all_data["documents"]:
                logger.warning("No documents in collection — hybrid search disabled")
                return

            self.document_texts = all_data["documents"]

            # Bug 2 & 5 fix: build the doc cache from the SAME fetch and in the
            # SAME order as the matrix so indices always correspond correctly.
            self._tfidf_doc_cache = [
                {
                    "chunk_id": m.get("chunk_id", ""),
                    "doc_id": m.get("doc_id", ""),
                    "chunk_index": m.get("chunk_index", 0),
                    "information_score": m.get("information_score", 0.5),
                    "semantic_boundary_score": m.get("semantic_boundary_score", 0.5),
                    "text": doc,
                }
                for m, doc in zip(all_data["metadatas"], all_data["documents"])
            ]

            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words="english",
                ngram_range=(1, 2),
                max_df=0.95,
                min_df=2,
            )
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(
                self.document_texts
            )

            logger.info(
                f"Hybrid search ready with {len(self.document_texts)} documents"
            )

        except Exception as e:
            logger.error(f"Failed to prepare hybrid search: {e}")
            self.tfidf_vectorizer = None
            self.tfidf_matrix = None
            self._tfidf_doc_cache = []

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 20,
        filter_doc_id: Optional[str] = None,
        use_hybrid: bool = True,
        include_real_time: bool = False,  # kept for API backward-compat; not used here
    ) -> List[Dict[str, Any]]:
        """
        Hybrid semantic + keyword search.

        Real-time retrieval is intentionally excluded from this layer.
        ResponseGenerator handles Tavily calls and merges results before
        passing context to the LLM, preventing duplicate web search calls.
        """
        try:
            if self.collection.count() == 0:
                logger.warning("Collection is empty")
                return []

            semantic_results = self._semantic_search(query, top_k, filter_doc_id)

            if use_hybrid and self.tfidf_vectorizer is not None:
                keyword_results = self._keyword_search(query, top_k // 2)
                combined = self._combine_search_results(semantic_results, keyword_results)
            else:
                combined = semantic_results

            if combined:
                combined = self._rerank_results(query, combined, top_k)

            return combined[:top_k]

        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

    def multi_query_search(
        self,
        query: str,
        top_k: int = 20,
        filter_doc_id: Optional[str] = None,
        use_query_expansion: bool = True,
    ) -> List[Dict[str, Any]]:
        """Multi-query search with optional query expansion."""
        try:
            original_results = self.search(
                query=query,
                top_k=int(top_k * 0.6),
                filter_doc_id=filter_doc_id,
                use_hybrid=True,
            )

            if not use_query_expansion:
                return original_results[:top_k]

            expanded_queries = self._generate_expanded_queries(query)
            all_results = original_results.copy()
            seen_chunks = {r["chunk_id"] for r in original_results}

            for expanded_query in expanded_queries:
                if len(all_results) >= top_k:
                    break

                try:
                    expanded_results = self.search(
                        query=expanded_query,
                        top_k=5,
                        filter_doc_id=filter_doc_id,
                        use_hybrid=True,
                    )

                    for result in expanded_results:
                        if result["chunk_id"] not in seen_chunks:
                            result["score"] *= 0.9
                            result["expanded_query"] = True
                            all_results.append(result)
                            seen_chunks.add(result["chunk_id"])

                except Exception as e:
                    logger.warning(f"Expanded query failed: {e}")
                    continue

            if len(all_results) > top_k:
                all_results = self._rerank_results(query, all_results, top_k)

            all_results.sort(
                key=lambda x: x.get("enhanced_score", x.get("score", 0)),
                reverse=True,
            )

            return all_results[:top_k]

        except Exception as e:
            logger.error(f"Multi-query search error: {e}")
            return self.search(query, top_k, filter_doc_id)

    # ------------------------------------------------------------------
    # Search internals
    # ------------------------------------------------------------------

    def _semantic_search(
        self, query: str, top_k: int, filter_doc_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Semantic search using dense embeddings."""
        query_embedding = self.embedding_service.embed_query(query)
        if query_embedding is None:
            return []

        where_filter = {"doc_id": filter_doc_id} if filter_doc_id else None

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=where_filter,
        )

        formatted = []
        if results and results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                metadata = results["metadatas"][0][i]

                # Bug 8 fix: ChromaDB cosine distance is in [0, 2]; converting
                # with (1 - distance) can yield negative values for dissimilar
                # pairs. Clamp to [0, 1] for consistent downstream arithmetic.
                raw_score = 1.0 - results["distances"][0][i]
                score = max(0.0, min(1.0, raw_score))

                formatted.append({
                    "chunk_id": metadata.get("chunk_id", ""),
                    "doc_id": metadata.get("doc_id", ""),
                    "chunk_index": metadata.get("chunk_index", 0),
                    "text": results["documents"][0][i],
                    "score": score,
                    "search_type": "semantic",
                    "information_score": metadata.get("information_score", 0.5),
                    "semantic_boundary_score": metadata.get(
                        "semantic_boundary_score", 0.5
                    ),
                })

        return formatted

    def _keyword_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Keyword search using the TF-IDF matrix built at prepare time.

        Bug 2 & 5 fix: uses self._tfidf_doc_cache instead of re-fetching
        from ChromaDB. The cache and the matrix share the same row order so
        positional indices are always correct and no extra I/O is needed.
        """
        if self.tfidf_vectorizer is None or self.tfidf_matrix is None:
            return []

        try:
            query_vector = self.tfidf_vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            top_indices = similarities.argsort()[-top_k:][::-1]

            results = []
            for idx in top_indices:
                if similarities[idx] <= 0.1:
                    continue
                if idx >= len(self._tfidf_doc_cache):
                    continue

                cached = self._tfidf_doc_cache[idx]
                results.append({
                    "chunk_id": cached["chunk_id"],
                    "doc_id": cached["doc_id"],
                    "chunk_index": cached["chunk_index"],
                    "text": cached["text"],
                    "score": float(similarities[idx]),
                    "search_type": "keyword",
                    "information_score": cached["information_score"],
                    "semantic_boundary_score": cached["semantic_boundary_score"],
                })

            return results

        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            return []

    def _combine_search_results(
        self, semantic_results: List[Dict], keyword_results: List[Dict]
    ) -> List[Dict]:
        """
        Merge semantic and keyword results.

        Chunks that appear in both lists get a weighted score boost.
        Keyword-only results are added with a small penalty since they
        lack the semantic relevance signal.
        """
        semantic_lookup = {r["chunk_id"]: r for r in semantic_results}
        combined = [dict(r) for r in semantic_results]  # work on copies

        for kw_result in keyword_results:
            chunk_id = kw_result["chunk_id"]

            if chunk_id in semantic_lookup:
                # Find the copy in combined and boost it
                for r in combined:
                    if r["chunk_id"] == chunk_id:
                        r["score"] = 0.7 * r["score"] + 0.3 * kw_result["score"]
                        r["search_type"] = "hybrid"
                        break
            else:
                penalised = dict(kw_result)
                penalised["score"] *= 0.8
                combined.append(penalised)

        combined.sort(key=lambda x: x["score"], reverse=True)
        return combined

    def _rerank_results(
        self, query: str, results: List[Dict], top_k: int
    ) -> List[Dict]:
        """
        Rerank by combining similarity score with chunk quality signals.

        Bug 4 fix:
        - Weights now sum to exactly 1.0 (no stray constant).
        - Real-time boost removed from this layer; Tavily results are
          integrated at the ResponseGenerator level with explicit ordering.
        - Scores are clamped before weighting so the formula stays in [0, 1].
        """
        if not results or len(results) <= 1:
            return results[:top_k]

        try:
            for result in results:
                original_score = max(0.0, min(1.0, result.get("score", 0.0)))

                # Mild quality signals from the chunking pipeline
                text_length = len(result.get("text", ""))
                length_penalty = min(1.0, text_length / 200)
                info_score = max(0.0, min(1.0, result.get("information_score", 0.5)))
                boundary_score = max(0.0, min(1.0, result.get("semantic_boundary_score", 0.5)))

                # Weights: 0.60 + 0.15 + 0.15 + 0.10 = 1.00
                result["enhanced_score"] = (
                    0.60 * original_score
                    + 0.15 * length_penalty
                    + 0.15 * info_score
                    + 0.10 * boundary_score
                )
                result["reranked"] = True

            results.sort(
                key=lambda x: x.get("enhanced_score", x.get("score", 0)),
                reverse=True,
            )
            return results[:top_k]

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return results[:top_k]

    def _generate_expanded_queries(self, query: str) -> List[str]:
        """Generate domain-specific query expansions for better recall."""
        query_lower = query.lower()
        expanded = []

        if "strathmore" not in query_lower:
            expanded.append(f"{query} Strathmore University")

        if any(w in query_lower for w in ["fee", "cost", "payment", "tuition"]):
            expanded.extend([
                f"{query} payment schedule deadline",
                f"{query} scholarship financial aid",
            ])
        elif any(w in query_lower for w in ["admission", "apply", "entry"]):
            expanded.extend([
                f"{query} requirements qualifications",
                f"{query} application process procedure",
            ])
        elif any(w in query_lower for w in ["class", "schedule", "timetable"]):
            expanded.extend([
                f"{query} semester academic calendar",
                f"{query} lecture tutorial lab",
            ])
        else:
            expanded.extend([
                f"{query} policy procedure",
                f"{query} student guide information",
            ])

        return expanded[:3]

    # ------------------------------------------------------------------
    # Similarity lookup
    # ------------------------------------------------------------------

    def get_similar_chunks(
        self, chunk_id: str, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find chunks similar to a given chunk by embedding similarity.

        Bug 1 fix: include=["embeddings"] must be passed to .get() because
        ChromaDB does not return embeddings by default.
        """
        try:
            source_data = self.collection.get(
                ids=[chunk_id], include=["embeddings", "documents", "metadatas"]
            )

            if not source_data["embeddings"] or not source_data["embeddings"][0]:
                logger.warning(f"Chunk {chunk_id} not found or has no embedding")
                return []

            source_embedding = source_data["embeddings"][0]

            results = self.collection.query(
                query_embeddings=[source_embedding], n_results=top_k + 1
            )

            similar = []
            if results and results["ids"] and results["ids"][0]:
                for i, result_id in enumerate(results["ids"][0]):
                    if result_id == chunk_id:
                        continue
                    metadata = results["metadatas"][0][i]
                    raw_score = 1.0 - results["distances"][0][i]
                    similar.append({
                        "chunk_id": result_id,
                        "doc_id": metadata.get("doc_id", ""),
                        "text": results["documents"][0][i],
                        "similarity": max(0.0, min(1.0, raw_score)),
                        "information_score": metadata.get("information_score", 0.5),
                    })

            return similar[:top_k]

        except Exception as e:
            logger.error(f"Error finding similar chunks: {e}")
            return []

    # ------------------------------------------------------------------
    # Document management
    # ------------------------------------------------------------------

    def delete_document(self, doc_id: str) -> bool:
        """Delete all chunks for a document and rebuild hybrid search index."""
        try:
            results = self.collection.get(where={"doc_id": doc_id})

            if not results["ids"]:
                logger.warning(f"No chunks found for document {doc_id}")
                return False

            self.collection.delete(ids=results["ids"])
            self._prepare_hybrid_search()
            logger.info(f"Deleted {len(results['ids'])} chunks for document {doc_id}")
            return True

        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {e}")
            return False

    def optimize_collection(self) -> Dict[str, Any]:
        """Rebuild hybrid search index and return before/after stats."""
        try:
            initial_stats = self.get_collection_stats()
            self._prepare_hybrid_search()
            final_stats = self.get_collection_stats()

            return {
                "initial_count": initial_stats.get("count", 0),
                "final_count": final_stats.get("count", 0),
                "hybrid_search_rebuilt": True,
                "optimization_timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Collection optimization failed: {e}")
            return {"error": str(e)}

    # ------------------------------------------------------------------
    # Stats and diagnostics
    # ------------------------------------------------------------------

    def get_collection_stats(self) -> Dict[str, Any]:
        """Return comprehensive collection statistics."""
        try:
            count = self.collection.count()

            stats = {
                "name": self.collection_name,
                "count": count,
                "dimension": self.dimension,
                "hybrid_search_enabled": self.tfidf_vectorizer is not None,
                "real_time_integration": self.tavily_service is not None,
            }

            if count > 0:
                sample_data = self.collection.get(limit=min(100, count))

                if sample_data["metadatas"]:
                    info_scores = [
                        m.get("information_score", 0)
                        for m in sample_data["metadatas"]
                    ]
                    stats["avg_information_score"] = sum(info_scores) / len(info_scores)

                    semantic_count = sum(
                        1 for m in sample_data["metadatas"] if m.get("semantic_chunking")
                    )
                    stats["semantic_chunked_ratio"] = semantic_count / len(
                        sample_data["metadatas"]
                    )

            return stats

        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"error": str(e)}

    def get_service_health(self) -> Dict[str, Any]:
        """Return comprehensive service health information."""
        health: Dict[str, Any] = {
            "vector_db_status": "healthy",
            "collection_count": 0,
            "embedding_service_status": "unknown",
            "tavily_service_status": "unknown",
            "hybrid_search_ready": False,
            "real_time_integration_ready": False,
            "errors": [],
        }

        try:
            health["collection_count"] = self.collection.count()

            if self.embedding_service:
                try:
                    test_embedding = self.embedding_service.embed_query("test")
                    health["embedding_service_status"] = (
                        "healthy" if test_embedding is not None else "error"
                    )
                except Exception as e:
                    health["embedding_service_status"] = f"error: {str(e)}"
                    health["errors"].append(f"Embedding service: {str(e)}")

            if self.tavily_service:
                try:
                    self.tavily_service.get_cache_stats()
                    health["tavily_service_status"] = "healthy"
                    health["real_time_integration_ready"] = True
                except Exception as e:
                    health["tavily_service_status"] = f"error: {str(e)}"
                    health["errors"].append(f"Tavily service: {str(e)}")
            else:
                health["tavily_service_status"] = "not_configured"

            health["hybrid_search_ready"] = (
                self.tfidf_vectorizer is not None and self.tfidf_matrix is not None
            )

            if health["errors"]:
                health["vector_db_status"] = "degraded"

        except Exception as e:
            health["vector_db_status"] = "error"
            health["errors"].append(f"Vector DB: {str(e)}")

        return health

    def test_real_time_integration(self, query: str) -> Dict[str, Any]:
        """Diagnostic: test the Tavily real-time service directly."""
        if not self.tavily_service:
            return {"error": "Tavily service not available", "real_time_active": False}

        try:
            enhanced_query = query
            if "strathmore" not in query.lower():
                enhanced_query = f"Strathmore University {query}"

            result = self.tavily_service.search(
                query=enhanced_query,
                max_results=3,
                search_depth="basic",
                include_answer=True,
            )

            return {
                "query": query,
                "enhanced_query": enhanced_query,
                "real_time_active": True,
                "real_time_results_count": len(result.get("results", [])),
                "real_time_results": result.get("results", [])[:3],
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            return {"error": str(e), "real_time_active": False}
