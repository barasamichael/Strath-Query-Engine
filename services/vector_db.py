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
    """Optimized vector database service with hybrid search and aggressive real-time integration."""

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

            # TF-IDF for hybrid search
            self.tfidf_vectorizer = None
            self.tfidf_matrix = None
            self.document_texts = []

            # Real-time integration settings
            self.real_time_triggers = [
                "current",
                "latest",
                "recent",
                "today",
                "now",
                "this year",
                "2024",
                "2025",
                "deadline",
                "announcement",
                "news",
                "update",
                "fee",
                "tuition",
                "cost",
                "admission",
                "registration",
            ]

            logger.info(
                f"VectorDBService initialized with {self.collection.count()} vectors"
            )

        except Exception as e:
            logger.error(f"Failed to initialize VectorDBService: {e}")
            raise

    def _get_or_create_collection(self):
        """Get existing collection or create new one."""
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
                    logger.info(
                        f"Deleted existing collection: {self.collection_name}"
                    )
                except Exception as e:
                    logger.warning(f"No existing collection to delete: {e}")

                self.collection = self.client.create_collection(
                    name=self.collection_name, metadata={"hnsw:space": "cosine"}
                )
                logger.info(f"Created new collection: {self.collection_name}")

            self.tfidf_vectorizer = None
            self.tfidf_matrix = None
            self.document_texts = []

        except Exception as e:
            logger.error(f"Error initializing collection: {e}")
            raise

    def index_chunks(
        self, chunks_file: Optional[Union[str, Path]] = None
    ) -> None:
        """Index chunks with hybrid search preparation."""
        try:
            self._index_chunks_standard(chunks_file)
            self._prepare_hybrid_search()
        except Exception as e:
            logger.error(f"Error during indexing: {e}")
            raise

    def _index_chunks_standard(
        self, chunks_file: Optional[Union[str, Path]] = None
    ) -> None:
        """Standard chunk indexing."""
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
        """Index chunks from a single file."""
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

        ids = []
        documents = []
        embeddings = []
        metadatas = []

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

        # Batch insert
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i : i + batch_size]
            batch_embeddings = embeddings[i : i + batch_size]
            batch_documents = documents[i : i + batch_size]
            batch_metadatas = metadatas[i : i + batch_size]

            self.collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                documents=batch_documents,
                metadatas=batch_metadatas,
            )

        logger.info(f"Successfully indexed {len(ids)} chunks from {doc_id}")

    def _prepare_hybrid_search(self) -> None:
        """Prepare TF-IDF components for hybrid search."""
        try:
            all_data = self.collection.get()

            if not all_data["documents"]:
                logger.warning(
                    "No documents found for hybrid search preparation"
                )
                return

            self.document_texts = all_data["documents"]

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
                f"Hybrid search prepared with {len(self.document_texts)} documents"
            )

        except Exception as e:
            logger.error(f"Failed to prepare hybrid search: {e}")
            self.tfidf_vectorizer = None
            self.tfidf_matrix = None

    def search(
        self,
        query: str,
        top_k: int = 20,
        filter_doc_id: Optional[str] = None,
        use_hybrid: bool = True,
        include_real_time: bool = False,
    ) -> List[Dict[str, Any]]:
        """Enhanced search with hybrid approach and aggressive real-time integration."""
        try:
            if self.collection.count() == 0:
                logger.warning("Collection is empty")
                return []

            # Check for real-time needs FIRST
            needs_real_time = self._should_use_real_time(query)
            if needs_real_time or include_real_time:
                real_time_results = self._get_aggressive_real_time_info(query)
            else:
                real_time_results = []

            # Get semantic search results
            semantic_results = self._semantic_search(
                query, top_k, filter_doc_id
            )

            # Add keyword search if hybrid enabled
            if use_hybrid and self.tfidf_vectorizer is not None:
                keyword_results = self._keyword_search(query, top_k // 2)
                combined_results = self._combine_search_results(
                    semantic_results, keyword_results
                )
            else:
                combined_results = semantic_results

            # Apply reranking
            if combined_results:
                combined_results = self._rerank_results(
                    query, combined_results, top_k
                )

            # Integrate real-time results
            if real_time_results:
                combined_results = self._integrate_real_time_results(
                    combined_results, real_time_results
                )

            return combined_results[:top_k]

        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

    def _should_use_real_time(self, query: str) -> bool:
        """Determine if query needs real-time information aggressively."""
        query_lower = query.lower()

        # Check for explicit triggers
        if any(trigger in query_lower for trigger in self.real_time_triggers):
            return True

        # Check for time-sensitive topics
        time_sensitive = [
            "fee",
            "admission",
            "schedule",
            "deadline",
            "announcement",
        ]
        if any(topic in query_lower for topic in time_sensitive):
            return True

        return False

    def _get_aggressive_real_time_info(
        self, query: str
    ) -> List[Dict[str, Any]]:
        """Get real-time information aggressively using Tavily."""
        if not self.tavily_service:
            return []

        try:
            # Enhance query for university context
            enhanced_query = self._enhance_query_for_tavily(query)

            # Search with multiple strategies
            strategies = [
                enhanced_query,
                f"Strathmore University {query} {datetime.now().year}",
                f"{query} Strathmore current information",
            ]

            all_results = []
            for strategy in strategies[:2]:  # Limit for cost control
                try:
                    result = self.tavily_service.search(
                        query=strategy,
                        max_results=2,
                        search_depth="basic",
                        include_answer=True,
                    )
                    if result.get("results"):
                        all_results.extend(result["results"])
                except Exception as e:
                    logger.warning(
                        f"Tavily search failed for '{strategy}': {e}"
                    )
                    continue

            # Deduplicate and format
            unique_results = {}
            for result in all_results:
                url = result.get("url", "")
                if url not in unique_results:
                    # Convert to our format
                    formatted_result = {
                        "chunk_id": f"realtime_{hash(url)}",
                        "doc_id": "real_time",
                        "chunk_index": 0,
                        "text": f"CURRENT: {result.get('title', '')} - {result.get('content', '')}",
                        "score": 0.95,
                        "search_type": "real_time",
                        "information_score": 1.0,
                        "semantic_boundary_score": 1.0,
                        "url": url,
                        "real_time": True,
                        "relevance_score": result.get("relevance_score", 0.8),
                        "source": "Tavily Real-time",
                    }
                    unique_results[url] = formatted_result

            return list(unique_results.values())

        except Exception as e:
            logger.error(f"Real-time search failed: {e}")
            return []

    def _enhance_query_for_tavily(self, query: str) -> str:
        """Enhance query for better Tavily results."""
        query_lower = query.lower()

        if "strathmore" not in query_lower:
            query = f"Strathmore University {query}"

        current_year = str(datetime.now().year)
        if current_year not in query:
            query = f"{query} {current_year}"

        return query

    def _semantic_search(
        self, query: str, top_k: int, filter_doc_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Perform semantic search using embeddings."""
        query_embedding = self.embedding_service.embed_query(query)
        if query_embedding is None:
            return []

        where_filter = {"doc_id": filter_doc_id} if filter_doc_id else None

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=where_filter,
        )

        formatted_results = []
        if results and results["ids"] and len(results["ids"]) > 0:
            for i in range(len(results["ids"][0])):
                metadata = results["metadatas"][0][i]

                result = {
                    "chunk_id": metadata.get("chunk_id", ""),
                    "doc_id": metadata.get("doc_id", ""),
                    "chunk_index": metadata.get("chunk_index", 0),
                    "text": results["documents"][0][i],
                    "score": 1 - results["distances"][0][i],
                    "search_type": "semantic",
                    "information_score": metadata.get("information_score", 0.5),
                    "semantic_boundary_score": metadata.get(
                        "semantic_boundary_score", 0.5
                    ),
                }

                formatted_results.append(result)

        return formatted_results

    def _keyword_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Perform keyword search using TF-IDF."""
        if self.tfidf_vectorizer is None or self.tfidf_matrix is None:
            return []

        try:
            query_vector = self.tfidf_vectorizer.transform([query])
            similarities = cosine_similarity(
                query_vector, self.tfidf_matrix
            ).flatten()
            top_indices = similarities.argsort()[-top_k:][::-1]

            all_data = self.collection.get()

            keyword_results = []
            for idx in top_indices:
                if similarities[idx] > 0.1:
                    metadata = (
                        all_data["metadatas"][idx]
                        if idx < len(all_data["metadatas"])
                        else {}
                    )

                    result = {
                        "chunk_id": metadata.get("chunk_id", ""),
                        "doc_id": metadata.get("doc_id", ""),
                        "chunk_index": metadata.get("chunk_index", 0),
                        "text": (
                            all_data["documents"][idx]
                            if idx < len(all_data["documents"])
                            else ""
                        ),
                        "score": float(similarities[idx]),
                        "search_type": "keyword",
                        "information_score": metadata.get(
                            "information_score", 0.5
                        ),
                        "semantic_boundary_score": metadata.get(
                            "semantic_boundary_score", 0.5
                        ),
                    }

                    keyword_results.append(result)

            return keyword_results

        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            return []

    def _combine_search_results(
        self, semantic_results: List[Dict], keyword_results: List[Dict]
    ) -> List[Dict]:
        """Combine semantic and keyword search results intelligently."""
        semantic_lookup = {
            result["chunk_id"]: result for result in semantic_results
        }
        combined_results = list(semantic_results)

        for keyword_result in keyword_results:
            chunk_id = keyword_result["chunk_id"]

            if chunk_id in semantic_lookup:
                # Boost score of existing result
                semantic_result = semantic_lookup[chunk_id]
                semantic_result["score"] = (
                    0.7 * semantic_result["score"]
                    + 0.3 * keyword_result["score"]
                )
                semantic_result["search_type"] = "hybrid"
            else:
                # Add new result with slight penalty
                keyword_result["score"] *= 0.8
                combined_results.append(keyword_result)

        combined_results.sort(key=lambda x: x["score"], reverse=True)
        return combined_results

    def _rerank_results(
        self, query: str, results: List[Dict], top_k: int
    ) -> List[Dict]:
        """Rerank search results using multiple factors."""
        if not results or len(results) <= 1:
            return results[:top_k]

        try:
            for result in results:
                original_score = result.get("score", 0.0)

                # Length penalty for very short chunks
                text_length = len(result.get("text", ""))
                length_penalty = min(1.0, text_length / 200)

                # Information score boost
                info_score = result.get("information_score", 0.5)

                # Semantic boundary score boost
                boundary_score = result.get("semantic_boundary_score", 0.5)

                # Real-time boost
                real_time_boost = 0.2 if result.get("real_time") else 0.0

                # Calculate enhanced score
                enhanced_score = (
                    0.5 * original_score
                    + 0.15 * length_penalty
                    + 0.15 * info_score
                    + 0.1 * boundary_score
                    + 0.1
                    + real_time_boost
                )

                result["enhanced_score"] = enhanced_score
                result["reranked"] = True

            # Sort by enhanced score
            results.sort(
                key=lambda x: x.get("enhanced_score", x.get("score", 0)),
                reverse=True,
            )

            logger.info(
                f"Reranked {len(results)} results, returning top {top_k}"
            )
            return results[:top_k]

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return results[:top_k]

    def _integrate_real_time_results(
        self, search_results: List[Dict], real_time_results: List[Dict]
    ) -> List[Dict]:
        """Integrate real-time results with search results strategically."""
        if not real_time_results:
            return search_results

        # Insert real-time results at the top, but merge intelligently
        integrated_results = []

        # Add top real-time results first
        integrated_results.extend(real_time_results[:2])

        # Add top search results
        integrated_results.extend(search_results[:15])

        # Add remaining real-time results
        if len(real_time_results) > 2:
            integrated_results.extend(real_time_results[2:])

        return integrated_results

    def multi_query_search(
        self,
        query: str,
        top_k: int = 20,
        filter_doc_id: Optional[str] = None,
        use_query_expansion: bool = True,
    ) -> List[Dict[str, Any]]:
        """Multi-query search with query expansion and enhanced retrieval."""
        try:
            # Original query results with aggressive real-time
            original_results = self.search(
                query=query,
                top_k=int(top_k * 0.6),
                filter_doc_id=filter_doc_id,
                use_hybrid=True,
                include_real_time=True,
            )

            if not use_query_expansion:
                return original_results[:top_k]

            # Generate expanded queries
            expanded_queries = self._generate_expanded_queries(query)
            all_results = original_results.copy()
            seen_chunks = {r["chunk_id"] for r in original_results}

            # Search with expanded queries
            for expanded_query in expanded_queries:
                if len(all_results) >= top_k:
                    break

                try:
                    expanded_results = self.search(
                        query=expanded_query,
                        top_k=5,
                        filter_doc_id=filter_doc_id,
                        use_hybrid=True,
                        include_real_time=False,  # Avoid duplicate real-time calls
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

            # Final reranking
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

    def _generate_expanded_queries(self, query: str) -> List[str]:
        """Generate expanded queries for better retrieval."""
        query_lower = query.lower()
        expanded_queries = []

        if "strathmore" not in query_lower:
            expanded_queries.append(f"{query} Strathmore University")

        # Domain-specific expansions
        if any(
            word in query_lower
            for word in ["fee", "cost", "payment", "tuition"]
        ):
            expanded_queries.extend(
                [
                    f"{query} payment schedule deadline",
                    f"{query} scholarship financial aid",
                ]
            )
        elif any(
            word in query_lower for word in ["admission", "apply", "entry"]
        ):
            expanded_queries.extend(
                [
                    f"{query} requirements qualifications",
                    f"{query} application process procedure",
                ]
            )
        elif any(
            word in query_lower for word in ["class", "schedule", "timetable"]
        ):
            expanded_queries.extend(
                [
                    f"{query} semester academic calendar",
                    f"{query} lecture tutorial lab",
                ]
            )
        else:
            expanded_queries.extend(
                [
                    f"{query} policy procedure",
                    f"{query} student guide information",
                ]
            )

        return expanded_queries[:3]

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get comprehensive collection statistics."""
        try:
            count = self.collection.count()

            stats = {
                "name": self.collection_name,
                "count": count,
                "dimension": self.dimension,
                "hybrid_search_enabled": self.tfidf_vectorizer is not None,
                "real_time_integration": self.tavily_service is not None,
                "real_time_triggers": len(self.real_time_triggers),
            }

            if count > 0:
                sample_data = self.collection.get(limit=min(100, count))

                if sample_data["metadatas"]:
                    info_scores = [
                        meta.get("information_score", 0)
                        for meta in sample_data["metadatas"]
                    ]
                    if info_scores:
                        stats["avg_information_score"] = sum(info_scores) / len(
                            info_scores
                        )

                    semantic_count = sum(
                        1
                        for meta in sample_data["metadatas"]
                        if meta.get("semantic_chunking")
                    )
                    stats["semantic_chunked_ratio"] = semantic_count / len(
                        sample_data["metadatas"]
                    )

            return stats

        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"error": str(e)}

    def get_similar_chunks(
        self, chunk_id: str, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Find chunks similar to a given chunk."""
        try:
            source_data = self.collection.get(ids=[chunk_id])

            if not source_data["embeddings"]:
                logger.warning(f"Chunk {chunk_id} not found")
                return []

            source_embedding = source_data["embeddings"][0]

            results = self.collection.query(
                query_embeddings=[source_embedding], n_results=top_k + 1
            )

            similar_chunks = []
            if results and results["ids"] and len(results["ids"]) > 0:
                for i, result_id in enumerate(results["ids"][0]):
                    if result_id != chunk_id:
                        metadata = results["metadatas"][0][i]

                        chunk = {
                            "chunk_id": result_id,
                            "doc_id": metadata.get("doc_id", ""),
                            "text": results["documents"][0][i],
                            "similarity": 1 - results["distances"][0][i],
                            "information_score": metadata.get(
                                "information_score", 0.5
                            ),
                        }
                        similar_chunks.append(chunk)

            return similar_chunks[:top_k]

        except Exception as e:
            logger.error(f"Error finding similar chunks: {e}")
            return []

    def delete_document(self, doc_id: str) -> bool:
        """Delete all chunks for a specific document."""
        try:
            results = self.collection.get(where={"doc_id": doc_id})

            if results["ids"]:
                self.collection.delete(ids=results["ids"])
                self._prepare_hybrid_search()

                logger.info(
                    f"Deleted {len(results['ids'])} chunks for document {doc_id}"
                )
                return True
            else:
                logger.warning(f"No chunks found for document {doc_id}")
                return False

        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {e}")
            return False

    def optimize_collection(self) -> Dict[str, Any]:
        """Optimize collection performance."""
        try:
            initial_stats = self.get_collection_stats()
            self._prepare_hybrid_search()
            final_stats = self.get_collection_stats()

            optimization_report = {
                "initial_count": initial_stats.get("count", 0),
                "final_count": final_stats.get("count", 0),
                "hybrid_search_rebuilt": True,
                "real_time_integration_active": self.tavily_service is not None,
                "optimization_timestamp": datetime.now().isoformat(),
            }

            logger.info("Collection optimization completed")
            return optimization_report

        except Exception as e:
            logger.error(f"Collection optimization failed: {e}")
            return {"error": str(e)}

    def test_real_time_integration(self, query: str) -> Dict[str, Any]:
        """Test real-time integration specifically."""
        if not self.tavily_service:
            return {
                "error": "Tavily service not available",
                "real_time_active": False,
            }

        try:
            real_time_results = self._get_aggressive_real_time_info(query)

            return {
                "query": query,
                "real_time_active": True,
                "real_time_results_count": len(real_time_results),
                "real_time_results": real_time_results[:3],  # Sample
                "should_use_real_time": self._should_use_real_time(query),
                "enhanced_query": self._enhance_query_for_tavily(query),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            return {"error": str(e), "real_time_active": False}

    def get_service_health(self) -> Dict[str, Any]:
        """Get comprehensive service health information."""
        health = {
            "vector_db_status": "healthy",
            "collection_count": 0,
            "embedding_service_status": "unknown",
            "tavily_service_status": "unknown",
            "hybrid_search_ready": False,
            "real_time_integration_ready": False,
            "errors": [],
        }

        try:
            # Check vector database
            health["collection_count"] = self.collection.count()

            # Check embedding service
            if self.embedding_service:
                try:
                    test_embedding = self.embedding_service.embed_query("test")
                    health["embedding_service_status"] = (
                        "healthy" if test_embedding is not None else "error"
                    )
                except Exception as e:
                    health["embedding_service_status"] = f"error: {str(e)}"
                    health["errors"].append(f"Embedding service: {str(e)}")

            # Check Tavily service
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

            # Check hybrid search
            health["hybrid_search_ready"] = (
                self.tfidf_vectorizer is not None
                and self.tfidf_matrix is not None
            )

            if health["errors"]:
                health["vector_db_status"] = "degraded"

        except Exception as e:
            health["vector_db_status"] = "error"
            health["errors"].append(f"Vector DB: {str(e)}")

        return health
