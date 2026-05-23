import os
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Union, Optional, Tuple, Any
from enum import Enum

from tqdm import tqdm
from openai import OpenAI

from config.settings import ROOT_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("embeddings")


class IntentType(str, Enum):
    """Intent types for query classification."""

    FACTUAL_QUERY = "factual_query"
    PROCEDURAL_QUERY = "procedural_query"
    EXPLANATION_QUERY = "explanation_query"
    COMPARISON_QUERY = "comparison_query"
    SCHEDULE_QUERY = "schedule_query"
    NAVIGATION_QUERY = "navigation_query"
    FEES_QUERY = "fees_query"
    ADMISSION_QUERY = "admission_query"
    GENERAL_CHAT = "general_chat"
    OFF_TOPIC = "off_topic"


class EmbeddingService:
    """Optimized embedding service focused on core functionality."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        chunks_dir: Optional[Union[str, Path]] = None,
        embeddings_dir: Optional[Union[str, Path]] = None,
    ):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name or "text-embedding-3-large"
        self.dimension = 3072 if "large" in self.model_name else 1536

        self.chunks_dir = (
            Path(chunks_dir) if chunks_dir else ROOT_DIR / "data" / "chunks"
        )
        self.embeddings_dir = (
            Path(embeddings_dir)
            if embeddings_dir
            else ROOT_DIR / "data" / "embeddings"
        )

        for dir_path in [self.chunks_dir, self.embeddings_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        self.metadata_file = self.embeddings_dir / "embeddings_metadata.json"
        self.embeddings_metadata = self._load_metadata()

        self.intent_templates = self._initialize_intent_templates()
        self.intent_embeddings = None

        logger.info(f"EmbeddingService initialized with {self.model_name}")

    def rerank_results(
        self, query: str, results: List[Dict], top_k: int = None
    ) -> List[Dict]:
        """Rerank search results based on query relevance."""
        if not results:
            return results

        try:
            # Simple reranking based on text similarity with query
            query_lower = query.lower()

            for result in results:
                text = result.get("text", "").lower()

                # Calculate simple relevance score
                query_words = set(query_lower.split())
                text_words = set(text.split())

                if text_words:
                    intersection = len(query_words.intersection(text_words))
                    union = len(query_words.union(text_words))
                    jaccard_similarity = (
                        intersection / union if union > 0 else 0
                    )
                else:
                    jaccard_similarity = 0

                # Boost existing score with jaccard similarity
                original_score = result.get("score", 0.0)
                result["rerank_score"] = (
                    0.7 * original_score + 0.3 * jaccard_similarity
                )

            # Sort by rerank score
            results.sort(
                key=lambda x: x.get("rerank_score", x.get("score", 0)),
                reverse=True,
            )

            return results[:top_k] if top_k else results

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return results

    def _initialize_intent_templates(self) -> Dict[IntentType, List[str]]:
        """Initialize intent recognition templates."""

        return {
            IntentType.FACTUAL_QUERY: [
                "What is the admission requirement for computer science?",
                "Who is the dean of the business school?",
                "When does the semester start?",
                "What are the library opening hours?",
                "How many students are enrolled this year?",
            ],
            IntentType.PROCEDURAL_QUERY: [
                "How do I apply for admission?",
                "What is the process for registering for courses?",
                "How can I pay my fees?",
                "What steps should I follow to get a transcript?",
                "How do I book accommodation in the hostel?",
            ],
            IntentType.EXPLANATION_QUERY: [
                "Why is attendance mandatory for all classes?",
                "Explain the grading system at Strathmore",
                "What does the academic probation policy mean?",
                "Why do I need to maintain a minimum GPA?",
                "What happens if I fail a course?",
            ],
            IntentType.SCHEDULE_QUERY: [
                "What classes do I have today?",
                "When is my next lecture?",
                "Show me the timetable for BICS 1A",
                "What time does the database class start?",
                "Do we have classes on Friday?",
            ],
            IntentType.FEES_QUERY: [
                "How much are the tuition fees?",
                "What is the cost of accommodation?",
                "Are there any scholarships available?",
                "When is the fee payment deadline?",
                "Can I pay fees in installments?",
            ],
            IntentType.ADMISSION_QUERY: [
                "What are the entry requirements?",
                "How do I get admission to Strathmore?",
                "What documents do I need for application?",
                "When does admission start?",
                "What is the minimum grade for engineering?",
            ],
            IntentType.NAVIGATION_QUERY: [
                "Where is the library located?",
                "How do I get to the computer lab?",
                "Where can I find the registrar's office?",
                "Which building is the cafeteria in?",
                "Where do I go for academic counseling?",
            ],
            IntentType.GENERAL_CHAT: [
                "Hello, how are you?",
                "Thank you for your help",
                "Good morning",
                "That's very helpful",
                "I appreciate your assistance",
            ],
            IntentType.OFF_TOPIC: [
                "What's the weather like today?",
                "Who won the World Cup?",
                "Tell me about cryptocurrency",
                "What's the latest news?",
                "How do I cook pasta?",
            ],
        }

    def _load_metadata(self) -> Dict:
        """Load embeddings metadata cache."""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, "r") as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load metadata: {e}")
        return {}

    def _save_metadata(self):
        """Save embeddings metadata cache."""
        try:
            with open(self.metadata_file, "w") as f:
                json.dump(self.embeddings_metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")

    def initialize_intent_recognition(self):
        """Initialize intent recognition by embedding templates."""
        if self.intent_embeddings is not None:
            return

        logger.info("Initializing intent recognition templates...")

        all_templates = []
        template_labels = []

        for intent_type, templates in self.intent_templates.items():
            all_templates.extend(templates)
            template_labels.extend([intent_type] * len(templates))

        try:
            template_embeddings = self._embed_texts(all_templates)

            self.intent_embeddings = {
                "embeddings": template_embeddings,
                "labels": template_labels,
                "templates": all_templates,
            }

            logger.info(
                f"Intent recognition initialized with {len(all_templates)} templates"
            )

        except Exception as e:
            logger.error(f"Failed to initialize intent recognition: {e}")
            self.intent_embeddings = None

    def recognize_intent(self, query: str) -> Tuple[IntentType, float]:
        """Recognize intent using embedding similarity."""
        if self.intent_embeddings is None:
            self.initialize_intent_recognition()

        if self.intent_embeddings is None:
            return IntentType.FACTUAL_QUERY, 0.5

        try:
            query_embedding = self._embed_texts([query])[0]

            similarities = np.dot(
                self.intent_embeddings["embeddings"], query_embedding
            ) / (
                np.linalg.norm(self.intent_embeddings["embeddings"], axis=1)
                * np.linalg.norm(query_embedding)
            )

            best_idx = np.argmax(similarities)
            best_intent = self.intent_embeddings["labels"][best_idx]
            confidence = similarities[best_idx]

            if confidence < 0.6:
                return IntentType.FACTUAL_QUERY, confidence

            return best_intent, confidence

        except Exception as e:
            logger.error(f"Intent recognition failed: {e}")
            return IntentType.FACTUAL_QUERY, 0.5

    def embed_chunks(
        self, chunks_file: Optional[Union[str, Path]] = None
    ) -> Dict[str, np.ndarray]:
        """Embed chunks with intelligent caching."""
        try:
            if chunks_file:
                chunks_file = Path(chunks_file)
                if chunks_file.exists():
                    return self._embed_chunks_file(chunks_file)
                else:
                    raise ValueError(f"Chunks file not found: {chunks_file}")

            all_embeddings = {}
            chunk_files = list(self.chunks_dir.glob("*_chunks.jsonl"))

            if not chunk_files:
                raise ValueError(f"No chunk files found in {self.chunks_dir}")

            for file_path in tqdm(chunk_files, desc="Embedding files"):
                try:
                    file_embeddings = self._embed_chunks_file(file_path)
                    all_embeddings.update(file_embeddings)
                except Exception as e:
                    logger.error(f"Failed to embed {file_path.name}: {e}")
                    continue

            return all_embeddings

        except Exception as e:
            logger.error(f"Error in embed_chunks: {e}")
            raise

    def _embed_chunks_file(self, chunks_file: Path) -> Dict[str, np.ndarray]:
        """Embed chunks from a single file with caching."""
        logger.info(f"Processing: {chunks_file.name}")

        output_path = (
            self.embeddings_dir
            / f"{chunks_file.stem.replace('_chunks', '')}_embeddings.npz"
        )

        if not self._needs_regeneration(chunks_file, output_path):
            logger.info(f"Loading cached embeddings for {chunks_file.name}")
            return self._load_existing_embeddings(output_path)

        chunks = []
        chunk_ids = []

        with open(chunks_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    chunk = json.loads(line)
                    chunks.append(chunk["text"])
                    chunk_ids.append(chunk["chunk_id"])
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Line {line_num}: Invalid data - {e}")

        if not chunks:
            raise ValueError(f"No valid chunks found in {chunks_file.name}")

        logger.info(f"Generating embeddings for {len(chunks)} chunks")

        embeddings = self._embed_texts(chunks)

        self._save_embeddings(output_path, embeddings, chunk_ids)
        self._update_metadata_cache(chunks_file, output_path, len(chunks))

        embeddings_dict = {
            chunk_id: embedding
            for chunk_id, embedding in zip(chunk_ids, embeddings)
        }

        logger.info(f"Successfully embedded {len(chunks)} chunks")
        return embeddings_dict

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed texts using OpenAI API with batching."""
        if not texts:
            return np.array([])

        all_embeddings = []
        batch_size = 100

        for i in tqdm(
            range(0, len(texts), batch_size), desc="Generating embeddings"
        ):
            batch_texts = texts[i : i + batch_size]

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.client.embeddings.create(
                        model=self.model_name,
                        input=batch_texts,
                        dimensions=self.dimension,
                    )

                    batch_embeddings = [
                        item.embedding for item in response.data
                    ]
                    all_embeddings.extend(batch_embeddings)
                    break

                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Attempt {attempt + 1} failed: {e}. Retrying..."
                        )
                        continue
                    else:
                        logger.error(
                            f"All retries failed for batch starting at {i}: {e}"
                        )
                        for _ in range(len(batch_texts)):
                            all_embeddings.append([0.0] * self.dimension)

        return np.array(all_embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> Optional[np.ndarray]:
        """Generate embedding for a single query."""
        if not query or not query.strip():
            logger.warning("Empty query provided")
            return None

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=[query],
                    dimensions=self.dimension,
                )
                embedding = np.array(
                    response.data[0].embedding, dtype=np.float32
                )
                return embedding

            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Query embedding attempt {attempt + 1} failed: {e}"
                    )
                    continue
                else:
                    logger.error(f"Failed to generate query embedding: {e}")
                    return None

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Embed a batch of texts efficiently."""
        if not texts:
            return []

        embeddings_matrix = self._embed_texts(texts)
        return [embeddings_matrix[i] for i in range(len(embeddings_matrix))]

    def _needs_regeneration(
        self, chunks_file: Path, embeddings_file: Path
    ) -> bool:
        """Check if embeddings need regeneration."""
        if not embeddings_file.exists():
            return True

        chunks_mtime = chunks_file.stat().st_mtime
        embeddings_mtime = embeddings_file.stat().st_mtime

        if chunks_mtime > embeddings_mtime:
            return True

        file_key = str(chunks_file)
        cached_info = self.embeddings_metadata.get(file_key, {})

        if cached_info.get("model") != self.model_name:
            return True

        return False

    def _save_embeddings(
        self,
        output_path: Union[str, Path],
        embeddings: np.ndarray,
        chunk_ids: List[str],
    ):
        """Save embeddings with error handling."""
        output_path = Path(output_path)

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            np.savez_compressed(
                output_path,
                embeddings=embeddings,
                chunk_ids=np.array(chunk_ids),
                model=self.model_name,
                dimension=self.dimension,
            )

            if not output_path.exists():
                raise RuntimeError(f"File was not created: {output_path}")

            logger.info(f"Saved embeddings to {output_path}")

        except Exception as e:
            logger.error(
                f"Failed to save embeddings to {output_path}: {str(e)}"
            )
            raise RuntimeError(f"Failed to save embeddings: {str(e)}")

    def _load_existing_embeddings(
        self, embeddings_file: Path
    ) -> Dict[str, np.ndarray]:
        """Load existing embeddings file."""
        try:
            data = np.load(embeddings_file)
            embeddings = data["embeddings"]
            chunk_ids = data["chunk_ids"]

            return {
                str(chunk_id): embedding
                for chunk_id, embedding in zip(chunk_ids, embeddings)
            }
        except Exception as e:
            logger.error(
                f"Failed to load embeddings from {embeddings_file}: {e}"
            )
            raise

    def _update_metadata_cache(
        self, chunks_file: Path, embeddings_file: Path, chunk_count: int
    ):
        """Update metadata cache."""
        self.embeddings_metadata[str(chunks_file)] = {
            "embeddings_file": str(embeddings_file),
            "chunk_count": chunk_count,
            "model": self.model_name,
            "dimension": self.dimension,
            "last_updated": embeddings_file.stat().st_mtime,
        }
        self._save_metadata()

    def similarity_search(
        self,
        query: str,
        embeddings_dict: Dict[str, np.ndarray],
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """Perform similarity search against embeddings."""
        query_embedding = self.embed_query(query)
        if query_embedding is None:
            return []

        similarities = []
        for chunk_id, embedding in embeddings_dict.items():
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )
            similarities.append((chunk_id, float(similarity)))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def calculate_chunk_similarity(
        self, chunk1_embedding: np.ndarray, chunk2_embedding: np.ndarray
    ) -> float:
        """Calculate similarity between two chunk embeddings."""
        try:
            similarity = np.dot(chunk1_embedding, chunk2_embedding) / (
                np.linalg.norm(chunk1_embedding)
                * np.linalg.norm(chunk2_embedding)
            )
            return float(similarity)
        except Exception:
            return 0.0

    def get_embedding_stats(self) -> Dict:
        """Get embedding statistics."""
        try:
            stats = {
                "model": self.model_name,
                "dimension": self.dimension,
                "cached_files": len(self.embeddings_metadata),
                "embedding_files": len(list(self.embeddings_dir.glob("*.npz"))),
                "intent_recognition_enabled": self.intent_embeddings
                is not None,
            }

            total_embeddings = 0
            for npz_file in self.embeddings_dir.glob("*.npz"):
                try:
                    data = np.load(npz_file)
                    total_embeddings += len(data["embeddings"])
                except:
                    continue

            stats["total_embeddings"] = total_embeddings

            if self.intent_embeddings:
                stats["intent_templates_count"] = len(
                    self.intent_embeddings["templates"]
                )

            return stats

        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"error": str(e)}

    def clear_cache(self, chunks_file: Optional[Path] = None):
        """Clear embedding cache."""
        try:
            if chunks_file:
                file_key = str(chunks_file)
                if file_key in self.embeddings_metadata:
                    del self.embeddings_metadata[file_key]
                    logger.info(f"Cleared cache for {chunks_file.name}")
            else:
                self.embeddings_metadata.clear()
                logger.info("Cleared all embedding caches")

            self._save_metadata()

        except Exception as e:
            logger.error(f"Error clearing cache: {e}")

    def validate_embeddings_integrity(self) -> Dict[str, Any]:
        """Validate embeddings integrity and consistency."""
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "stats": {},
        }

        try:
            # Check metadata consistency
            for file_key, metadata in self.embeddings_metadata.items():
                embeddings_file = Path(metadata.get("embeddings_file", ""))

                if not embeddings_file.exists():
                    validation_results["errors"].append(
                        f"Missing embeddings file: {embeddings_file}"
                    )
                    validation_results["valid"] = False
                    continue

                try:
                    data = np.load(embeddings_file)
                    expected_count = metadata.get("chunk_count", 0)
                    actual_count = len(data["embeddings"])

                    if expected_count != actual_count:
                        validation_results["warnings"].append(
                            f"Count mismatch in {embeddings_file.name}: expected {expected_count}, got {actual_count}"
                        )

                    stored_model = data.get("model", "unknown")
                    if stored_model != self.model_name:
                        validation_results["warnings"].append(
                            f"Model mismatch in {embeddings_file.name}: {stored_model} vs {self.model_name}"
                        )

                except Exception as e:
                    validation_results["errors"].append(
                        f"Cannot validate {embeddings_file.name}: {str(e)}"
                    )

            # Intent recognition validation
            if self.intent_embeddings is None:
                validation_results["warnings"].append(
                    "Intent recognition not initialized"
                )
            else:
                validation_results["stats"]["intent_templates"] = len(
                    self.intent_embeddings["templates"]
                )

            validation_results["stats"].update(self.get_embedding_stats())

        except Exception as e:
            validation_results["errors"].append(f"Validation failed: {str(e)}")
            validation_results["valid"] = False

        return validation_results

    def optimize_storage(self) -> Dict[str, Any]:
        """Optimize embedding storage and clean up unused files."""
        optimization_results = {
            "cleaned_files": 0,
            "recovered_space_mb": 0,
            "errors": [],
        }

        try:
            # Find orphaned embedding files
            metadata_files = set(
                Path(info["embeddings_file"]).name
                for info in self.embeddings_metadata.values()
            )

            actual_files = set(
                f.name for f in self.embeddings_dir.glob("*.npz")
            )
            orphaned_files = actual_files - metadata_files

            for orphan in orphaned_files:
                try:
                    orphan_path = self.embeddings_dir / orphan
                    file_size = orphan_path.stat().st_size
                    orphan_path.unlink()

                    optimization_results["cleaned_files"] += 1
                    optimization_results["recovered_space_mb"] += file_size / (
                        1024 * 1024
                    )

                except Exception as e:
                    optimization_results["errors"].append(
                        f"Failed to remove {orphan}: {str(e)}"
                    )

            logger.info(
                f"Optimization complete: {optimization_results['cleaned_files']} files cleaned, "
                f"{optimization_results['recovered_space_mb']:.2f} MB recovered"
            )

        except Exception as e:
            optimization_results["errors"].append(
                f"Optimization failed: {str(e)}"
            )

        return optimization_results
