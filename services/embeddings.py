import os
import json
import logging
import hashlib
from typing import Dict
from typing import List
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm
from openai import OpenAI

from config.settings import settings
from config.settings import ROOT_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("embeddings")


class EmbeddingError(Exception):
    """Custom exception for embedding errors."""

    pass


class EmbeddingService:
    """
    Production-ready embedding service with:
    - Cost optimization through caching
    - Hash-based validation to prevent regeneration
    - Comprehensive error handling
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        chunks_dir: Path = None,
        embeddings_dir: Path = None,
        dedup_dir: Path = None,
    ):
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EmbeddingError("OPENAI_API_KEY environment variable not set")

        try:
            self.client = OpenAI(api_key=api_key)
        except Exception as e:
            raise EmbeddingError(
                f"Failed to initialize OpenAI client: {str(e)}"
            )

        self.model_name = model_name or settings.embedding.model
        self.chunks_dir = chunks_dir or ROOT_DIR / "data" / "chunks"
        self.embeddings_dir = embeddings_dir or ROOT_DIR / "data" / "embeddings"
        self.dedup_dir = dedup_dir or ROOT_DIR / "data" / "deduplicated"

        # Ensure all necessary directories exist
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        self.dedup_dir.mkdir(parents=True, exist_ok=True)

        self.dimension = settings.embedding.dimension

        # Cache for embeddings metadata
        self.metadata_file = self.embeddings_dir / "embeddings_metadata.json"
        self.embeddings_metadata = self._load_metadata()

        logger.info(
            f"EmbeddingService initialized with model: {self.model_name}"
        )

    def _load_metadata(self) -> Dict:
        """Load embeddings metadata cache."""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, "r") as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load metadata: {str(e)}")
        return {}

    def _save_metadata(self):
        """Save embeddings metadata cache."""
        try:
            with open(self.metadata_file, "w") as f:
                json.dump(self.embeddings_metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metadata: {str(e)}")

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate hash of file content for change detection."""
        try:
            hasher = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating hash for {file_path}: {str(e)}")
            return ""

    def _needs_regeneration(
        self, chunks_file: Path, embeddings_file: Path
    ) -> bool:
        """
        Check if embeddings need regeneration based on:
        1. Embeddings file existence
        2. File hash comparison
        """
        # Check if embeddings file exists
        if not embeddings_file.exists():
            logger.info(
                f"Embeddings file doesn't exist: {embeddings_file.name}"
            )
            return True

        # Calculate current hash
        current_hash = self._calculate_file_hash(chunks_file)

        # Get cached hash
        file_key = str(chunks_file)
        cached_info = self.embeddings_metadata.get(file_key, {})
        cached_hash = cached_info.get("hash", "")

        # Compare hashes
        if current_hash != cached_hash:
            logger.info(f"File has changed: {chunks_file.name}")
            return True

        # Verify embeddings file integrity
        try:
            data = np.load(embeddings_file)
            if "embeddings" not in data or "chunk_ids" not in data:
                logger.warning(
                    f"Corrupted embeddings file: {embeddings_file.name}"
                )
                return True
        except Exception as e:
            logger.warning(f"Failed to load embeddings: {str(e)}")
            return True

        logger.info(f"Embeddings up-to-date for: {chunks_file.name}")
        return False

    def embed_chunks(
        self, chunks_file: Optional[Path] = None
    ) -> Dict[str, np.ndarray]:
        """
        Embed chunks with intelligent caching and validation.
        Only regenerates embeddings when necessary.
        """
        try:
            if chunks_file and chunks_file.exists():
                return self._embed_chunks_file(chunks_file)

            # Check for deduplicated chunks first
            dedup_file = self.dedup_dir / "deduplicated_chunks.jsonl"
            if dedup_file.exists():
                logger.info("Using deduplicated chunks for embedding")
                return self._embed_chunks_file(dedup_file, is_deduplicated=True)

            # Process all chunks files
            all_embeddings = {}
            chunk_files = list(self.chunks_dir.glob("*_chunks.jsonl"))

            if not chunk_files:
                raise EmbeddingError(
                    f"No chunk files found in {self.chunks_dir}"
                )

            for file_path in tqdm(chunk_files, desc="Embedding files"):
                try:
                    file_embeddings = self._embed_chunks_file(file_path)
                    all_embeddings.update(file_embeddings)
                except Exception as e:
                    logger.error(f"Failed to embed {file_path.name}: {str(e)}")
                    continue

            return all_embeddings

        except Exception as e:
            logger.error(f"Error in embed_chunks: {str(e)}")
            raise EmbeddingError(f"Failed to embed chunks: {str(e)}")

    def _embed_chunks_file(
        self, chunks_file: Path, is_deduplicated: bool = False
    ) -> Dict[str, np.ndarray]:
        """Embed chunks from a single file with caching."""
        try:
            logger.info(f"Processing: {chunks_file.name}")

            # Determine output path
            if is_deduplicated:
                output_path = (
                    self.embeddings_dir / "deduplicated_embeddings.npz"
                )
            else:
                output_path = (
                    self.embeddings_dir
                    / f"{chunks_file.stem.replace('_chunks', '')}_embeddings.npz"
                )

            # Check if regeneration is needed (COST OPTIMIZATION)
            if not self._needs_regeneration(chunks_file, output_path):
                logger.info(f"Skipping regeneration for {chunks_file.name}")
                # Load and return existing embeddings
                return self._load_existing_embeddings(output_path)

            # Load chunks
            chunks = []
            chunk_ids = []

            with open(chunks_file, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        chunk = json.loads(line)
                        chunks.append(chunk["text"])
                        chunk_ids.append(chunk["chunk_id"])
                    except json.JSONDecodeError as e:
                        logger.warning(
                            f"Line {line_num}: Invalid JSON - {str(e)}"
                        )
                    except KeyError as e:
                        logger.warning(f"Line {line_num}: Missing key {str(e)}")

            if not chunks:
                raise EmbeddingError(
                    f"No valid chunks found in {chunks_file.name}"
                )

            logger.info(f"Generating embeddings for {len(chunks)} chunks")

            # Generate embeddings
            embeddings = self.embed_batch(chunks)

            # Validate embeddings
            if len(embeddings) != len(chunks):
                raise EmbeddingError(
                    f"Embedding count mismatch: {len(embeddings)} != {len(chunks)}"
                )

            # Save embeddings
            self._save_embeddings(output_path, embeddings, chunk_ids)

            # Update metadata cache
            file_hash = self._calculate_file_hash(chunks_file)
            self.embeddings_metadata[str(chunks_file)] = {
                "hash": file_hash,
                "embeddings_file": str(output_path),
                "chunk_count": len(chunks),
            }
            self._save_metadata()

            # Create mapping
            embeddings_dict = {
                chunk_id: embedding
                for chunk_id, embedding in zip(chunk_ids, embeddings)
            }

            logger.info(f"Successfully embedded {len(chunks)} chunks")
            return embeddings_dict

        except Exception as e:
            logger.error(
                f"Error embedding chunks file {chunks_file.name}: {str(e)}"
            )
            raise EmbeddingError(
                f"Failed to embed {chunks_file.name}: {str(e)}"
            )

    def _save_embeddings(
        self, output_path: Path, embeddings: np.ndarray, chunk_ids: List[str]
    ):
        """Save embeddings with atomic write."""
        temp_path = None
        try:
            # Ensure parent directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Create temp filename by appending to stem, not replacing suffix
            # This prevents np.savez_compressed from creating .tmp.npz
            temp_path = output_path.parent / f"{output_path.stem}_temp.npz"

            # Use np.savez_compressed for better space efficiency
            np.savez_compressed(
                temp_path,
                embeddings=np.array(embeddings),
                chunk_ids=np.array(chunk_ids),
            )
            
            # Verify temp file was created
            if not temp_path.exists():
                raise EmbeddingError(f"Temporary file was not created: {temp_path}")

            # Atomic rename using os.rename which is more reliable
            import os
            os.rename(str(temp_path), str(output_path))
            
            logger.info(f"Saved embeddings to {output_path}")

        except Exception as e:
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                except:
                    pass
            raise EmbeddingError(f"Failed to save embeddings: {str(e)}")

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
                f"Failed to load embeddings from {embeddings_file}: {str(e)}"
            )
            raise EmbeddingError(f"Cannot load embeddings: {str(e)}")

    def embed_batch(
        self, texts: List[str], batch_size: int = 100
    ) -> np.ndarray:
        """
        Generate embeddings with batching and retry logic.
        Reduced batch size for better reliability.
        """
        if not texts:
            return np.array([])

        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size

        for i in tqdm(
            range(0, len(texts), batch_size),
            total=total_batches,
            desc="Generating embeddings",
        ):
            batch_texts = texts[i : i + batch_size]

            # Retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.client.embeddings.create(
                        model=self.model_name, input=batch_texts
                    )

                    batch_embeddings = [
                        item.embedding for item in response.data
                    ]
                    all_embeddings.extend(batch_embeddings)
                    break

                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Attempt {attempt + 1} failed: {str(e)}. Retrying..."
                        )
                        continue
                    else:
                        logger.error(
                            f"All retries failed for batch starting at {i}: {str(e)}"
                        )
                        # Add zero embeddings as fallback
                        for _ in range(len(batch_texts)):
                            all_embeddings.append([0.0] * self.dimension)

        return np.array(all_embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> Optional[np.ndarray]:
        """Generate embedding for a single query with error handling."""
        if not query or not query.strip():
            logger.warning("Empty query provided")
            return None

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.model_name, input=[query]
                )
                embedding = np.array(
                    response.data[0].embedding, dtype=np.float32
                )
                return embedding

            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Query embedding attempt {attempt + 1} failed: {str(e)}"
                    )
                    continue
                else:
                    logger.error(
                        f"Failed to generate query embedding: {str(e)}"
                    )
                    return None

    def embed_deduplicated_chunks(self) -> Dict[str, np.ndarray]:
        """Embed deduplicated chunks specifically."""
        try:
            dedup_file = self.dedup_dir / "deduplicated_chunks.jsonl"

            if not dedup_file.exists():
                raise EmbeddingError(
                    f"Deduplicated chunks not found: {dedup_file}"
                )

            return self._embed_chunks_file(dedup_file, is_deduplicated=True)

        except Exception as e:
            logger.error(f"Error embedding deduplicated chunks: {str(e)}")
            raise EmbeddingError(
                f"Failed to embed deduplicated chunks: {str(e)}"
            )

    def load_embeddings(
        self, embeddings_file: Path
    ) -> Optional[Dict[str, np.ndarray]]:
        """Load embeddings from file with validation."""
        try:
            if not embeddings_file.exists():
                logger.error(f"Embeddings file not found: {embeddings_file}")
                return None

            data = np.load(embeddings_file)

            # Validate structure
            if "embeddings" not in data or "chunk_ids" not in data:
                logger.error(
                    f"Invalid embeddings file structure: {embeddings_file}"
                )
                return None

            logger.info(
                f"Loaded {len(data['embeddings'])} embeddings from {embeddings_file}"
            )
            return {
                "embeddings": data["embeddings"],
                "chunk_ids": data["chunk_ids"],
            }

        except Exception as e:
            logger.error(
                f"Error loading embeddings from {embeddings_file}: {str(e)}"
            )
            return None

    def clear_cache(self, chunks_file: Optional[Path] = None):
        """Clear metadata cache for specific file or all files."""
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
            logger.error(f"Error clearing cache: {str(e)}")

    def get_embedding_stats(self) -> Dict:
        """Get statistics about embeddings."""
        try:
            stats = {
                "model": self.model_name,
                "dimension": self.dimension,
                "cached_files": len(self.embeddings_metadata),
                "embedding_files": len(list(self.embeddings_dir.glob("*.npz"))),
            }

            # Calculate total embeddings
            total_embeddings = 0
            for npz_file in self.embeddings_dir.glob("*.npz"):
                try:
                    data = np.load(npz_file)
                    total_embeddings += len(data["embeddings"])
                except:
                    continue

            stats["total_embeddings"] = total_embeddings
            return stats

        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {"error": str(e)}
