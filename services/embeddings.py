import os
import json
import logging
from typing import Dict
from typing import List
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm
from openai import OpenAI

from config.settings import settings
from config.settings import ROOT_DIR

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("embeddings")

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class EmbeddingService:
    def __init__(
        self,
        model_name: Optional[str] = None,
        chunks_dir: Path = None,
        embeddings_dir: Path = None,
        dedup_dir: Path = None,
    ):
        self.model_name = model_name or settings.embedding.model
        self.chunks_dir = chunks_dir or ROOT_DIR / "data" / "chunks"
        self.embeddings_dir = embeddings_dir or ROOT_DIR / "data" / "embeddings"
        self.dedup_dir = dedup_dir or ROOT_DIR / "data" / "deduplicated"

        # Ensure directories exist
        for dir_path in [self.embeddings_dir, self.dedup_dir]:
            if not dir_path.exists():
                dir_path.mkdir(parents=True)

        # For OpenAI ada-002, the dimension is 1536
        self.dimension = settings.embedding.dimension
        logger.info(
            f"Using OpenAI embedding model: {self.model_name} with dimension {self.dimension}"
        )

    def embed_chunks(
        self, chunks_file: Optional[Path] = None
    ) -> Dict[str, np.ndarray]:
        """
        Embed all chunks from a specific file or all files in chunks directory.
        Returns a dictionary mapping chunk IDs to their embeddings.
        """
        if chunks_file and chunks_file.exists():
            # Embed a single chunks file
            return self._embed_chunks_file(chunks_file)
        else:
            # Check if deduplicated chunks exist and use those preferentially
            dedup_file = self.dedup_dir / "deduplicated_chunks.jsonl"
            if dedup_file.exists():
                logger.info(
                    f"Using deduplicated chunks for embedding: {dedup_file}"
                )
                return self._embed_chunks_file(dedup_file, is_deduplicated=True)

            # Otherwise, embed all chunks files in the directory
            all_embeddings = {}
            for file_path in tqdm(
                list(self.chunks_dir.glob("*_chunks.jsonl")),
                desc="Embedding files",
            ):
                file_embeddings = self._embed_chunks_file(file_path)
                all_embeddings.update(file_embeddings)

            return all_embeddings

    def _embed_chunks_file(
        self, chunks_file: Path, is_deduplicated: bool = False
    ) -> Dict[str, np.ndarray]:
        """Embed all chunks from a single file using OpenAI API."""
        logger.info(f"Embedding chunks from: {chunks_file}")

        # Load chunks from file
        chunks = []
        chunk_ids = []

        with open(chunks_file, "r", encoding="utf-8") as f:
            for line in f:
                chunk = json.loads(line)
                # Handle both original and deduplicated chunk formats
                if is_deduplicated:
                    chunks.append(chunk["text"])
                    chunk_ids.append(chunk["chunk_id"])
                else:
                    chunks.append(chunk["text"])
                    chunk_ids.append(chunk["chunk_id"])

        if not chunks:
            logger.warning(f"No chunks found in {chunks_file}")
            return {}

        # Generate embeddings in batches
        embeddings = self.embed_batch(chunks)

        # Create mapping from chunk IDs to embeddings
        embeddings_dict = {
            chunk_id: embedding
            for chunk_id, embedding in zip(chunk_ids, embeddings)
        }

        # Save embeddings to file
        if is_deduplicated:
            output_path = self.embeddings_dir / "deduplicated_embeddings.npz"
        else:
            output_path = (
                self.embeddings_dir
                / f"{chunks_file.stem.replace('_chunks', '')}_embeddings.npz"
            )

        np.savez(
            output_path,
            embeddings=np.array(embeddings),
            chunk_ids=np.array(chunk_ids),
        )

        logger.info(f"Embedded {len(chunks)} chunks. Saved to {output_path}")
        return embeddings_dict

    def embed_batch(self, texts: List[str], batch_size: int = 20) -> np.ndarray:
        """Generate embeddings for a list of texts in batches using OpenAI API."""
        all_embeddings = []

        for i in tqdm(
            range(0, len(texts), batch_size), desc="Generating embeddings"
        ):
            batch_texts = texts[i : i + batch_size]

            try:
                # Call OpenAI API for embeddings
                response = client.embeddings.create(
                    model=self.model_name, input=batch_texts
                )

                # Extract embeddings from response
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)

            except Exception as e:
                logger.error(f"Error generating embeddings: {str(e)}")
                # Add zero embeddings as fallback
                for _ in range(len(batch_texts)):
                    all_embeddings.append([0.0] * self.dimension)

        return np.array(all_embeddings)

    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a single query text using OpenAI API."""
        try:
            response = client.embeddings.create(
                model=self.model_name, input=[query]
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            logger.error(f"Error generating query embedding: {str(e)}")
            return np.zeros(self.dimension)

    def embed_deduplicated_chunks(self) -> Dict[str, np.ndarray]:
        """Specifically embed the deduplicated chunks."""
        dedup_file = self.dedup_dir / "deduplicated_chunks.jsonl"
        if not dedup_file.exists():
            logger.warning(f"Deduplicated chunks file not found: {dedup_file}")
            return {}

        return self._embed_chunks_file(dedup_file, is_deduplicated=True)
