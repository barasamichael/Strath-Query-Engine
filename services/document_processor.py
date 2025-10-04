import os
import re
import json
import logging
import hashlib
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field

from tqdm import tqdm
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from pathlib import Path
from typing import Optional

import nltk
import spacy
from bs4 import BeautifulSoup
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from nltk.tokenize import sent_tokenize

from config.settings import settings
from config.settings import ROOT_DIR
from services.embeddings import EmbeddingService

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("document_processor")

# Download required NLTK data
nltk.download("punkt", quiet=True)

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")

except:
    logger.info("Downloading spaCy model...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


@dataclass
class Chunk:
    """Enhanced chunk representation with metadata and embedding."""

    chunk_id: str
    doc_id: str
    chunk_index: int
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    information_score: float = 0.0  # Score representing information density
    merged_from: List[str] = field(
        default_factory=list
    )  # IDs of chunks this was merged from
    is_primary: bool = False  # Whether this is a primary (authoritative) chunk
    source_file: str = ""  # Path to source file

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "metadata": self.metadata,
        }

        # Add optional fields if they exist
        if self.information_score:
            result["information_score"] = self.information_score
        if self.merged_from:
            result["merged_from"] = self.merged_from
        if self.is_primary:
            result["is_primary"] = self.is_primary
        if self.source_file:
            result["source_file"] = self.source_file

        return result


class DocumentProcessor:
    def __init__(
        self,
        raw_dir: Path = None,
        processed_dir: Path = None,
        chunk_dir: Path = None,
        dedup_dir: Path = None,
        embedding_service: Optional[EmbeddingService] = None,
        enable_deduplication: bool = True,
        similarity_threshold: float = 0.92,
    ):
        self.raw_dir = raw_dir or ROOT_DIR / "data" / "raw"
        self.processed_dir = processed_dir or ROOT_DIR / "data" / "processed"
        self.chunk_dir = chunk_dir or ROOT_DIR / "data" / "chunks"
        self.dedup_dir = dedup_dir or ROOT_DIR / "data" / "deduplicated"

        # Deduplication settings
        self.enable_deduplication = enable_deduplication
        self.similarity_threshold = similarity_threshold

        # Initialize embedding service
        self.embedding_service = embedding_service or EmbeddingService()

        # Ensure directories exist
        for dir_path in [
            self.raw_dir,
            self.processed_dir,
            self.chunk_dir,
            self.dedup_dir,
        ]:
            if not dir_path.exists():
                dir_path.mkdir(parents=True)

        # Storage for chunks during processing
        self.all_chunks = []
        self.deduplicated_chunks = []

    def process_all_documents(self) -> List[Dict[str, Any]]:
        """Process all documents in the raw directory and return their metadata."""
        logger.info(f"Processing all documents in {self.raw_dir}")

        documents_metadata = []

        for file_path in tqdm(
            list(self.raw_dir.glob("**/*")), desc="Processing files"
        ):
            if not file_path.is_file():
                continue

            try:
                metadata = self.process_document(file_path)
                if metadata:
                    documents_metadata.append(metadata)
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")

        # If deduplication is enabled and we have processed multiple documents,
        # deduplicate chunks across all documents
        if self.enable_deduplication and len(documents_metadata) > 1:
            logger.info(
                f"Starting cross-document deduplication with {len(self.all_chunks)} chunks"
            )
            self._deduplicate_chunks()

            # Save deduplicated chunks
            deduplicated_path = self.dedup_dir / "deduplicated_chunks.jsonl"
            self._save_deduplicated_chunks(deduplicated_path)
            logger.info(
                f"Saved {len(self.deduplicated_chunks)} deduplicated chunks to {deduplicated_path}"
            )

            # Generate deduplication report
            self._generate_deduplication_report()

        # Clear chunk storage to free memory
        self.all_chunks = []
        self.deduplicated_chunks = []

        return documents_metadata

    def process_document(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Process a single document and return its metadata."""
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return None

        logger.info(f"Processing document: {file_path}")

        # Extract document content based on file type
        try:
            text, doc_type = self._extract_text(file_path)
        except Exception as e:
            logger.error(f"Failed to extract text from {file_path}: {str(e)}")
            return None

        if not text:
            logger.warning(f"No text extracted from {file_path}")
            return None

        # Generate document ID
        doc_id = self._generate_document_id(file_path)

        # Clean and normalize text
        cleaned_text = self._clean_text(text)

        # Save processed text
        processed_path = self.processed_dir / f"{doc_id}.txt"
        with open(processed_path, "w", encoding="utf-8") as f:
            f.write(cleaned_text)

        # Create chunks
        chunks = self._create_chunks(cleaned_text, doc_id, str(file_path))

        # Save chunks
        chunk_path = self.chunk_dir / f"{doc_id}_chunks.jsonl"
        self._save_chunks(chunks, chunk_path)

        # Store chunks for cross-document deduplication
        self.all_chunks.extend(chunks)

        # Return document metadata
        metadata = {
            "doc_id": doc_id,
            "file_name": file_path.name,
            "file_path": str(file_path),
            "doc_type": doc_type,
            "processed_path": str(processed_path),
            "chunks_path": str(chunk_path),
            "num_chunks": len(chunks),
        }

        logger.info(
            f"Document processed: {doc_id}, created {len(chunks)} chunks"
        )
        return metadata

    def _extract_text(self, file_path: Path) -> Tuple[str, str]:
        """Extract text from a document based on its file type."""
        file_extension = file_path.suffix.lower()

        if file_extension == ".pdf":
            loader = PyPDFLoader(str(file_path))
            pages = loader.load()
            text = "\n\n".join(page.page_content for page in pages)
            return text, "pdf"

        elif file_extension == ".txt":
            loader = TextLoader(str(file_path), encoding="utf-8")
            documents = loader.load()
            text = "\n\n".join(doc.page_content for doc in documents)
            return text, "txt"

        elif file_extension in [".docx", ".doc"]:
            loader = Docx2txtLoader(str(file_path))
            documents = loader.load()
            text = "\n\n".join(doc.page_content for doc in documents)
            return text, "docx"

        elif file_extension == ".html":
            with open(file_path, "r", encoding="utf-8") as f:
                html = f.read()
            soup = BeautifulSoup(html, "html.parser")
            text = soup.get_text(separator="\n")
            return text, "html"

        elif file_extension == ".md":
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            return text, "markdown"

        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Replace multiple newlines with a single newline
        text = re.sub(r"\n+", "\n", text)

        # Replace multiple spaces with a single space
        text = re.sub(r"\s+", " ", text)

        # Remove strange control characters
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]", "", text)

        # Strip whitespace
        text = text.strip()

        return text

    def _create_chunks(
        self, text: str, doc_id: str, source_file: str = ""
    ) -> List[Chunk]:
        """Split text into chunks using LangChain's RecursiveCharacterTextSplitter."""
        # Initialize text splitter with settings from config
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunking.chunk_size,
            chunk_overlap=settings.chunking.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        # Split text into chunks
        texts = text_splitter.split_text(text)

        # Create chunk objects with metadata
        chunks = []
        for i, chunk_text in enumerate(texts):
            chunk_id = f"{doc_id}_{i:04d}"
            chunk = Chunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                chunk_index=i,
                text=chunk_text,
                metadata={"doc_id": doc_id, "chunk_index": i},
                source_file=source_file,
            )
            chunks.append(chunk)

        return chunks

    def _save_chunks(self, chunks: List[Chunk], output_path: Path) -> None:
        """Save chunks to a JSONL file."""
        with open(output_path, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk.to_dict()) + "\n")

    def _save_deduplicated_chunks(self, output_path: Path) -> None:
        """Save deduplicated chunks to a JSONL file."""
        with open(output_path, "w", encoding="utf-8") as f:
            for chunk in self.deduplicated_chunks:
                chunk_data = chunk.to_dict()
                # Add RAG-specific metadata
                chunk_data["metadata"]["merged"] = len(chunk.merged_from) > 1
                chunk_data["metadata"][
                    "information_score"
                ] = chunk.information_score
                if len(chunk.merged_from) > 1:
                    chunk_data["metadata"]["merged_from"] = chunk.merged_from
                f.write(json.dumps(chunk_data) + "\n")

    def _generate_document_id(self, file_path: Path) -> str:
        """Generate a unique document ID based on file path and modification time."""
        file_stat = file_path.stat()
        unique_string = f"{file_path}_{file_stat.st_mtime}"
        return hashlib.md5(unique_string.encode()).hexdigest()

    # --- DEDUPLICATION METHODS ---

    def _deduplicate_chunks(self) -> None:
        """Deduplicate chunks across all documents."""
        if not self.all_chunks:
            logger.warning("No chunks to deduplicate")
            return

        # Step 1: Generate embeddings for all chunks
        logger.info(f"Generating embeddings for {len(self.all_chunks)} chunks")
        self._generate_embeddings()

        # Step 2: Analyze information content in chunks
        logger.info("Analyzing information density in chunks")
        self._analyze_information()

        # Step 3: Find similar chunks
        logger.info(
            f"Finding similar chunks with threshold {self.similarity_threshold}"
        )
        similar_clusters = self._find_similar_chunks()

        # Step 4: Merge similar chunks preserving information
        logger.info(
            f"Merging {len(similar_clusters)} clusters of similar chunks"
        )
        merged_chunks = self._merge_similar_clusters(similar_clusters)

        # Step 5: Combine merged chunks with non-duplicated chunks
        merged_chunk_ids = set()
        for cluster in similar_clusters:
            for chunk in cluster:
                merged_chunk_ids.add(chunk.chunk_id)

        # Add all merged chunks
        self.deduplicated_chunks = []
        self.deduplicated_chunks.extend(merged_chunks)

        # Add non-duplicated chunks
        for chunk in self.all_chunks:
            if chunk.chunk_id not in merged_chunk_ids:
                self.deduplicated_chunks.append(chunk)

        logger.info(
            f"Deduplication complete: {len(self.all_chunks)} original chunks → "
            f"{len(self.deduplicated_chunks)} deduplicated chunks "
            f"({len(merged_chunks)} merged, {len(self.deduplicated_chunks) - len(merged_chunks)} unchanged)"
        )

    def _generate_embeddings(self) -> None:
        """Generate embeddings for all chunks."""
        # Process in batches of 20
        batch_size = 20
        for i in tqdm(
            range(0, len(self.all_chunks), batch_size),
            desc="Generating embeddings",
        ):
            batch = self.all_chunks[i : i + batch_size]
            texts = [chunk.text for chunk in batch]

            # Generate embeddings
            embeddings = self.embedding_service.embed_batch(texts)

            # Assign embeddings to chunks
            for chunk, embedding in zip(batch, embeddings):
                chunk.embedding = embedding

    def _analyze_information(self) -> None:
        """Analyze information density in chunks."""
        for chunk in tqdm(self.all_chunks, desc="Analyzing information"):
            chunk.information_score = self._calculate_information_score(chunk)

    def _calculate_information_score(self, chunk: Chunk) -> float:
        """Calculate an information score based on various metrics."""
        text = chunk.text

        # Basic metrics
        length_score = min(
            1.0, len(text) / 1000
        )  # Favor longer chunks up to a point

        # Count named entities (approximate using capitalized words)
        capitalized_words = len(re.findall(r"\b[A-Z][a-zA-Z]*\b", text))
        entity_score = min(1.0, capitalized_words / 20)  # Cap at 20 entities

        # Count numbers and dates
        numbers = len(re.findall(r"\b\d+\b", text))
        number_score = min(1.0, numbers / 10)  # Cap at 10 numbers

        # Check for lists and structured content
        has_lists = 1.0 if re.search(r"(\n\s*[-*•]\s+|\d+\.\s+)", text) else 0.0

        # Check for specific phrases indicating important information
        important_phrases = [
            "important",
            "critical",
            "essential",
            "required",
            "must",
            "policy",
            "regulation",
            "rule",
            "procedure",
            "deadline",
            "contact",
            "email",
            "phone",
            "address",
            "website",
            "fee",
            "payment",
            "cost",
            "price",
            "discount",
            "schedule",
            "timetable",
            "date",
            "time",
        ]

        phrase_score = sum(
            1
            for phrase in important_phrases
            if re.search(rf"\b{phrase}\b", text.lower())
        ) / len(important_phrases)

        # Calculate final score (weights can be adjusted)
        final_score = (
            0.2 * length_score
            + 0.25 * entity_score
            + 0.2 * number_score
            + 0.15 * has_lists
            + 0.2 * phrase_score
        )

        return final_score

    def _find_similar_chunks(self) -> List[List[Chunk]]:
        """Find clusters of similar chunks."""
        # Build similarity graph
        similarity_graph = defaultdict(list)

        # Compare all chunks (can be optimized with approximate nearest neighbors)
        n = len(self.all_chunks)
        for i in tqdm(range(n), desc="Building similarity graph"):
            for j in range(i + 1, n):
                if (
                    self.all_chunks[i].embedding is None
                    or self.all_chunks[j].embedding is None
                ):
                    continue

                similarity = self._cosine_similarity(
                    self.all_chunks[i].embedding, self.all_chunks[j].embedding
                )

                if similarity >= self.similarity_threshold:
                    similarity_graph[i].append((j, similarity))
                    similarity_graph[j].append((i, similarity))

        # Find connected components (clusters)
        visited = set()
        clusters = []

        for i in range(n):
            if i in visited:
                continue

            # BFS to find connected component
            cluster = []
            queue = [i]
            visited.add(i)

            while queue:
                node = queue.pop(0)
                cluster.append(self.all_chunks[node])

                for neighbor, _ in similarity_graph[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            if len(cluster) > 1:  # Only add clusters with more than one chunk
                clusters.append(cluster)

        logger.info(f"Found {len(clusters)} clusters of similar chunks")
        return clusters

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)

        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0

        return dot_product / (norm_v1 * norm_v2)

    def _get_unique_sentences(
        self, primary_text: str, secondary_text: str
    ) -> List[str]:
        """Extract sentences from secondary_text that are not semantically present in primary_text."""
        # Tokenize into sentences
        primary_sentences = sent_tokenize(primary_text)
        secondary_sentences = sent_tokenize(secondary_text)

        # Convert to lowercase for comparison
        primary_set = {
            sentence.lower().strip() for sentence in primary_sentences
        }

        unique_sentences = []

        for sentence in secondary_sentences:
            sentence_lower = sentence.lower().strip()

            # Check if sentence or very similar is already in primary
            if sentence_lower not in primary_set:
                # Check for near-duplicates (could be enhanced with embeddings)
                is_unique = True
                for primary_sentence in primary_set:
                    # Simple word overlap check
                    primary_words = set(primary_sentence.split())
                    secondary_words = set(sentence_lower.split())

                    if len(primary_words) > 0 and len(secondary_words) > 0:
                        overlap = len(
                            primary_words.intersection(secondary_words)
                        ) / max(len(primary_words), len(secondary_words))

                        if overlap > 0.8:  # High word overlap
                            is_unique = False
                            break

                if is_unique:
                    unique_sentences.append(sentence)

        return unique_sentences

    def _merge_chunks(self, chunks: List[Chunk]) -> Chunk:
        """Merge a list of similar chunks, preserving all important information."""
        if not chunks:
            return None

        # Find primary chunk (highest information score)
        primary_chunk = max(chunks, key=lambda c: c.information_score)
        primary_chunk.is_primary = True

        # Start with primary chunk text
        merged_text = primary_chunk.text
        merged_from = [chunk.chunk_id for chunk in chunks]
        source_files = {
            chunk.source_file for chunk in chunks if chunk.source_file
        }

        # Extract and add unique information from other chunks
        for chunk in chunks:
            if chunk.chunk_id == primary_chunk.chunk_id:
                continue

            unique_sentences = self._get_unique_sentences(
                merged_text, chunk.text
            )

            if unique_sentences:
                # Add unique sentences at the end
                merged_text += "\n\n--- Additional Information ---\n"
                merged_text += " ".join(unique_sentences)

        # Create merged chunk
        merged_chunk_id = (
            f"merged_{hashlib.md5(merged_text[:100].encode()).hexdigest()}"
        )
        merged_chunk = Chunk(
            chunk_id=merged_chunk_id,
            doc_id=primary_chunk.doc_id,
            chunk_index=primary_chunk.chunk_index,
            text=merged_text,
            metadata={
                "doc_id": primary_chunk.doc_id,
                "chunk_index": primary_chunk.chunk_index,
                "merged_from_chunks": [chunk.chunk_id for chunk in chunks],
                "merged_from_files": list(source_files),
                "primary_chunk_id": primary_chunk.chunk_id,
            },
            information_score=primary_chunk.information_score,
            merged_from=merged_from,
            is_primary=True,
            source_file=f"MERGED({','.join(sorted(source_files))})"
            if source_files
            else "",
        )

        return merged_chunk

    def _merge_similar_clusters(
        self, clusters: List[List[Chunk]]
    ) -> List[Chunk]:
        """Process all clusters and merge similar chunks."""
        merged_chunks = []

        for cluster in tqdm(clusters, desc="Merging clusters"):
            merged_chunk = self._merge_chunks(cluster)
            if merged_chunk:
                merged_chunks.append(merged_chunk)

        logger.info(f"Created {len(merged_chunks)} merged chunks")
        return merged_chunks

    def _generate_deduplication_report(self) -> None:
        """Generate a report summarizing the deduplication process."""
        # Calculate statistics
        total_original_text = sum(len(chunk.text) for chunk in self.all_chunks)
        total_deduplicated_text = sum(
            len(chunk.text) for chunk in self.deduplicated_chunks
        )
        text_reduction = total_original_text - total_deduplicated_text

        merged_chunks = [
            chunk
            for chunk in self.deduplicated_chunks
            if chunk.merged_from and len(chunk.merged_from) > 1
        ]

        # Build report
        report = {
            "timestamp": logging.Formatter.formatTime(
                logging.Formatter(),
                logging.LogRecord("", 0, "", 0, None, None, None),
            ),
            "chunk_size": settings.chunking.chunk_size,
            "chunk_overlap": settings.chunking.chunk_overlap,
            "similarity_threshold": self.similarity_threshold,
            "stats": {
                "total_original_chunks": len(self.all_chunks),
                "total_deduplicated_chunks": len(self.deduplicated_chunks),
                "merged_chunks": len(merged_chunks),
                "unchanged_chunks": len(self.deduplicated_chunks)
                - len(merged_chunks),
                "total_original_text": total_original_text,
                "total_deduplicated_text": total_deduplicated_text,
                "text_reduction": text_reduction,
                "reduction_percentage": (
                    text_reduction / total_original_text * 100
                )
                if total_original_text > 0
                else 0,
            },
            "merged_chunks": [
                {
                    "id": chunk.chunk_id,
                    "merged_from": chunk.merged_from,
                    "source_files": chunk.metadata.get("merged_from_files", []),
                    "information_score": chunk.information_score,
                }
                for chunk in merged_chunks
            ],
        }

        # Save report to JSON file
        report_path = self.dedup_dir / "deduplication_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        # Generate human-readable summary
        summary_path = self.dedup_dir / "deduplication_summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("Smart Chunk Deduplication Summary\n")
            f.write("===============================\n\n")

            f.write(
                f"Chunk Size: {settings.chunking.chunk_size}, Chunk Overlap: {settings.chunking.chunk_overlap}\n"
            )
            f.write(f"Similarity Threshold: {self.similarity_threshold}\n\n")

            f.write(f"Original Chunks: {len(self.all_chunks)}\n")
            f.write(f"Deduplicated Chunks: {len(self.deduplicated_chunks)}\n")
            f.write(f"Merged Chunks: {len(merged_chunks)}\n")
            f.write(
                f"Unchanged Chunks: {len(self.deduplicated_chunks) - len(merged_chunks)}\n\n"
            )

            f.write(
                f"Text Reduction: {text_reduction} characters ({report['stats']['reduction_percentage']:.2f}%)\n\n"
            )

            # List top merged chunks
            if merged_chunks:
                f.write("Top Merged Chunks (by information score):\n")
                top_merged = sorted(
                    merged_chunks,
                    key=lambda c: c.information_score,
                    reverse=True,
                )[:10]
                for i, chunk in enumerate(top_merged, 1):
                    source_files = chunk.metadata.get("merged_from_files", [])
                    f.write(
                        f"  {i}. Merged from {len(chunk.merged_from)} chunks across {len(source_files)} files\n"
                    )
                    f.write(
                        f"     Information score: {chunk.information_score:.4f}\n"
                    )
                    f.write(f"     First 100 chars: {chunk.text[:100]}...\n\n")

        logger.info(f"Generated deduplication reports at {self.dedup_dir}")
