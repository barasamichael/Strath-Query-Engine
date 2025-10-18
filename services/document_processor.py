import os
import re
import json
import hashlib
import logging
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union
from pathlib import Path
from typing import Optional
from dataclasses import field
from dataclasses import dataclass
from collections import defaultdict

import nltk
import spacy
import numpy as np
from bs4 import BeautifulSoup
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config.settings import ROOT_DIR
from config.settings import settings
from services.embeddings import EmbeddingService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("document_processor")

# Download required NLTK data
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.info("Downloading spaCy model...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


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
            ):
                if isinstance(value, str):
                    new_kwargs[key] = Path(value)
                else:
                    new_kwargs[key] = value
            else:
                new_kwargs[key] = value

        return func(*new_args, **new_kwargs)

    return wrapper


class DocumentProcessingError(Exception):
    """Custom exception for document processing errors."""

    pass


@dataclass
class Chunk:
    """Enhanced chunk representation with metadata and embedding."""

    chunk_id: str
    doc_id: str
    chunk_index: int
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    information_score: float = 0.0
    merged_from: List[str] = field(default_factory=list)
    is_primary: bool = False
    source_file: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "metadata": self.metadata,
        }

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
    """Production-ready document processor with comprehensive error handling and bulletproof path management."""

    def __init__(
        self,
        raw_dir: Optional[Union[str, Path]] = None,
        processed_dir: Optional[Union[str, Path]] = None,
        chunk_dir: Optional[Union[str, Path]] = None,
        dedup_dir: Optional[Union[str, Path]] = None,
        embedding_service: Optional[EmbeddingService] = None,
        enable_deduplication: bool = True,
        similarity_threshold: float = 0.92,
    ):
        # Convert all path inputs to Path objects with bulletproof handling
        self.raw_dir = ensure_path(raw_dir) or ROOT_DIR / "data" / "raw"
        self.processed_dir = (
            ensure_path(processed_dir) or ROOT_DIR / "data" / "processed"
        )
        self.chunk_dir = ensure_path(chunk_dir) or ROOT_DIR / "data" / "chunks"
        self.dedup_dir = (
            ensure_path(dedup_dir) or ROOT_DIR / "data" / "deduplicated"
        )

        self.enable_deduplication = enable_deduplication
        self.similarity_threshold = similarity_threshold

        # Ensure directories exist with proper error handling
        for dir_path in [
            self.raw_dir,
            self.processed_dir,
            self.chunk_dir,
            self.dedup_dir,
        ]:
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise DocumentProcessingError(
                    f"Failed to create directory {dir_path}: {str(e)}"
                )

        # Index to track document metadata
        self.document_index_path = self.processed_dir / "document_index.json"
        self.document_index = self._load_document_index()

        # Initialize embedding service
        try:
            self.embedding_service = embedding_service or EmbeddingService()
        except Exception as e:
            logger.error(f"Failed to initialize embedding service: {str(e)}")
            raise DocumentProcessingError(
                f"Embedding service initialization failed: {str(e)}"
            )

        self.all_chunks = []
        self.deduplicated_chunks = []

    def _load_document_index(self) -> Dict[str, Dict[str, Any]]:
        """Load document index from disk or create if it doesn't exist."""
        try:
            if self.document_index_path.exists():
                with open(self.document_index_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Error loading document index: {str(e)}")
            return {}

    def _save_document_index(self) -> bool:
        """Save document index to disk."""
        try:
            # Ensure parent directory exists
            self.document_index_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self.document_index_path, "w", encoding="utf-8") as f:
                json.dump(self.document_index, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving document index: {str(e)}")
            return False

    @safe_path_operation
    def process_all_documents(self) -> List[Dict[str, Any]]:
        """Process all documents with comprehensive error handling."""
        logger.info(f"Processing all documents in {self.raw_dir}")

        if not self.raw_dir.exists():
            raise DocumentProcessingError(
                f"Raw directory not found: {self.raw_dir}"
            )

        documents_metadata = []
        failed_files = []

        try:
            files = list(self.raw_dir.glob("**/*"))
            if not files:
                logger.warning(f"No files found in {self.raw_dir}")
                return []

            for file_path in tqdm(files, desc="Processing files"):
                if not file_path.is_file():
                    continue

                try:
                    metadata = self.process_document(file_path)
                    if metadata:
                        documents_metadata.append(metadata)
                except Exception as e:
                    logger.error(
                        f"Failed to process {file_path.name}: {str(e)}"
                    )
                    failed_files.append((file_path.name, str(e)))
                    continue

            # Report results
            logger.info(
                f"Successfully processed: {len(documents_metadata)} files"
            )
            if failed_files:
                logger.warning(f"Failed to process: {len(failed_files)} files")
                for filename, error in failed_files[
                    :5
                ]:  # Show first 5 failures
                    logger.warning(f"  - {filename}: {error}")

            # Deduplication if enabled
            if (
                self.enable_deduplication
                and len(documents_metadata) > 1
                and self.all_chunks
            ):
                try:
                    logger.info(
                        f"Starting deduplication with {len(self.all_chunks)} chunks"
                    )
                    self._deduplicate_chunks()

                    deduplicated_path = (
                        self.dedup_dir / "deduplicated_chunks.jsonl"
                    )
                    self._save_deduplicated_chunks(deduplicated_path)
                    logger.info(
                        f"Saved {len(self.deduplicated_chunks)} deduplicated chunks"
                    )

                    self._generate_deduplication_report()
                except Exception as e:
                    logger.error(f"Deduplication failed: {str(e)}")
                    # Continue without deduplication

            return documents_metadata

        except Exception as e:
            logger.error(f"Critical error in process_all_documents: {str(e)}")
            raise DocumentProcessingError(f"Batch processing failed: {str(e)}")
        finally:
            # Clear memory
            self.all_chunks = []
            self.deduplicated_chunks = []

    @safe_path_operation
    def process_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Process a single file from any location.

        Args:
            file_path: Path to the file to process

        Returns:
            Dict: Document metadata
        """
        file_path = ensure_path(file_path)
        if not file_path:
            raise DocumentProcessingError("Invalid file path provided")

        if not file_path.exists():
            raise DocumentProcessingError(f"File not found: {file_path}")

        if not file_path.is_file():
            raise DocumentProcessingError(f"Path is not a file: {file_path}")

        # Process the file directly
        return self.process_document(file_path)

    @safe_path_operation
    def process_folder(
        self, folder_path: Union[str, Path]
    ) -> List[Dict[str, Any]]:
        """
        Process all files in a folder.

        Args:
            folder_path: Path to the folder containing files to process

        Returns:
            List[Dict]: List of document metadata
        """
        folder_path = ensure_path(folder_path)
        if not folder_path:
            raise DocumentProcessingError("Invalid folder path provided")

        if not folder_path.exists():
            raise DocumentProcessingError(f"Folder not found: {folder_path}")

        if not folder_path.is_dir():
            raise DocumentProcessingError(
                f"Path is not a directory: {folder_path}"
            )

        results = []
        failed_files = []

        # Process each file in the folder
        files = list(folder_path.glob("**/*"))

        if not files:
            logger.warning(f"No files found in {folder_path}")
            return []

        for file_path in tqdm(
            files, desc=f"Processing files in {folder_path.name}"
        ):
            if not file_path.is_file():
                continue

            try:
                metadata = self.process_document(file_path)
                if metadata:
                    results.append(metadata)
            except Exception as e:
                logger.error(f"Failed to process {file_path.name}: {str(e)}")
                failed_files.append((file_path.name, str(e)))

        # Report results
        logger.info(f"Successfully processed: {len(results)} files")
        if failed_files:
            logger.warning(f"Failed to process: {len(failed_files)} files")

        return results

    @safe_path_operation
    def process_document(
        self, file_path: Union[str, Path]
    ) -> Optional[Dict[str, Any]]:
        """Process a single document with error handling."""
        file_path = ensure_path(file_path)
        if not file_path:
            raise DocumentProcessingError("Invalid file path provided")

        if not file_path.exists():
            raise DocumentProcessingError(f"File not found: {file_path}")

        logger.info(f"Processing: {file_path.name}")

        try:
            # Extract text
            text, doc_type = self._extract_text(file_path)

            if not text or not text.strip():
                logger.warning(f"No text extracted from {file_path.name}")
                return None

            # Generate document ID
            doc_id = self._generate_document_id(file_path)

            # Clean text
            cleaned_text = self._clean_text(text)

            if not cleaned_text:
                logger.warning(
                    f"Text cleaning resulted in empty content for {file_path.name}"
                )
                return None

            # Save processed text
            processed_path = self.processed_dir / f"{doc_id}.txt"
            try:
                with open(processed_path, "w", encoding="utf-8") as f:
                    f.write(cleaned_text)
            except Exception as e:
                logger.error(f"Failed to save processed text: {str(e)}")
                # Continue anyway

            # Create chunks
            chunks = self._create_chunks(cleaned_text, doc_id, str(file_path))

            if not chunks:
                logger.warning(f"No chunks created for {file_path.name}")
                return None

            # Save chunks
            chunk_path = self.chunk_dir / f"{doc_id}_chunks.jsonl"
            self._save_chunks(chunks, chunk_path)

            # Store for deduplication
            self.all_chunks.extend(chunks)

            metadata = {
                "doc_id": doc_id,
                "file_name": file_path.name,
                "file_path": str(file_path),
                "doc_type": doc_type,
                "processed_path": str(processed_path),
                "chunks_path": str(chunk_path),
                "num_chunks": len(chunks),
                "last_modified": file_path.stat().st_mtime,
                "processed_date": os.path.getmtime(processed_path)
                if processed_path.exists()
                else None,
            }

            # Update document index
            self.document_index[doc_id] = metadata
            self._save_document_index()

            logger.info(f"Processed {file_path.name}: {len(chunks)} chunks")
            return metadata

        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {str(e)}")
            raise DocumentProcessingError(
                f"Failed to process {file_path.name}: {str(e)}"
            )

    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document and its associated chunks from the system.

        Args:
            doc_id: Document ID to delete

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check if document exists in the index
            if doc_id not in self.document_index:
                logger.warning(f"Document {doc_id} not found in index")
                return False

            # Get document metadata
            metadata = self.document_index[doc_id]

            # Delete processed text file
            processed_path = ensure_path(metadata["processed_path"])
            if processed_path and processed_path.exists():
                processed_path.unlink()
                logger.info(f"Deleted processed file: {processed_path}")

            # Delete chunks file
            chunks_path = ensure_path(metadata["chunks_path"])
            if chunks_path and chunks_path.exists():
                chunks_path.unlink()
                logger.info(f"Deleted chunks file: {chunks_path}")

            # Update deduplication files if they exist
            dedup_path = self.dedup_dir / "deduplicated_chunks.jsonl"
            if dedup_path.exists():
                self._remove_document_from_deduplicated(doc_id)

            # Remove from index
            del self.document_index[doc_id]
            self._save_document_index()

            logger.info(f"Successfully deleted document: {doc_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {str(e)}")
            return False

    @safe_path_operation
    def update_document(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Update an existing document. If the document doesn't exist, it will be added.

        Args:
            file_path: Path to the updated file

        Returns:
            Dict: Updated document metadata
        """
        file_path = ensure_path(file_path)
        if not file_path:
            raise DocumentProcessingError("Invalid file path provided")

        if not file_path.exists():
            raise DocumentProcessingError(f"File not found: {file_path}")

        # Generate document ID to check if it exists
        doc_id = self._generate_document_id(file_path)

        # If document exists, delete it first
        if doc_id in self.document_index:
            logger.info(f"Document {doc_id} already exists, updating...")
            self.delete_document(doc_id)

        # Process the document
        return self.process_document(file_path)

    def list_documents(self) -> List[Dict[str, Any]]:
        """
        List all processed documents in the system.

        Returns:
            List[Dict]: List of document metadata
        """
        return list(self.document_index.values())

    def get_document_info(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific document.

        Args:
            doc_id: Document ID

        Returns:
            Dict or None: Document metadata if found
        """
        return self.document_index.get(doc_id)

    def _remove_document_from_deduplicated(self, doc_id: str) -> None:
        """
        Remove a document's chunks from the deduplicated file.

        Args:
            doc_id: Document ID to remove
        """
        dedup_path = self.dedup_dir / "deduplicated_chunks.jsonl"
        if not dedup_path.exists():
            return

        try:
            # Create a temporary file
            temp_path = dedup_path.with_suffix(".tmp")

            # Copy chunks that don't belong to the document
            filtered_chunks = []
            with open(dedup_path, "r", encoding="utf-8") as f:
                for line in f:
                    chunk_data = json.loads(line)
                    if chunk_data.get("doc_id") != doc_id:
                        # Check if this chunk was merged with chunks from the document
                        if (
                            "metadata" in chunk_data
                            and "merged_from_chunks" in chunk_data["metadata"]
                        ):
                            merged_chunks = chunk_data["metadata"][
                                "merged_from_chunks"
                            ]
                            # Remove references to chunks from the document
                            filtered_chunks_ids = [
                                c
                                for c in merged_chunks
                                if not c.startswith(f"{doc_id}_")
                            ]
                            if filtered_chunks_ids:
                                # Update the merged chunk data
                                chunk_data["metadata"][
                                    "merged_from_chunks"
                                ] = filtered_chunks_ids
                                filtered_chunks.append(chunk_data)
                        else:
                            # Not a merged chunk, keep it
                            filtered_chunks.append(chunk_data)

            # Write the filtered chunks to the temporary file
            with open(temp_path, "w", encoding="utf-8") as f:
                for chunk_data in filtered_chunks:
                    f.write(json.dumps(chunk_data) + "\n")

            # Replace the original file
            temp_path.replace(dedup_path)

            logger.info(f"Removed document {doc_id} from deduplicated chunks")

        except Exception as e:
            logger.error(
                f"Error removing document from deduplicated chunks: {str(e)}"
            )

    def _extract_text(self, file_path: Union[str, Path]) -> Tuple[str, str]:
        """Extract text with comprehensive error handling."""
        file_path = ensure_path(file_path)
        if not file_path:
            raise DocumentProcessingError(
                "Invalid file path for text extraction"
            )

        file_extension = file_path.suffix.lower()

        try:
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
                raise DocumentProcessingError(
                    f"Unsupported file type: {file_extension}"
                )

        except Exception as e:
            raise DocumentProcessingError(
                f"Text extraction failed for {file_path.name}: {str(e)}"
            )

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ""

        try:
            # Replace multiple newlines
            text = re.sub(r"\n+", "\n", text)
            # Replace multiple spaces
            text = re.sub(r"\s+", " ", text)
            # Remove control characters
            text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]", "", text)
            # Strip whitespace
            text = text.strip()
            return text
        except Exception as e:
            logger.error(f"Text cleaning error: {str(e)}")
            return text  # Return original if cleaning fails

    def _create_chunks(
        self, text: str, doc_id: str, source_file: str = ""
    ) -> List[Chunk]:
        """Create chunks with error handling."""
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.chunking.chunk_size,
                chunk_overlap=settings.chunking.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""],
            )

            texts = text_splitter.split_text(text)

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

        except Exception as e:
            logger.error(f"Chunking error: {str(e)}")
            raise DocumentProcessingError(f"Failed to create chunks: {str(e)}")

    def _save_chunks(
        self, chunks: List[Chunk], output_path: Union[str, Path]
    ) -> None:
        """Save chunks with error handling."""
        output_path = ensure_path(output_path)
        if not output_path:
            raise DocumentProcessingError(
                "Invalid output path for saving chunks"
            )

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save to temp file first
            temp_path = output_path.with_suffix(".tmp")

            with open(temp_path, "w", encoding="utf-8") as f:
                for chunk in chunks:
                    f.write(json.dumps(chunk.to_dict()) + "\n")

            # Atomic rename
            temp_path.replace(output_path)

        except Exception as e:
            if temp_path and temp_path.exists():
                temp_path.unlink()
            raise DocumentProcessingError(f"Failed to save chunks: {str(e)}")

    def _save_deduplicated_chunks(self, output_path: Union[str, Path]) -> None:
        """Save deduplicated chunks with error handling."""
        output_path = ensure_path(output_path)
        if not output_path:
            raise DocumentProcessingError(
                "Invalid output path for saving deduplicated chunks"
            )

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = output_path.with_suffix(".tmp")

            with open(temp_path, "w", encoding="utf-8") as f:
                for chunk in self.deduplicated_chunks:
                    chunk_data = chunk.to_dict()
                    chunk_data["metadata"]["merged"] = (
                        len(chunk.merged_from) > 1
                    )
                    chunk_data["metadata"][
                        "information_score"
                    ] = chunk.information_score
                    if len(chunk.merged_from) > 1:
                        chunk_data["metadata"][
                            "merged_from"
                        ] = chunk.merged_from
                    f.write(json.dumps(chunk_data) + "\n")

            temp_path.replace(output_path)

        except Exception as e:
            if temp_path and temp_path.exists():
                temp_path.unlink()
            raise DocumentProcessingError(
                f"Failed to save deduplicated chunks: {str(e)}"
            )

    def _generate_document_id(self, file_path: Union[str, Path]) -> str:
        """Generate unique document ID."""
        file_path = ensure_path(file_path)
        if not file_path:
            raise DocumentProcessingError(
                "Invalid file path for generating document ID"
            )

        try:
            file_stat = file_path.stat()
            unique_string = f"{file_path}_{file_stat.st_mtime}"
            return hashlib.md5(unique_string.encode()).hexdigest()
        except Exception as e:
            # Fallback to filename-based ID
            logger.warning(f"Using fallback ID generation: {str(e)}")
            return hashlib.md5(str(file_path).encode()).hexdigest()

    def _deduplicate_chunks(self) -> None:
        """Deduplicate chunks with comprehensive error handling."""
        if not self.all_chunks:
            logger.warning("No chunks to deduplicate")
            return

        try:
            # Step 1: Generate embeddings
            logger.info(
                f"Generating embeddings for {len(self.all_chunks)} chunks"
            )
            self._generate_embeddings()

            # Step 2: Analyze information
            logger.info("Analyzing information density")
            self._analyze_information()

            # Step 3: Find similar chunks
            logger.info(
                f"Finding similar chunks (threshold: {self.similarity_threshold})"
            )
            similar_clusters = self._find_similar_chunks()

            # Step 4: Merge clusters
            logger.info(f"Merging {len(similar_clusters)} clusters")
            merged_chunks = self._merge_similar_clusters(similar_clusters)

            # Step 5: Combine results
            merged_chunk_ids = set()
            for cluster in similar_clusters:
                for chunk in cluster:
                    merged_chunk_ids.add(chunk.chunk_id)

            self.deduplicated_chunks = []
            self.deduplicated_chunks.extend(merged_chunks)

            for chunk in self.all_chunks:
                if chunk.chunk_id not in merged_chunk_ids:
                    self.deduplicated_chunks.append(chunk)

            logger.info(
                f"Deduplication complete: {len(self.all_chunks)} → {len(self.deduplicated_chunks)} chunks"
            )

        except Exception as e:
            logger.error(f"Deduplication failed: {str(e)}")
            # Use original chunks if deduplication fails
            self.deduplicated_chunks = self.all_chunks.copy()
            raise DocumentProcessingError(f"Deduplication failed: {str(e)}")

    def _generate_embeddings(self) -> None:
        """Generate embeddings with batching and error recovery."""
        batch_size = 20
        failed_chunks = []

        for i in tqdm(
            range(0, len(self.all_chunks), batch_size),
            desc="Generating embeddings",
        ):
            batch = self.all_chunks[i: i + batch_size]
            texts = [chunk.text for chunk in batch]

            try:
                embeddings = self.embedding_service.embed_batch(texts)

                for chunk, embedding in zip(batch, embeddings):
                    # Check for zero embeddings (failures)
                    if np.sum(np.abs(embedding)) > 0:
                        chunk.embedding = embedding
                    else:
                        failed_chunks.append(chunk.chunk_id)
                        chunk.embedding = None

            except Exception as e:
                logger.error(f"Batch embedding failed: {str(e)}")
                for chunk in batch:
                    failed_chunks.append(chunk.chunk_id)
                    chunk.embedding = None

        if failed_chunks:
            logger.warning(f"Failed to embed {len(failed_chunks)} chunks")

    def _analyze_information(self) -> None:
        """Analyze information density."""
        try:
            for chunk in tqdm(self.all_chunks, desc="Analyzing information"):
                chunk.information_score = self._calculate_information_score(
                    chunk
                )
        except Exception as e:
            logger.error(f"Information analysis error: {str(e)}")
            # Continue with zero scores

    def _calculate_information_score(self, chunk: Chunk) -> float:
        """Calculate information score."""
        try:
            text = chunk.text

            length_score = min(1.0, len(text) / 1000)
            capitalized_words = len(re.findall(r"\b[A-Z][a-zA-Z]*\b", text))
            entity_score = min(1.0, capitalized_words / 20)
            numbers = len(re.findall(r"\b\d+\b", text))
            number_score = min(1.0, numbers / 10)
            has_lists = (
                1.0 if re.search(r"(\n\s*[-*•]\s+|\d+\.\s+)", text) else 0.0
            )

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

            final_score = (
                0.2 * length_score
                + 0.25 * entity_score
                + 0.2 * number_score
                + 0.15 * has_lists
                + 0.2 * phrase_score
            )

            return final_score

        except Exception as e:
            logger.error(f"Error calculating information score: {str(e)}")
            return 0.0

    def _find_similar_chunks(self) -> List[List[Chunk]]:
        """Find similar chunk clusters."""
        try:
            similarity_graph = defaultdict(list)
            valid_chunks = [
                c for c in self.all_chunks if c.embedding is not None
            ]

            n = len(valid_chunks)
            for i in tqdm(range(n), desc="Building similarity graph"):
                for j in range(i + 1, n):
                    similarity = self._cosine_similarity(
                        valid_chunks[i].embedding, valid_chunks[j].embedding
                    )

                    if similarity >= self.similarity_threshold:
                        similarity_graph[i].append((j, similarity))
                        similarity_graph[j].append((i, similarity))

            # Find connected components
            visited = set()
            clusters = []

            for i in range(n):
                if i in visited:
                    continue

                cluster = []
                queue = [i]
                visited.add(i)

                while queue:
                    node = queue.pop(0)
                    cluster.append(valid_chunks[node])

                    for neighbor, _ in similarity_graph[node]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)

                if len(cluster) > 1:
                    clusters.append(cluster)

            logger.info(f"Found {len(clusters)} clusters")
            return clusters

        except Exception as e:
            logger.error(f"Error finding similar chunks: {str(e)}")
            return []

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate cosine similarity."""
        try:
            dot_product = np.dot(v1, v2)
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)

            if norm_v1 == 0 or norm_v2 == 0:
                return 0.0

            return float(dot_product / (norm_v1 * norm_v2))

        except Exception as e:
            logger.error(f"Similarity calculation error: {str(e)}")
            return 0.0

    def _get_unique_sentences(
        self, primary_text: str, secondary_text: str
    ) -> List[str]:
        """Extract unique sentences."""
        try:
            primary_sentences = sent_tokenize(primary_text)
            secondary_sentences = sent_tokenize(secondary_text)

            primary_set = {s.lower().strip() for s in primary_sentences}
            unique_sentences = []

            for sentence in secondary_sentences:
                sentence_lower = sentence.lower().strip()

                if sentence_lower not in primary_set:
                    is_unique = True
                    for primary_sentence in primary_set:
                        primary_words = set(primary_sentence.split())
                        secondary_words = set(sentence_lower.split())

                        if len(primary_words) > 0 and len(secondary_words) > 0:
                            overlap = len(
                                primary_words.intersection(secondary_words)
                            ) / max(len(primary_words), len(secondary_words))

                            if overlap > 0.8:
                                is_unique = False
                                break

                    if is_unique:
                        unique_sentences.append(sentence)

            return unique_sentences

        except Exception as e:
            logger.error(f"Sentence extraction error: {str(e)}")
            return []

    def _merge_chunks(self, chunks: List[Chunk]) -> Chunk:
        """Merge similar chunks."""
        try:
            if not chunks:
                return None

            primary_chunk = max(chunks, key=lambda c: c.information_score)
            primary_chunk.is_primary = True

            merged_text = primary_chunk.text
            merged_from = [chunk.chunk_id for chunk in chunks]
            source_files = {
                chunk.source_file for chunk in chunks if chunk.source_file
            }

            for chunk in chunks:
                if chunk.chunk_id == primary_chunk.chunk_id:
                    continue

                unique_sentences = self._get_unique_sentences(
                    merged_text, chunk.text
                )

                if unique_sentences:
                    merged_text += "\n\n--- Additional Information ---\n"
                    merged_text += " ".join(unique_sentences)

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

        except Exception as e:
            logger.error(f"Chunk merging error: {str(e)}")
            return chunks[0] if chunks else None

    def _merge_similar_clusters(
        self, clusters: List[List[Chunk]]
    ) -> List[Chunk]:
        """Merge all clusters."""
        merged_chunks = []

        for cluster in tqdm(clusters, desc="Merging clusters"):
            try:
                merged_chunk = self._merge_chunks(cluster)
                if merged_chunk:
                    merged_chunks.append(merged_chunk)
            except Exception as e:
                logger.error(f"Cluster merge failed: {str(e)}")
                continue

        return merged_chunks

    def _generate_deduplication_report(self) -> None:
        """Generate deduplication report."""
        try:
            total_original_text = sum(
                len(chunk.text) for chunk in self.all_chunks
            )
            total_deduplicated_text = sum(
                len(chunk.text) for chunk in self.deduplicated_chunks
            )
            text_reduction = total_original_text - total_deduplicated_text

            merged_chunks = [
                chunk
                for chunk in self.deduplicated_chunks
                if chunk.merged_from and len(chunk.merged_from) > 1
            ]

            report = {
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
                        "source_files": chunk.metadata.get(
                            "merged_from_files", []
                        ),
                        "information_score": chunk.information_score,
                    }
                    for chunk in merged_chunks
                ],
            }

            report_path = self.dedup_dir / "deduplication_report.json"
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)

            # Generate summary
            summary_path = self.dedup_dir / "deduplication_summary.txt"
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write("Smart Chunk Deduplication Summary\n")
                f.write("===============================\n\n")
                f.write(
                    f"Chunk Size: {settings.chunking.chunk_size}, Overlap: {settings.chunking.chunk_overlap}\n"
                )
                f.write(
                    f"Similarity Threshold: {self.similarity_threshold}\n\n"
                )
                f.write(f"Original Chunks: {len(self.all_chunks)}\n")
                f.write(
                    f"Deduplicated Chunks: {len(self.deduplicated_chunks)}\n"
                )
                f.write(f"Merged Chunks: {len(merged_chunks)}\n")
                f.write(
                    f"Unchanged Chunks: {len(self.deduplicated_chunks) - len(merged_chunks)}\n\n"
                )
                f.write(
                    f"Text Reduction: {text_reduction} characters ({report['stats']['reduction_percentage']:.2f}%)\n\n"
                )

                if merged_chunks:
                    f.write("Top Merged Chunks (by information score):\n")
                    top_merged = sorted(
                        merged_chunks,
                        key=lambda c: c.information_score,
                        reverse=True,
                    )[:10]
                    for i, chunk in enumerate(top_merged, 1):
                        source_files = chunk.metadata.get(
                            "merged_from_files", []
                        )
                        f.write(
                            f"  {i}. Merged from {len(chunk.merged_from)} chunks across {len(source_files)} files\n"
                        )
                        f.write(
                            f"     Information score: {chunk.information_score:.4f}\n"
                        )
                        f.write(
                            f"     First 100 chars: {chunk.text[:100]}...\n\n"
                        )

            logger.info(f"Deduplication reports saved to {self.dedup_dir}")

        except Exception as e:
            logger.error(f"Failed to generate deduplication report: {str(e)}")
            # Don't raise - this is not critical
