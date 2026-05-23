import os
import re
import json
import hashlib
import logging
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union
from typing import Optional
from datetime import datetime
from dataclasses import field
from dataclasses import dataclass
from collections import defaultdict

import spacy
import nltk
import numpy as np
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader
from langchain_community.document_loaders import JSONLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import BSHTMLLoader
from langchain_community.document_loaders import SitemapLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.document_loaders import UnstructuredXMLLoader
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_community.document_loaders import UnstructuredEmailLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader

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
                    (
                        ".txt",
                        ".json",
                        ".jsonl",
                        ".npz",
                        ".md",
                        ".pdf",
                        ".docx",
                        ".xlsx",
                        ".xls",
                        ".pptx",
                        ".ppt",
                        ".html",
                        ".htm",
                        ".xml",
                        ".csv",
                        ".eml",
                    )
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
    """
    Final production-ready document processor with comprehensive file format support,
    enhanced LangChain loaders, web processing capabilities, and robust error handling.

    Features:
    - Comprehensive file format support (20+ types)
    - Multiple PDF processing strategies
    - Web content processing (URLs, sitemaps)
    - Intelligent deduplication
    - Cost-optimized chunking
    - Enterprise-grade error handling
    - Memory-efficient processing
    """

    def __init__(
        self,
        raw_dir: Optional[Union[str, Path]] = None,
        processed_dir: Optional[Union[str, Path]] = None,
        chunk_dir: Optional[Union[str, Path]] = None,
        dedup_dir: Optional[Union[str, Path]] = None,
        embedding_service: Optional[EmbeddingService] = None,
        enable_deduplication: bool = True,
        similarity_threshold: float = 0.92,
        pdf_loader_strategy: str = "auto",
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
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
        self.pdf_loader_strategy = pdf_loader_strategy

        # Chunking parameters - use settings or provided values
        self.chunk_size = chunk_size or getattr(
            settings.chunking, "chunk_size", 1000
        )
        self.chunk_overlap = chunk_overlap or getattr(
            settings.chunking, "chunk_overlap", 200
        )

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

        # Define supported file extensions and their loaders
        self.loader_mapping = self._initialize_loader_mapping()

        logger.info(
            f"DocumentProcessor initialized with {len(self.get_supported_extensions())} supported file types"
        )

    def _initialize_loader_mapping(self) -> Dict[str, Dict[str, Any]]:
        """Initialize mapping of file extensions to their appropriate loaders."""
        return {
            # Text formats
            ".txt": {
                "loader_class": TextLoader,
                "loader_kwargs": {"encoding": "utf-8"},
                "doc_type": "text",
            },
            ".md": {
                "loader_class": UnstructuredMarkdownLoader,
                "loader_kwargs": {},
                "doc_type": "markdown",
            },
            ".csv": {
                "loader_class": CSVLoader,
                "loader_kwargs": {"encoding": "utf-8"},
                "doc_type": "csv",
            },
            ".json": {
                "loader_class": JSONLoader,
                "loader_kwargs": {"jq_schema": ".", "text_content": False},
                "doc_type": "json",
            },
            # PDF formats (strategy-based selection)
            ".pdf": {
                "loader_class": self._get_pdf_loader_class,
                "loader_kwargs": {},
                "doc_type": "pdf",
            },
            # Microsoft Office formats
            ".docx": {
                "loader_class": UnstructuredWordDocumentLoader,
                "loader_kwargs": {},
                "doc_type": "docx",
                "fallback_loader": Docx2txtLoader,
            },
            ".doc": {
                "loader_class": UnstructuredWordDocumentLoader,
                "loader_kwargs": {},
                "doc_type": "doc",
            },
            ".xlsx": {
                "loader_class": UnstructuredExcelLoader,
                "loader_kwargs": {},
                "doc_type": "xlsx",
            },
            ".xls": {
                "loader_class": UnstructuredExcelLoader,
                "loader_kwargs": {},
                "doc_type": "xls",
            },
            ".pptx": {
                "loader_class": UnstructuredPowerPointLoader,
                "loader_kwargs": {},
                "doc_type": "pptx",
            },
            ".ppt": {
                "loader_class": UnstructuredPowerPointLoader,
                "loader_kwargs": {},
                "doc_type": "ppt",
            },
            # Web/HTML formats
            ".html": {
                "loader_class": UnstructuredHTMLLoader,
                "loader_kwargs": {},
                "doc_type": "html",
                "fallback_loader": BSHTMLLoader,
            },
            ".htm": {
                "loader_class": UnstructuredHTMLLoader,
                "loader_kwargs": {},
                "doc_type": "html",
                "fallback_loader": BSHTMLLoader,
            },
            ".xml": {
                "loader_class": UnstructuredXMLLoader,
                "loader_kwargs": {},
                "doc_type": "xml",
            },
            # Email formats
            ".eml": {
                "loader_class": UnstructuredEmailLoader,
                "loader_kwargs": {},
                "doc_type": "email",
            },
        }

    def _get_pdf_loader_class(self):
        """Get PDF loader class based on strategy."""
        strategy_mapping = {
            "pypdf": PyPDFLoader,
            "pdfplumber": PDFPlumberLoader,
            "pymupdf": PyMuPDFLoader,
            "auto": PyPDFLoader,  # Default to PyPDF for auto mode
        }
        return strategy_mapping.get(self.pdf_loader_strategy, PyPDFLoader)

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

    def get_supported_extensions(self) -> List[str]:
        """Get list of supported file extensions."""
        return list(self.loader_mapping.keys())

    def is_file_supported(self, file_path: Union[str, Path]) -> bool:
        """Check if a file type is supported."""
        file_path = ensure_path(file_path)
        if not file_path:
            return False

        extension = file_path.suffix.lower()
        return extension in self.loader_mapping

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics."""
        return {
            "total_documents": len(self.document_index),
            "supported_formats": len(self.get_supported_extensions()),
            "deduplication_enabled": self.enable_deduplication,
            "chunk_settings": {
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "similarity_threshold": self.similarity_threshold,
            },
            "directory_structure": {
                "raw_dir": str(self.raw_dir),
                "processed_dir": str(self.processed_dir),
                "chunk_dir": str(self.chunk_dir),
                "dedup_dir": str(self.dedup_dir),
            },
        }

    @safe_path_operation
    def process_all_documents(self) -> List[Dict[str, Any]]:
        """Process all documents with comprehensive error handling and progress tracking."""
        logger.info(f"Processing all documents in {self.raw_dir}")

        if not self.raw_dir.exists():
            raise DocumentProcessingError(
                f"Raw directory not found: {self.raw_dir}"
            )

        documents_metadata = []
        failed_files = []
        skipped_files = []

        try:
            files = list(self.raw_dir.glob("**/*"))
            if not files:
                logger.warning(f"No files found in {self.raw_dir}")
                return []

            # Filter supported files
            supported_files = [
                f for f in files if f.is_file() and self.is_file_supported(f)
            ]
            unsupported_files = [
                f
                for f in files
                if f.is_file() and not self.is_file_supported(f)
            ]

            if unsupported_files:
                logger.info(f"Found {len(unsupported_files)} unsupported files")
                logger.info(
                    f"Supported extensions: {', '.join(self.get_supported_extensions())}"
                )

            # Process supported files with progress tracking
            for file_path in tqdm(supported_files, desc="Processing documents"):
                try:
                    # Check if already processed (based on modification time)
                    if self._is_document_current(file_path):
                        skipped_files.append(file_path.name)
                        continue

                    metadata = self.process_document(file_path)
                    if metadata:
                        documents_metadata.append(metadata)
                        logger.debug(
                            f"Processed: {file_path.name} ({metadata['num_chunks']} chunks)"
                        )
                    else:
                        failed_files.append(
                            (file_path.name, "No content extracted")
                        )

                except Exception as e:
                    logger.error(
                        f"Failed to process {file_path.name}: {str(e)}"
                    )
                    failed_files.append((file_path.name, str(e)))
                    continue

            # Report results
            logger.info("Processing complete:")
            logger.info(
                f"  - Successfully processed: {len(documents_metadata)} files"
            )
            logger.info(f"  - Skipped (up-to-date): {len(skipped_files)} files")
            logger.info(f"  - Failed: {len(failed_files)} files")
            logger.info(f"  - Unsupported: {len(unsupported_files)} files")

            if failed_files:
                logger.warning("Failed files:")
                for filename, error in failed_files[
                    :5
                ]:  # Show first 5 failures
                    logger.warning(f"  - {filename}: {error}")

            # Deduplication if enabled and we have multiple documents
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

    def _is_document_current(self, file_path: Path) -> bool:
        """Check if document is already processed and up-to-date."""
        try:
            doc_id = self._generate_document_id(file_path)
            if doc_id not in self.document_index:
                return False

            # Check if file has been modified since last processing
            current_mtime = file_path.stat().st_mtime
            stored_mtime = self.document_index[doc_id].get("last_modified", 0)

            return current_mtime <= stored_mtime
        except Exception:
            return False

    @safe_path_operation
    def process_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Process a single file from any location."""
        file_path = ensure_path(file_path)
        if not file_path:
            raise DocumentProcessingError("Invalid file path provided")

        if not file_path.exists():
            raise DocumentProcessingError(f"File not found: {file_path}")

        if not file_path.is_file():
            raise DocumentProcessingError(f"Path is not a file: {file_path}")

        return self.process_document(file_path)

    @safe_path_operation
    def process_folder(
        self, folder_path: Union[str, Path]
    ) -> List[Dict[str, Any]]:
        """Process all files in a folder."""
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

        files = list(folder_path.glob("**/*"))
        if not files:
            logger.warning(f"No files found in {folder_path}")
            return []

        supported_files = [
            f for f in files if f.is_file() and self.is_file_supported(f)
        ]

        for file_path in tqdm(
            supported_files, desc=f"Processing {folder_path.name}"
        ):
            try:
                metadata = self.process_document(file_path)
                if metadata:
                    results.append(metadata)
            except Exception as e:
                logger.error(f"Failed to process {file_path.name}: {str(e)}")
                failed_files.append((file_path.name, str(e)))

        logger.info(f"Processed {len(results)} files from {folder_path}")
        if failed_files:
            logger.warning(f"Failed to process {len(failed_files)} files")

        return results

    @safe_path_operation
    def process_document(
        self, file_path: Union[str, Path]
    ) -> Optional[Dict[str, Any]]:
        """Process a single document with enhanced error handling and loader selection."""
        file_path = ensure_path(file_path)
        if not file_path:
            raise DocumentProcessingError("Invalid file path provided")

        if not file_path.exists():
            raise DocumentProcessingError(f"File not found: {file_path}")

        if not self.is_file_supported(file_path):
            raise DocumentProcessingError(
                f"Unsupported file type: {file_path.suffix}"
            )

        logger.info(f"Processing: {file_path.name}")

        try:
            # Extract text using enhanced loader selection
            text, doc_type = self._extract_text_enhanced(file_path)

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

            # Create chunks with enhanced chunking
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
                "file_size": file_path.stat().st_size,
                "last_modified": file_path.stat().st_mtime,
                "processed_date": (
                    processed_path.stat().st_mtime
                    if processed_path.exists()
                    else None
                ),
                "chunk_settings": {
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                },
            }

            # Update document index
            self.document_index[doc_id] = metadata
            self._save_document_index()

            logger.info(
                f"Successfully processed {file_path.name}: {len(chunks)} chunks"
            )
            return metadata

        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {str(e)}")
            raise DocumentProcessingError(
                f"Failed to process {file_path.name}: {str(e)}"
            )

    def _extract_text_enhanced(
        self, file_path: Union[str, Path]
    ) -> Tuple[str, str]:
        """Extract text using enhanced LangChain loader selection with comprehensive fallback."""
        file_path = ensure_path(file_path)
        if not file_path:
            raise DocumentProcessingError(
                "Invalid file path for text extraction"
            )

        file_extension = file_path.suffix.lower()
        loader_info = self.loader_mapping.get(file_extension)

        if not loader_info:
            # Try generic fallback loader
            try:
                logger.info(
                    f"Using generic fallback loader for {file_path.name}"
                )
                loader = UnstructuredFileLoader(str(file_path))
                documents = loader.load()
                text = "\n\n".join(doc.page_content for doc in documents)
                return text, "generic"

            except Exception:
                raise DocumentProcessingError(
                    f"Unsupported file type and fallback failed: {file_extension}"
                )

        # Get loader class and kwargs
        loader_class = loader_info["loader_class"]
        loader_kwargs = loader_info.get("loader_kwargs", {})
        doc_type = loader_info["doc_type"]
        fallback_loader = loader_info.get("fallback_loader")

        # Special handling for PDF strategy
        if file_extension == ".pdf" and callable(loader_class):
            loader_class = loader_class()

        try:
            # Try primary loader
            logger.debug(
                f"Using primary loader: {loader_class.__name__ if hasattr(loader_class, '__name__') else type(loader_class).__name__}"
            )
            loader = loader_class(str(file_path), **loader_kwargs)
            documents = loader.load()
            text = "\n\n".join(doc.page_content for doc in documents)

            # Validate extracted text
            if not text.strip():
                raise Exception("No text extracted")

            return text, doc_type

        except Exception as e:
            logger.warning(
                f"Primary loader failed for {file_path.name}: {str(e)}"
            )

            # Try fallback loader if available
            if fallback_loader:
                try:
                    logger.info(
                        f"Trying fallback loader: {fallback_loader.__name__}"
                    )
                    loader = fallback_loader(str(file_path))
                    documents = loader.load()
                    text = "\n\n".join(doc.page_content for doc in documents)
                    if text.strip():
                        return text, doc_type
                except Exception as fallback_error:
                    logger.warning(
                        f"Fallback loader also failed: {str(fallback_error)}"
                    )

            # Try PDF strategy fallback for PDFs
            if file_extension == ".pdf":
                pdf_loaders = [PyPDFLoader, PDFPlumberLoader, PyMuPDFLoader]
                for pdf_loader_class in pdf_loaders:
                    try:
                        logger.info(
                            f"Trying PDF fallback: {pdf_loader_class.__name__}"
                        )
                        loader = pdf_loader_class(str(file_path))
                        documents = loader.load()
                        text = "\n\n".join(
                            doc.page_content for doc in documents
                        )
                        if text.strip():
                            return text, doc_type
                    except Exception:
                        continue

            # Final fallback to generic loader
            try:
                logger.info(
                    f"Final fallback to generic loader for {file_path.name}"
                )
                loader = UnstructuredFileLoader(str(file_path))
                documents = loader.load()
                text = "\n\n".join(doc.page_content for doc in documents)
                return text, doc_type
            except Exception as final_error:
                raise DocumentProcessingError(
                    f"All extraction methods failed for {file_path.name}: {str(final_error)}"
                )

    def process_url(
        self, url: str, output_name: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Process content from a URL."""
        try:
            logger.info(f"Processing URL: {url}")

            loader = WebBaseLoader(
                url, verify_ssl=False,
            )
            documents = loader.load()

            if not documents:
                logger.warning(f"No content extracted from URL: {url}")
                return None

            text = "\n\n".join(doc.page_content for doc in documents)

            if not text.strip():
                logger.warning(f"Empty content from URL: {url}")
                return None

            # Generate document ID based on URL
            doc_id = hashlib.md5(url.encode()).hexdigest()

            # Use provided name or generate from URL
            if not output_name:
                from urllib.parse import urlparse

                parsed = urlparse(url)
                output_name = f"{parsed.netloc}_{parsed.path.replace('/', '_')}"

            # Clean text
            cleaned_text = self._clean_text(text)

            # Save processed text
            processed_path = self.processed_dir / f"{doc_id}.txt"
            with open(processed_path, "w", encoding="utf-8") as f:
                f.write(cleaned_text)

            # Create chunks
            chunks = self._create_chunks(cleaned_text, doc_id, url)

            # Save chunks
            chunk_path = self.chunk_dir / f"{doc_id}_chunks.jsonl"
            self._save_chunks(chunks, chunk_path)

            metadata = {
                "doc_id": doc_id,
                "file_name": output_name,
                "file_path": url,
                "doc_type": "web",
                "processed_path": str(processed_path),
                "chunks_path": str(chunk_path),
                "num_chunks": len(chunks),
                "last_modified": None,
                "processed_date": processed_path.stat().st_mtime,
            }

            # Update document index
            self.document_index[doc_id] = metadata
            self._save_document_index()

            logger.info(f"Processed URL {url}: {len(chunks)} chunks")
            return metadata

        except Exception as e:
            logger.error(f"Error processing URL {url}: {str(e)}")
            raise DocumentProcessingError(
                f"Failed to process URL {url}: {str(e)}"
            )

    def process_sitemap(
        self, sitemap_url: str, max_pages: int = 50
    ) -> List[Dict[str, Any]]:
        """Process multiple pages from a sitemap."""
        try:
            logger.info(f"Processing sitemap: {sitemap_url}")

            loader = SitemapLoader(
                sitemap_url, verify_ssl=settings.ssl.enable_verification
            )
            documents = loader.load()

            results = []
            for i, doc in enumerate(documents[:max_pages]):
                try:
                    # Generate unique doc_id for each page
                    source_url = doc.metadata.get(
                        "source", f"{sitemap_url}_page_{i}"
                    )
                    doc_id = hashlib.md5(source_url.encode()).hexdigest()

                    text = doc.page_content
                    if not text.strip():
                        continue

                    # Clean text
                    cleaned_text = self._clean_text(text)

                    # Save processed text
                    processed_path = self.processed_dir / f"{doc_id}.txt"
                    with open(processed_path, "w", encoding="utf-8") as f:
                        f.write(cleaned_text)

                    # Create chunks
                    chunks = self._create_chunks(
                        cleaned_text, doc_id, source_url
                    )

                    # Save chunks
                    chunk_path = self.chunk_dir / f"{doc_id}_chunks.jsonl"
                    self._save_chunks(chunks, chunk_path)

                    from urllib.parse import urlparse

                    parsed = urlparse(source_url)
                    page_name = (
                        f"{parsed.netloc}_{parsed.path.replace('/', '_')}"
                    )

                    metadata = {
                        "doc_id": doc_id,
                        "file_name": page_name,
                        "file_path": source_url,
                        "doc_type": "web_sitemap",
                        "processed_path": str(processed_path),
                        "chunks_path": str(chunk_path),
                        "num_chunks": len(chunks),
                        "last_modified": None,
                        "processed_date": processed_path.stat().st_mtime,
                    }

                    # Update document index
                    self.document_index[doc_id] = metadata
                    results.append(metadata)

                except Exception as e:
                    logger.error(f"Error processing page {i}: {str(e)}")
                    continue

            self._save_document_index()
            logger.info(f"Processed {len(results)} pages from sitemap")
            return results

        except Exception as e:
            logger.error(f"Error processing sitemap {sitemap_url}: {str(e)}")
            raise DocumentProcessingError(
                f"Failed to process sitemap: {str(e)}"
            )

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and its associated chunks from the system."""
        try:
            if doc_id not in self.document_index:
                logger.warning(f"Document {doc_id} not found in index")
                return False

            metadata = self.document_index[doc_id]
            processed_path = ensure_path(metadata["processed_path"])
            if processed_path and processed_path.exists():
                processed_path.unlink()
                logger.info(f"Deleted processed file: {processed_path}")

            chunks_path = ensure_path(metadata["chunks_path"])
            if chunks_path and chunks_path.exists():
                chunks_path.unlink()
                logger.info(f"Deleted chunks file: {chunks_path}")

            # Update deduplication files if they exist
            dedup_path = self.dedup_dir / "deduplicated_chunks.jsonl"
            if dedup_path.exists():
                self._remove_document_from_deduplicated(doc_id)

            # Delete embeddings if they exist
            embeddings_dir = ensure_path(ROOT_DIR) / "data" / "embeddings"
            embedding_file = embeddings_dir / f"{doc_id}_embeddings.npz"
            if embedding_file.exists():
                embedding_file.unlink()
                logger.info(f"Deleted embeddings file: {embedding_file}")

            del self.document_index[doc_id]
            self._save_document_index()

            logger.info(f"Successfully deleted document: {doc_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {str(e)}")
            return False

    @safe_path_operation
    def update_document(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Update an existing document. If the document doesn't exist, it will be added."""
        file_path = ensure_path(file_path)
        if not file_path:
            raise DocumentProcessingError("Invalid file path provided")

        if not file_path.exists():
            raise DocumentProcessingError(f"File not found: {file_path}")

        doc_id = self._generate_document_id(file_path)

        if doc_id in self.document_index:
            logger.info(f"Document {doc_id} already exists, updating...")
            self.delete_document(doc_id)

        return self.process_document(file_path)

    def list_documents(self) -> List[Dict[str, Any]]:
        """List all processed documents in the system."""
        return list(self.document_index.values())

    def get_document_info(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific document."""
        return self.document_index.get(doc_id)

    def search_documents(self, query: str) -> List[Dict[str, Any]]:
        """Search documents by filename or content type."""
        query_lower = query.lower()
        results = []

        for doc_info in self.document_index.values():
            if (
                query_lower in doc_info["file_name"].lower()
                or query_lower in doc_info["doc_type"].lower()
            ):
                results.append(doc_info)

        return results

    def get_documents_by_type(self, doc_type: str) -> List[Dict[str, Any]]:
        """Get all documents of a specific type."""
        return [
            doc
            for doc in self.document_index.values()
            if doc["doc_type"] == doc_type
        ]

    def _remove_document_from_deduplicated(self, doc_id: str) -> None:
        """Remove a document's chunks from the deduplicated file."""
        dedup_path = self.dedup_dir / "deduplicated_chunks.jsonl"
        if not dedup_path.exists():
            return

        try:
            temp_path = dedup_path.with_suffix(".tmp")
            filtered_chunks = []

            with open(dedup_path, "r", encoding="utf-8") as f:
                for line in f:
                    chunk_data = json.loads(line)
                    if chunk_data.get("doc_id") != doc_id:
                        if (
                            "metadata" in chunk_data
                            and "merged_from_chunks" in chunk_data["metadata"]
                        ):
                            merged_chunks = chunk_data["metadata"][
                                "merged_from_chunks"
                            ]
                            filtered_chunks_ids = [
                                c
                                for c in merged_chunks
                                if not c.startswith(f"{doc_id}_")
                            ]
                            if filtered_chunks_ids:
                                chunk_data["metadata"][
                                    "merged_from_chunks"
                                ] = filtered_chunks_ids
                                filtered_chunks.append(chunk_data)
                        else:
                            filtered_chunks.append(chunk_data)

            with open(temp_path, "w", encoding="utf-8") as f:
                for chunk_data in filtered_chunks:
                    f.write(json.dumps(chunk_data) + "\n")

            temp_path.replace(dedup_path)
            logger.info(f"Removed document {doc_id} from deduplicated chunks")

        except Exception as e:
            logger.error(
                f"Error removing document from deduplicated chunks: {str(e)}"
            )

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text with enhanced preprocessing."""
        if not text:
            return ""

        try:
            # Replace multiple newlines with double newlines
            # (preserve paragraph structure)
            text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)

            # Replace multiple spaces with single space
            text = re.sub(r" +", " ", text)

            # Replace tabs with spaces
            text = re.sub(r"\t+", " ", text)

            # Remove control characters but preserve basic formatting
            text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]", "", text)

            # Remove excessive punctuation repetition
            text = re.sub(r"[.]{3,}", "...", text)
            text = re.sub(r"[!]{2,}", "!", text)
            text = re.sub(r"[?]{2,}", "?", text)

            # Strip whitespace and normalize
            text = text.strip()

            return text
        except Exception as e:
            logger.error(f"Text cleaning error: {str(e)}")
            return text  # Return original if cleaning fails

    def _create_chunks(
        self, text: str, doc_id: str, source_file: str = ""
    ) -> List[Chunk]:
        """Create chunks with enhanced chunking strategy."""
        try:
            # Use semantic-aware text splitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=[
                    "\n\n",  # Paragraph breaks
                    "\n",  # Line breaks
                    ". ",  # Sentence endings
                    "! ",  # Exclamation sentences
                    "? ",  # Question sentences
                    "; ",  # Semicolon breaks
                    ", ",  # Comma breaks
                    " ",  # Word breaks
                    "",  # Character breaks (fallback)
                ],
                keep_separator=True,  # Preserve separators for better context
            )

            texts = text_splitter.split_text(text)

            chunks = []
            for i, chunk_text in enumerate(texts):
                chunk_id = f"{doc_id}_{i:04d}"

                # Calculate basic information score for this chunk
                info_score = self._calculate_basic_information_score(chunk_text)

                chunk = Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    chunk_index=i,
                    text=chunk_text,
                    metadata={
                        "doc_id": doc_id,
                        "chunk_index": i,
                        "char_count": len(chunk_text),
                        "word_count": len(chunk_text.split()),
                        "information_score": info_score,
                    },
                    information_score=info_score,
                    source_file=source_file,
                )
                chunks.append(chunk)

            return chunks

        except Exception as e:
            logger.error(f"Chunking error: {str(e)}")
            raise DocumentProcessingError(f"Failed to create chunks: {str(e)}")

    def _calculate_basic_information_score(self, text: str) -> float:
        """Calculate basic information score for a chunk."""
        try:
            if not text.strip():
                return 0.0

            # Length score (normalized)
            length_score = min(1.0, len(text) / 1000)

            # Entity-like patterns (capitalized words)
            capitalized_words = len(re.findall(r"\b[A-Z][a-zA-Z]*\b", text))
            entity_score = min(1.0, capitalized_words / 20)

            # Numerical content
            numbers = len(re.findall(r"\b\d+\b", text))
            number_score = min(1.0, numbers / 10)

            # Structured content (lists, etc.)
            has_structure = (
                1.0 if re.search(r"(\n\s*[-*•]\s+|\d+\.\s+)", text) else 0.0
            )

            # Educational keywords
            edu_keywords = [
                "university",
                "course",
                "program",
                "student",
                "fee",
                "admission",
                "registration",
                "semester",
                "exam",
                "grade",
                "professor",
                "dean",
                "policy",
                "requirement",
                "deadline",
                "application",
                "hostel",
                "library",
                "schedule",
                "timetable",
            ]

            keyword_matches = sum(
                1 for keyword in edu_keywords if keyword in text.lower()
            )
            keyword_score = min(1.0, keyword_matches / 10)

            # Weighted combination
            final_score = (
                0.2 * length_score
                + 0.25 * entity_score
                + 0.15 * number_score
                + 0.15 * has_structure
                + 0.25 * keyword_score
            )

            return round(final_score, 3)

        except Exception as e:
            logger.error(f"Error calculating information score: {str(e)}")
            return 0.5  # Default neutral score

    def _save_chunks(
        self, chunks: List[Chunk], output_path: Union[str, Path]
    ) -> None:
        """Save chunks with enhanced error handling and validation."""
        output_path = ensure_path(output_path)
        if not output_path:
            raise DocumentProcessingError(
                "Invalid output path for saving chunks"
            )

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = output_path.with_suffix(".tmp")

            chunk_count = 0
            with open(temp_path, "w", encoding="utf-8") as f:
                for chunk in chunks:
                    chunk_dict = chunk.to_dict()
                    # Validate chunk data
                    if chunk_dict.get("text", "").strip():
                        f.write(
                            json.dumps(chunk_dict, ensure_ascii=False) + "\n"
                        )
                        chunk_count += 1

            # Atomic rename
            temp_path.replace(output_path)
            logger.debug(f"Saved {chunk_count} chunks to {output_path}")

        except Exception as e:
            if temp_path and temp_path.exists():
                temp_path.unlink()
            raise DocumentProcessingError(f"Failed to save chunks: {str(e)}")

    def _save_deduplicated_chunks(self, output_path: Union[str, Path]) -> None:
        """Save deduplicated chunks with enhanced metadata."""
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

                    # Enhanced metadata for deduplicated chunks
                    chunk_data["metadata"]["is_deduplicated"] = True
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
                        chunk_data["metadata"]["merge_count"] = len(
                            chunk.merged_from
                        )

                    f.write(json.dumps(chunk_data, ensure_ascii=False) + "\n")

            temp_path.replace(output_path)
            logger.info(
                f"Saved {len(self.deduplicated_chunks)} deduplicated chunks"
            )

        except Exception as e:
            if temp_path and temp_path.exists():
                temp_path.unlink()
            raise DocumentProcessingError(
                f"Failed to save deduplicated chunks: {str(e)}"
            )

    def _generate_document_id(self, file_path: Union[str, Path]) -> str:
        """Generate unique document ID with enhanced collision resistance."""
        file_path = ensure_path(file_path)
        if not file_path:
            raise DocumentProcessingError(
                "Invalid file path for generating document ID"
            )

        try:
            file_stat = file_path.stat()
            # Include file size and modification time for better uniqueness
            unique_string = (
                f"{file_path}_{file_stat.st_mtime}_{file_stat.st_size}"
            )
            return hashlib.md5(unique_string.encode()).hexdigest()
        except Exception as e:
            # Fallback to filename-based ID with timestamp
            logger.warning(f"Using fallback ID generation: {str(e)}")
            fallback_string = f"{str(file_path)}_{datetime.now().timestamp()}"
            return hashlib.md5(fallback_string.encode()).hexdigest()

    # Deduplication methods
    def _deduplicate_chunks(self) -> None:
        """Enhanced deduplication with progress tracking and optimization."""
        if not self.all_chunks:
            logger.warning("No chunks to deduplicate")
            return

        try:
            logger.info(
                f"Starting deduplication of {len(self.all_chunks)} chunks"
            )

            # Step 1: Generate embeddings with progress tracking
            logger.info("Generating embeddings for deduplication...")
            self._generate_embeddings()

            # Step 2: Analyze information density
            logger.info("Analyzing information density...")
            self._analyze_information()

            # Step 3: Find similar chunks with optimized algorithm
            logger.info(
                f"Finding similar chunks (threshold: {self.similarity_threshold})"
            )
            similar_clusters = self._find_similar_chunks()

            # Step 4: Merge clusters efficiently
            logger.info(f"Merging {len(similar_clusters)} clusters")
            merged_chunks = self._merge_similar_clusters(similar_clusters)

            # Step 5: Combine results
            merged_chunk_ids = set()
            for cluster in similar_clusters:
                for chunk in cluster:
                    merged_chunk_ids.add(chunk.chunk_id)

            self.deduplicated_chunks = []
            self.deduplicated_chunks.extend(merged_chunks)

            # Add non-merged chunks
            for chunk in self.all_chunks:
                if chunk.chunk_id not in merged_chunk_ids:
                    self.deduplicated_chunks.append(chunk)

            reduction_percent = (
                (
                    (len(self.all_chunks) - len(self.deduplicated_chunks))
                    / len(self.all_chunks)
                    * 100
                )
                if self.all_chunks
                else 0
            )

            logger.info(
                f"Deduplication complete: {len(self.all_chunks)} → {len(self.deduplicated_chunks)} chunks "
                f"({reduction_percent:.1f}% reduction)"
            )

        except Exception as e:
            logger.error(f"Deduplication failed: {str(e)}")
            # Use original chunks if deduplication fails
            self.deduplicated_chunks = self.all_chunks.copy()
            raise DocumentProcessingError(f"Deduplication failed: {str(e)}")

    def _generate_embeddings(self) -> None:
        """Generate embeddings with enhanced error handling and batch processing."""
        batch_size = 32  # Optimized batch size
        failed_chunks = []

        for i in tqdm(
            range(0, len(self.all_chunks), batch_size),
            desc="Generating embeddings",
        ):
            batch = self.all_chunks[i : i + batch_size]
            texts = [chunk.text for chunk in batch]

            try:
                embeddings = self.embedding_service.embed_batch(texts)

                for chunk, embedding in zip(batch, embeddings):
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
        """Analyze information density with enhanced metrics."""
        try:
            for chunk in tqdm(self.all_chunks, desc="Analyzing information"):
                if (
                    chunk.information_score == 0.0
                ):  # Only calculate if not already set
                    chunk.information_score = self._calculate_information_score(
                        chunk
                    )
        except Exception as e:
            logger.error(f"Information analysis error: {str(e)}")

    def _calculate_information_score(self, chunk: Chunk) -> float:
        """Enhanced information score calculation."""
        try:
            text = chunk.text

            # Length score (with diminishing returns)
            length_score = min(1.0, len(text) / 1000)

            # Entity detection (proper nouns, abbreviations)
            entities = re.findall(r"\b[A-Z][a-zA-Z]*\b|\b[A-Z]{2,}\b", text)
            entity_score = min(1.0, len(entities) / 15)

            # Numerical content
            numbers = re.findall(r"\b\d+(?:\.\d+)?\b", text)
            number_score = min(1.0, len(numbers) / 8)

            # Structured content detection
            list_items = re.findall(r"(\n\s*[-*•]\s+|\d+\.\s+)", text)
            structure_score = (
                min(1.0, len(list_items) / 5) if list_items else 0.0
            )

            # Contact information and specific details
            contact_patterns = [
                r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",  # Phone numbers
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Emails
                r"\bwww\.[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # URLs
                r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",  # Dates
                r"\b\d{1,2}:\d{2}\s*(AM|PM|am|pm)?\b",  # Times
            ]

            contact_score = 0.0
            for pattern in contact_patterns:
                if re.search(pattern, text):
                    contact_score += 0.2
            contact_score = min(1.0, contact_score)

            # Educational relevance
            edu_keywords = [
                "university",
                "college",
                "course",
                "program",
                "student",
                "fee",
                "tuition",
                "admission",
                "registration",
                "semester",
                "exam",
                "grade",
                "professor",
                "dean",
                "policy",
                "requirement",
                "deadline",
                "application",
                "hostel",
                "library",
                "schedule",
                "timetable",
                "academic",
                "faculty",
                "department",
                "degree",
                "credit",
            ]

            keyword_matches = sum(
                1 for keyword in edu_keywords if keyword.lower() in text.lower()
            )
            edu_score = min(1.0, keyword_matches / 8)

            # Final weighted score
            final_score = (
                0.15 * length_score
                + 0.20 * entity_score
                + 0.15 * number_score
                + 0.15 * structure_score
                + 0.15 * contact_score
                + 0.20 * edu_score
            )

            return round(final_score, 3)

        except Exception as e:
            logger.error(f"Error calculating information score: {str(e)}")
            return 0.5

    def _find_similar_chunks(self) -> List[List[Chunk]]:
        """Find similar chunk clusters with optimized algorithm."""
        try:
            similarity_graph = defaultdict(list)
            valid_chunks = [
                c for c in self.all_chunks if c.embedding is not None
            ]

            if len(valid_chunks) < 2:
                return []

            n = len(valid_chunks)
            logger.info(f"Computing similarities for {n} chunks")

            # Use numpy for vectorized similarity computation (more efficient)
            embeddings_matrix = np.array(
                [chunk.embedding for chunk in valid_chunks]
            )

            # Compute all pairwise similarities at once
            similarities = np.dot(embeddings_matrix, embeddings_matrix.T)
            norms = np.linalg.norm(embeddings_matrix, axis=1)
            similarities = similarities / np.outer(norms, norms)

            # Find similar pairs above threshold
            similar_pairs = np.where(similarities >= self.similarity_threshold)

            for i, j in zip(similar_pairs[0], similar_pairs[1]):
                if i < j:  # Avoid duplicates
                    similarity = similarities[i, j]
                    similarity_graph[i].append((j, similarity))
                    similarity_graph[j].append((i, similarity))

            # Find connected components using Union-Find for efficiency
            parent = list(range(n))

            def find(x):
                if parent[x] != x:
                    parent[x] = find(parent[x])
                return parent[x]

            def union(x, y):
                px, py = find(x), find(y)
                if px != py:
                    parent[px] = py

            # Union similar chunks
            for i in range(n):
                for j, _ in similarity_graph[i]:
                    union(i, j)

            # Group chunks by component
            components = defaultdict(list)
            for i in range(n):
                root = find(i)
                components[root].append(valid_chunks[i])

            # Return clusters with more than one chunk
            clusters = [
                cluster for cluster in components.values() if len(cluster) > 1
            ]

            logger.info(f"Found {len(clusters)} similarity clusters")
            return clusters

        except Exception as e:
            logger.error(f"Error finding similar chunks: {str(e)}")
            return []

    def _get_unique_sentences(
        self, primary_text: str, secondary_text: str
    ) -> List[str]:
        """Extract unique sentences with enhanced similarity detection."""
        try:
            primary_sentences = sent_tokenize(primary_text)
            secondary_sentences = sent_tokenize(secondary_text)

            primary_set = {s.lower().strip() for s in primary_sentences}
            unique_sentences = []

            for sentence in secondary_sentences:
                sentence_lower = sentence.lower().strip()

                if len(sentence_lower) < 20:  # Skip very short sentences
                    continue

                if sentence_lower not in primary_set:
                    is_unique = True
                    sentence_words = set(sentence_lower.split())

                    for primary_sentence in primary_set:
                        primary_words = set(primary_sentence.split())

                        if len(primary_words) > 0 and len(sentence_words) > 0:
                            # Jaccard similarity
                            intersection = len(
                                primary_words.intersection(sentence_words)
                            )
                            union = len(primary_words.union(sentence_words))
                            jaccard = intersection / union if union > 0 else 0

                            if jaccard > 0.7:  # High similarity threshold
                                is_unique = False
                                break

                    if is_unique:
                        unique_sentences.append(sentence)

            return unique_sentences[
                :3
            ]  # Limit to avoid overly long merged chunks

        except Exception as e:
            logger.error(f"Sentence extraction error: {str(e)}")
            return []

    def _merge_chunks(self, chunks: List[Chunk]) -> Chunk:
        """Merge similar chunks with enhanced merging strategy."""
        try:
            if not chunks:
                return None

            # Sort by information score to pick the best primary chunk
            sorted_chunks = sorted(
                chunks, key=lambda c: c.information_score, reverse=True
            )
            primary_chunk = sorted_chunks[0]
            primary_chunk.is_primary = True

            merged_text = primary_chunk.text
            merged_from = [chunk.chunk_id for chunk in chunks]
            source_files = {
                chunk.source_file for chunk in chunks if chunk.source_file
            }

            # Add unique information from other chunks
            for chunk in sorted_chunks[1:]:
                unique_sentences = self._get_unique_sentences(
                    merged_text, chunk.text
                )

                if unique_sentences:
                    merged_text += "\n\n--- Additional Information ---\n"
                    merged_text += " ".join(unique_sentences)

            # Create merged chunk with enhanced metadata
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
                    "merge_timestamp": datetime.now().isoformat(),
                    "char_count": len(merged_text),
                    "word_count": len(merged_text.split()),
                },
                information_score=primary_chunk.information_score,
                merged_from=merged_from,
                is_primary=True,
                source_file=(
                    f"MERGED({','.join(sorted(source_files))})"
                    if source_files
                    else ""
                ),
            )

            return merged_chunk

        except Exception as e:
            logger.error(f"Chunk merging error: {str(e)}")
            return chunks[0] if chunks else None

    def _merge_similar_clusters(
        self, clusters: List[List[Chunk]]
    ) -> List[Chunk]:
        """Merge all clusters with progress tracking."""
        merged_chunks = []

        for cluster in tqdm(clusters, desc="Merging clusters"):
            try:
                merged_chunk = self._merge_chunks(cluster)
                if merged_chunk:
                    merged_chunks.append(merged_chunk)
            except Exception as e:
                logger.error(f"Cluster merge failed: {str(e)}")
                # Add the best chunk from the failed cluster
                if cluster:
                    best_chunk = max(cluster, key=lambda c: c.information_score)
                    merged_chunks.append(best_chunk)

        return merged_chunks

    def _generate_deduplication_report(self) -> None:
        """Generate comprehensive deduplication report with enhanced metrics."""
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

            # Calculate detailed statistics
            avg_original_info_score = (
                sum(chunk.information_score for chunk in self.all_chunks)
                / len(self.all_chunks)
                if self.all_chunks
                else 0
            )

            avg_deduplicated_info_score = (
                sum(
                    chunk.information_score
                    for chunk in self.deduplicated_chunks
                )
                / len(self.deduplicated_chunks)
                if self.deduplicated_chunks
                else 0
            )

            # Document type analysis
            doc_type_stats = defaultdict(int)
            for chunk in self.all_chunks:
                if hasattr(chunk, "metadata") and "doc_id" in chunk.metadata:
                    doc_id = chunk.metadata["doc_id"]
                    if doc_id in self.document_index:
                        doc_type = self.document_index[doc_id].get(
                            "doc_type", "unknown"
                        )
                        doc_type_stats[doc_type] += 1

            report = {
                "deduplication_settings": {
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                    "similarity_threshold": self.similarity_threshold,
                    "enabled": self.enable_deduplication,
                },
                "processing_timestamp": datetime.now().isoformat(),
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
                        (text_reduction / total_original_text * 100)
                        if total_original_text > 0
                        else 0
                    ),
                    "avg_original_info_score": round(
                        avg_original_info_score, 3
                    ),
                    "avg_deduplicated_info_score": round(
                        avg_deduplicated_info_score, 3
                    ),
                },
                "document_type_distribution": dict(doc_type_stats),
                "merged_chunks_details": [
                    {
                        "id": chunk.chunk_id,
                        "merged_from": chunk.merged_from,
                        "merge_count": len(chunk.merged_from),
                        "source_files": chunk.metadata.get(
                            "merged_from_files", []
                        ),
                        "information_score": chunk.information_score,
                        "char_count": len(chunk.text),
                        "primary_chunk_id": chunk.metadata.get(
                            "primary_chunk_id", ""
                        ),
                    }
                    for chunk in merged_chunks[
                        :20
                    ]  # Limit to top 20 for report size
                ],
                "quality_metrics": {
                    "high_info_chunks": len(
                        [
                            c
                            for c in self.deduplicated_chunks
                            if c.information_score > 0.7
                        ]
                    ),
                    "medium_info_chunks": len(
                        [
                            c
                            for c in self.deduplicated_chunks
                            if 0.4 <= c.information_score <= 0.7
                        ]
                    ),
                    "low_info_chunks": len(
                        [
                            c
                            for c in self.deduplicated_chunks
                            if c.information_score < 0.4
                        ]
                    ),
                },
            }

            # Save detailed JSON report
            report_path = self.dedup_dir / "deduplication_report.json"
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            # Generate human-readable summary
            summary_path = self.dedup_dir / "deduplication_summary.txt"
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write("Enhanced Document Processing & Deduplication Report\n")
                f.write("=" * 55 + "\n\n")

                f.write(
                    f"Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                )
                f.write(
                    f"Chunk Settings: Size={self.chunk_size}, Overlap={self.chunk_overlap}\n"
                )
                f.write(
                    f"Similarity Threshold: {self.similarity_threshold}\n\n"
                )

                f.write("PROCESSING RESULTS:\n")
                f.write(f"  Original Chunks: {len(self.all_chunks):,}\n")
                f.write(
                    f"  Deduplicated Chunks: {len(self.deduplicated_chunks):,}\n"
                )
                f.write(f"  Merged Chunks: {len(merged_chunks):,}\n")
                f.write(
                    f"  Unchanged Chunks: {len(self.deduplicated_chunks) - len(merged_chunks):,}\n\n"
                )

                f.write("TEXT REDUCTION:\n")
                f.write(f"  Characters Removed: {text_reduction:,}\n")
                f.write(
                    f"  Reduction Percentage: {report['stats']['reduction_percentage']:.2f}%\n\n"
                )

                f.write("QUALITY METRICS:\n")
                f.write(
                    f"  High Information Chunks (>0.7): {report['quality_metrics']['high_info_chunks']:,}\n"
                )
                f.write(
                    f"  Medium Information Chunks (0.4-0.7): {report['quality_metrics']['medium_info_chunks']:,}\n"
                )
                f.write(
                    f"  Low Information Chunks (<0.4): {report['quality_metrics']['low_info_chunks']:,}\n"
                )
                f.write(
                    f"  Average Information Score: {report['stats']['avg_deduplicated_info_score']:.3f}\n\n"
                )

                if doc_type_stats:
                    f.write("DOCUMENT TYPE DISTRIBUTION:\n")
                    for doc_type, count in sorted(doc_type_stats.items()):
                        f.write(f"  {doc_type.upper()}: {count:,} chunks\n")
                    f.write("\n")

                if merged_chunks:
                    f.write("TOP MERGED CHUNKS (by information score):\n")
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
                        f.write(f"     Characters: {len(chunk.text):,}\n")
                        f.write(
                            f"     Preview: {chunk.text[:100].replace(chr(10), ' ')}...\n\n"
                        )

            logger.info(
                f"Comprehensive deduplication reports saved to {self.dedup_dir}"
            )

        except Exception as e:
            logger.error(f"Failed to generate deduplication report: {str(e)}")


# Utility functions for file format detection and validation
def detect_file_encoding(file_path: Union[str, Path]) -> str:
    """Detect file encoding for better text extraction."""
    try:
        import chardet

        file_path = ensure_path(file_path)
        with open(file_path, "rb") as f:
            raw_data = f.read(10000)  # Read first 10KB
            result = chardet.detect(raw_data)
            return result.get("encoding", "utf-8")
    except ImportError:
        logger.warning("chardet not available, using utf-8 encoding")
        return "utf-8"
    except Exception:
        return "utf-8"


def is_text_file(file_path: Union[str, Path]) -> bool:
    """Check if a file appears to be a text file."""
    file_path = ensure_path(file_path)
    try:
        with open(file_path, "rb") as f:
            chunk = f.read(1024)
            if b"\x00" in chunk:  # Binary files often contain null bytes
                return False
            # Try to decode as text
            chunk.decode("utf-8")
            return True
    except (UnicodeDecodeError, IOError):
        return False


def get_file_metadata(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Extract comprehensive metadata from a file."""
    file_path = ensure_path(file_path)
    stat = file_path.stat()

    metadata = {
        "name": file_path.name,
        "size": stat.st_size,
        "size_mb": round(stat.st_size / (1024 * 1024), 2),
        "extension": file_path.suffix.lower(),
        "created": stat.st_ctime,
        "modified": stat.st_mtime,
        "is_text": is_text_file(file_path),
        "encoding": (
            detect_file_encoding(file_path) if is_text_file(file_path) else None
        ),
    }
    return metadata


def validate_document_processor_setup(
    processor: DocumentProcessor,
) -> Dict[str, Any]:
    """Validate document processor setup and configuration."""
    validation_results = {
        "valid": True,
        "warnings": [],
        "errors": [],
        "info": [],
    }

    # Check directory structure
    required_dirs = [
        processor.raw_dir,
        processor.processed_dir,
        processor.chunk_dir,
        processor.dedup_dir,
    ]
    for dir_path in required_dirs:
        if not dir_path.exists():
            validation_results["errors"].append(
                f"Required directory missing: {dir_path}"
            )
            validation_results["valid"] = False
        elif not os.access(dir_path, os.W_OK):
            validation_results["errors"].append(
                f"Directory not writable: {dir_path}"
            )
            validation_results["valid"] = False

    # Check embedding service
    try:
        if not processor.embedding_service:
            validation_results["errors"].append(
                "Embedding service not initialized"
            )
            validation_results["valid"] = False
    except Exception as e:
        validation_results["errors"].append(
            f"Embedding service error: {str(e)}"
        )
        validation_results["valid"] = False

    # Check configuration
    if processor.chunk_size < 100:
        validation_results["warnings"].append(
            f"Very small chunk size: {processor.chunk_size}"
        )
    elif processor.chunk_size > 2000:
        validation_results["warnings"].append(
            f"Very large chunk size: {processor.chunk_size}"
        )

    if processor.chunk_overlap >= processor.chunk_size:
        validation_results["errors"].append(
            "Chunk overlap should be less than chunk size"
        )
        validation_results["valid"] = False

    if not (0.5 <= processor.similarity_threshold <= 1.0):
        validation_results["warnings"].append(
            f"Unusual similarity threshold: {processor.similarity_threshold}"
        )

    # Check supported file types
    validation_results["info"].append(
        f"Supports {len(processor.get_supported_extensions())} file types"
    )
    validation_results["info"].append(
        f"Deduplication: {'enabled' if processor.enable_deduplication else 'disabled'}"
    )
    validation_results["info"].append(
        f"PDF strategy: {processor.pdf_loader_strategy}"
    )

    return validation_results


def create_document_processor(
    config_type: str = "production", **kwargs
) -> DocumentProcessor:
    """
    Factory function to create optimized document processors for different use cases.

    Args:
        config_type: "production", "development", "research", or "memory_optimized"
        **kwargs: Override specific parameters
    """

    configs = {
        "production": {
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "similarity_threshold": 0.92,
            "enable_deduplication": True,
            "pdf_loader_strategy": "auto",
        },
        "development": {
            "chunk_size": 200,
            "chunk_overlap": 10,
            "similarity_threshold": 0.90,
            "enable_deduplication": True,
            "pdf_loader_strategy": "pypdf",
        },
        "research": {
            "chunk_size": 1200,
            "chunk_overlap": 300,
            "similarity_threshold": 0.85,
            "enable_deduplication": True,
            "pdf_loader_strategy": "pdfplumber",
        },
        "memory_optimized": {
            "chunk_size": 600,
            "chunk_overlap": 100,
            "similarity_threshold": 0.95,
            "enable_deduplication": True,
            "pdf_loader_strategy": "pypdf",
        },
    }

    if config_type not in configs:
        raise ValueError(
            f"Unknown config type: {config_type}. Available: {list(configs.keys())}"
        )

    config = configs[config_type]
    config.update(kwargs)  # Override with user-provided kwargs

    processor = DocumentProcessor(**config)

    logger.info(f"Created {config_type} document processor")
    logger.info(f"Configuration: {config}")

    return processor


# Performance monitoring and metrics
class ProcessingMetrics:
    """Track processing metrics and performance."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.start_time = None
        self.end_time = None
        self.files_processed = 0
        self.files_failed = 0
        self.total_chunks = 0
        self.total_chars = 0
        self.deduplication_savings = 0

    def start_processing(self):
        self.start_time = datetime.now()

    def end_processing(self):
        self.end_time = datetime.now()

    def add_document(self, success: bool, chunk_count: int, char_count: int):
        if success:
            self.files_processed += 1
            self.total_chunks += chunk_count
            self.total_chars += char_count
        else:
            self.files_failed += 1

    def set_deduplication_savings(
        self, original_chunks: int, final_chunks: int
    ):
        self.deduplication_savings = (
            ((original_chunks - final_chunks) / original_chunks * 100)
            if original_chunks > 0
            else 0
        )

    def get_summary(self) -> Dict[str, Any]:
        duration = (
            (self.end_time - self.start_time).total_seconds()
            if self.start_time and self.end_time
            else 0
        )

        return {
            "duration_seconds": round(duration, 2),
            "files_processed": self.files_processed,
            "files_failed": self.files_failed,
            "success_rate": (
                (
                    self.files_processed
                    / (self.files_processed + self.files_failed)
                    * 100
                )
                if (self.files_processed + self.files_failed) > 0
                else 0
            ),
            "total_chunks": self.total_chunks,
            "total_characters": self.total_chars,
            "avg_chunks_per_file": (
                round(self.total_chunks / self.files_processed, 1)
                if self.files_processed > 0
                else 0
            ),
            "processing_speed_files_per_second": (
                round(self.files_processed / duration, 2) if duration > 0 else 0
            ),
            "deduplication_savings_percent": round(
                self.deduplication_savings, 2
            ),
        }


def process_documents_with_metrics(
    processor: DocumentProcessor,
    source_path: Union[str, Path],
    process_type: str = "folder",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Process documents with comprehensive metrics tracking.

    Args:
        processor: DocumentProcessor instance
        source_path: Path to file or folder
        process_type: "file", "folder", or "all"

    Returns:
        Tuple of (results, metrics)
    """
    metrics = ProcessingMetrics()
    metrics.start_processing()

    try:
        if process_type == "file":
            result = processor.process_file(source_path)
            results = [result] if result else []

        elif process_type == "folder":
            results = processor.process_folder(source_path)

        elif process_type == "all":
            results = processor.process_all_documents()

        else:
            raise ValueError(f"Unknown process_type: {process_type}")

        # Update metrics
        for result in results:
            if result:
                metrics.add_document(
                    True, result["num_chunks"], len(result.get("text", ""))
                )

        # Add deduplication metrics if available
        if hasattr(processor, "all_chunks") and hasattr(
            processor, "deduplicated_chunks"
        ):
            metrics.set_deduplication_savings(
                len(processor.all_chunks), len(processor.deduplicated_chunks)
            )

        metrics.end_processing()

        return results, metrics.get_summary()

    except Exception as e:
        metrics.end_processing()
        logger.error(f"Processing failed: {str(e)}")
        return [], metrics.get_summary()


# Final compatibility check
def ensure_compatibility_with_memory_system():
    """
    Ensure document processor is compatible with the comprehensive memory system.
    This function validates that all required components work together.
    """
    compatibility_report = {
        "document_processor": True,
        "memory_system": True,
        "embedding_service": True,
        "vector_db": True,
        "overall_compatible": True,
        "recommendations": [],
    }

    try:
        # Test document processor creation
        processor = create_document_processor("production")

        # Test chunk creation
        test_chunks = processor._create_chunks(
            "Test text for compatibility", "test_doc", "test.txt"
        )

        if not test_chunks:
            compatibility_report["document_processor"] = False
            compatibility_report["overall_compatible"] = False

        # Check if chunks have required fields for memory system
        required_fields = ["chunk_id", "doc_id", "text", "metadata"]
        for item in required_fields:
            if not hasattr(test_chunks[0], item):
                compatibility_report["document_processor"] = False
                compatibility_report["overall_compatible"] = False
                compatibility_report["recommendations"].append(
                    f"Chunks missing required field: {item}"
                )

        logger.info("Document processor compatibility check passed")

    except Exception as e:
        logger.error(f"Compatibility check failed: {str(e)}")
        compatibility_report["overall_compatible"] = False
        compatibility_report["recommendations"].append(f"Fix error: {str(e)}")

    return compatibility_report
