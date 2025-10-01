import os
import re
import logging
import hashlib

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

from config.settings import settings
from config.settings import ROOT_DIR

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


class DocumentProcessor:
    def __init__(
        self,
        raw_dir: Path = None,
        processed_dir: Path = None,
        chunk_dir: Path = None,
    ):
        self.raw_dir = raw_dir or ROOT_DIR / "data" / "raw"
        self.processed_dir = processed_dir or ROOT_DIR / "data" / "processed"
        self.chunk_dir = chunk_dir or ROOT_DIR / "data" / "chunks"

        # Ensure directories exist
        for dir_path in [self.raw_dir, self.processed_dir, self.chunk_dir]:
            if not dir_path.exists():
                dir_path.mkdir(parents=True)

    def process_all_documents(self) -> List[Dict[str, Any]]:
        """Process all documents in the raw directory and return their metadata."""
        logger.info(f"Processing all documents in {self.raw_dir}")

        documents_metadata = []

        for file_path in tqdm(list(self.raw_dir.glob("**/*"))):
            if not file_path.is_file():
                continue

            try:
                metadata = self.process_document(file_path)
                if metadata:
                    documents_metadata.append(metadata)
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")

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
        chunks = self._create_chunks(cleaned_text, doc_id)

        # Save chunks
        chunk_path = self.chunk_dir / f"{doc_id}_chunks.jsonl"
        self._save_chunks(chunks, chunk_path)

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

    def _create_chunks(self, text: str, doc_id: str) -> List[Dict[str, Any]]:
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
            chunk = {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "chunk_index": i,
                "text": chunk_text,
                "metadata": {"doc_id": doc_id, "chunk_index": i},
            }
            chunks.append(chunk)

        return chunks

    def _save_chunks(
        self, chunks: List[Dict[str, Any]], output_path: Path
    ) -> None:
        """Save chunks to a JSONL file."""
        import json

        with open(output_path, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk) + "\n")

    def _generate_document_id(self, file_path: Path) -> str:
        """Generate a unique document ID based on file path and modification time."""
        file_stat = file_path.stat()
        unique_string = f"{file_path}_{file_stat.st_mtime}"
        return hashlib.md5(unique_string.encode()).hexdigest()
