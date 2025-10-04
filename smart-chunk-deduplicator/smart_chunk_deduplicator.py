"""
Smart Chunk-Level Deduplication for RAG Systems

This tool intelligently deduplicates content at the chunk level while
preserving important information, similar to how a human would curate content.
"""

import os
import re
import json
import logging
import hashlib
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
import time
from collections import defaultdict
import heapq
import numpy as np
from tqdm import tqdm
import nltk
from nltk.tokenize import sent_tokenize
from openai import OpenAI
import concurrent.futures

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("smart_deduplicator")

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Constants
SUPPORTED_EXTENSIONS = {'.txt', '.md', '.html'}
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_SIMILARITY_THRESHOLD = 0.92
BATCH_SIZE = 20

# Initialize OpenAI client (can be set via environment variable OPENAI_API_KEY)
client = OpenAI()


@dataclass
class Chunk:
    """Represents a chunk of text with metadata and embedding."""
    id: str
    text: str
    source_file: str
    position: int  # Position in source file
    embedding: Optional[np.ndarray] = None
    information_score: float = 0.0  # Score representing information density
    merged_from: List[str] = field(default_factory=list)  # IDs of chunks this was merged from
    is_primary: bool = False  # Whether this is a primary (authoritative) chunk
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            # Generate ID if not provided
            self.id = hashlib.md5(f"{self.source_file}_{self.position}_{self.text[:50]}".encode()).hexdigest()


class TextProcessor:
    """Process and clean text."""
    
    @staticmethod
    def extract_text(file_path: Path) -> str:
        """Extract text from a file based on its extension."""
        extension = file_path.suffix.lower()
        
        try:
            # Simple text extraction
            if extension in ['.txt', '.md']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            
            elif extension == '.html':
                with open(file_path, 'r', encoding='utf-8') as f:
                    html = f.read()
                # Simple HTML tag removal (use BeautifulSoup for better results)
                text = re.sub(r'<[^>]+>', ' ', html)
                return re.sub(r'\s+', ' ', text).strip()
            
            else:
                logger.warning(f"Unsupported file type: {extension}")
                return ""
                
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {str(e)}")
            return ""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text."""
        # Replace multiple newlines with a single newline
        text = re.sub(r'\n+', '\n', text)
        
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove control characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text)
        
        return text.strip()
    
    @staticmethod
    def split_into_chunks(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, 
                          chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[str]:
        """Split text into chunks with overlap, trying to maintain paragraph integrity."""
        if not text:
            return []
        
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = []
        current_length = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            paragraph_length = len(paragraph)
            
            # If paragraph is too large, split it by sentences
            if paragraph_length > chunk_size:
                # Add current chunk if it exists
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    # Keep last paragraph for overlap
                    if current_chunk and len(current_chunk[-1]) < chunk_overlap:
                        current_chunk = [current_chunk[-1]]
                        current_length = len(current_chunk[-1])
                    else:
                        current_chunk = []
                        current_length = 0
                
                # Split large paragraph into sentences
                sentences = sent_tokenize(paragraph)
                current_sentence_chunk = []
                current_sentence_length = 0
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    sentence_length = len(sentence)
                    
                    if current_sentence_length + sentence_length <= chunk_size:
                        current_sentence_chunk.append(sentence)
                        current_sentence_length += sentence_length + 1  # +1 for space
                    else:
                        if current_sentence_chunk:
                            chunks.append(' '.join(current_sentence_chunk))
                        
                        # If sentence is too long, split it by words
                        if sentence_length > chunk_size:
                            words = sentence.split()
                            current_words = []
                            current_words_length = 0
                            
                            for word in words:
                                if current_words_length + len(word) <= chunk_size:
                                    current_words.append(word)
                                    current_words_length += len(word) + 1  # +1 for space
                                else:
                                    if current_words:
                                        chunks.append(' '.join(current_words))
                                    current_words = [word]
                                    current_words_length = len(word)
                            
                            if current_words:
                                current_sentence_chunk = [' '.join(current_words)]
                                current_sentence_length = current_words_length
                            else:
                                current_sentence_chunk = []
                                current_sentence_length = 0
                        else:
                            current_sentence_chunk = [sentence]
                            current_sentence_length = sentence_length
                
                if current_sentence_chunk:
                    chunks.append(' '.join(current_sentence_chunk))
            
            # Normal case: add paragraph to current chunk if it fits
            elif current_length + paragraph_length <= chunk_size:
                current_chunk.append(paragraph)
                current_length += paragraph_length + 2  # +2 for newline chars
            
            # Start a new chunk
            else:
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                current_chunk = [paragraph]
                current_length = paragraph_length
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks


class EmbeddingsService:
    """Generate and manage embeddings."""
    
    def __init__(self, model: str = "text-embedding-ada-002", api_key: Optional[str] = None):
        """Initialize the embeddings service."""
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
        
        self.model = model
        self.client = OpenAI(api_key=self.api_key)
        self.dimension = 1536  # For ada-002 model
        
        logger.info(f"Initialized embeddings service with model {self.model}")
        
    def get_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for a batch of texts."""
        if not texts:
            return []
            
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts
            )
            
            embeddings = [np.array(data.embedding) for data in response.data]
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            # Return zero embeddings as fallback
            return [np.zeros(self.dimension) for _ in range(len(texts))]
    
    def generate_embeddings(self, chunks: List[Chunk], batch_size: int = BATCH_SIZE) -> None:
        """Generate embeddings for all chunks."""
        # Skip chunks that already have embeddings
        chunks_without_embeddings = [chunk for chunk in chunks if chunk.embedding is None]
        
        if not chunks_without_embeddings:
            return
            
        logger.info(f"Generating embeddings for {len(chunks_without_embeddings)} chunks")
        
        # Process in batches
        for i in tqdm(range(0, len(chunks_without_embeddings), batch_size), desc="Generating embeddings"):
            batch = chunks_without_embeddings[i:i + batch_size]
            texts = [chunk.text for chunk in batch]
            
            embeddings = self.get_embeddings_batch(texts)
            
            # Assign embeddings back to chunks
            for chunk, embedding in zip(batch, embeddings):
                chunk.embedding = embedding


class InformationAnalyzer:
    """Analyze information content and importance in chunks."""
    
    @staticmethod
    def calculate_information_score(chunk: Chunk) -> float:
        """Calculate an information score for a chunk based on various metrics."""
        text = chunk.text
        
        # Basic metrics
        length_score = min(1.0, len(text) / 1000)  # Favor longer chunks up to a point
        
        # Count named entities (approximate using capitalized words)
        capitalized_words = len(re.findall(r'\b[A-Z][a-zA-Z]*\b', text))
        entity_score = min(1.0, capitalized_words / 20)  # Cap at 20 entities
        
        # Count numbers and dates
        numbers = len(re.findall(r'\b\d+\b', text))
        number_score = min(1.0, numbers / 10)  # Cap at 10 numbers
        
        # Check for lists and structured content
        has_lists = 1.0 if re.search(r'(\n\s*[-*•]\s+|\d+\.\s+)', text) else 0.0
        
        # Check for specific phrases indicating important information
        important_phrases = [
            'important', 'critical', 'essential', 'required', 'must', 
            'policy', 'regulation', 'rule', 'procedure', 'deadline',
            'contact', 'email', 'phone', 'address', 'website',
            'fee', 'payment', 'cost', 'price', 'discount',
            'schedule', 'timetable', 'date', 'time'
        ]
        
        phrase_score = sum(1 for phrase in important_phrases 
                           if re.search(fr'\b{phrase}\b', text.lower())) / len(important_phrases)
        
        # Calculate final score (weights can be adjusted)
        final_score = (
            0.2 * length_score + 
            0.25 * entity_score + 
            0.2 * number_score + 
            0.15 * has_lists + 
            0.2 * phrase_score
        )
        
        return final_score
    
    @staticmethod
    def analyze_chunks(chunks: List[Chunk]) -> None:
        """Analyze all chunks and assign information scores."""
        logger.info(f"Analyzing information content in {len(chunks)} chunks")
        
        for chunk in tqdm(chunks, desc="Analyzing information"):
            chunk.information_score = InformationAnalyzer.calculate_information_score(chunk)


class SimilarityCalculator:
    """Calculate similarity between chunks."""
    
    @staticmethod
    def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
            
        return dot_product / (norm_v1 * norm_v2)
    
    @staticmethod
    def find_similar_chunks(chunks: List[Chunk], similarity_threshold: float) -> List[List[Chunk]]:
        """
        Find clusters of similar chunks above the similarity threshold.
        Returns a list of chunk clusters (each cluster is a list of similar chunks).
        """
        logger.info(f"Finding similar chunks with threshold {similarity_threshold}")
        
        # Build similarity graph
        similarity_graph = defaultdict(list)
        
        # Compare all chunks (can be optimized with approximate nearest neighbors)
        n = len(chunks)
        for i in tqdm(range(n), desc="Building similarity graph"):
            for j in range(i + 1, n):
                similarity = SimilarityCalculator.cosine_similarity(
                    chunks[i].embedding, chunks[j].embedding
                )
                
                if similarity >= similarity_threshold:
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
                cluster.append(chunks[node])
                
                for neighbor, _ in similarity_graph[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            
            if len(cluster) > 1:  # Only add clusters with more than one chunk
                clusters.append(cluster)
        
        logger.info(f"Found {len(clusters)} clusters of similar chunks")
        return clusters


class ChunkMerger:
    """Merge similar chunks preserving all important information."""
    
    @staticmethod
    def get_unique_sentences(primary_text: str, secondary_text: str) -> List[str]:
        """
        Extract sentences from secondary_text that are not semantically present in primary_text.
        Uses a simple approach based on sentence similarity.
        """
        # Tokenize into sentences
        primary_sentences = sent_tokenize(primary_text)
        secondary_sentences = sent_tokenize(secondary_text)
        
        # Convert to lowercase for comparison
        primary_set = {sentence.lower().strip() for sentence in primary_sentences}
        
        unique_sentences = []
        
        for sentence in secondary_sentences:
            sentence_lower = sentence.lower().strip()
            
            # Check if sentence or very similar is already in primary
            if sentence_lower not in primary_set:
                # Check for near-duplicates (could be enhanced with embeddings)
                is_unique = True
                for primary_sentence in primary_set:
                    # Simple word overlap check (could be improved)
                    primary_words = set(primary_sentence.split())
                    secondary_words = set(sentence_lower.split())
                    
                    if len(primary_words) > 0 and len(secondary_words) > 0:
                        overlap = len(primary_words.intersection(secondary_words)) / max(
                            len(primary_words), len(secondary_words)
                        )
                        
                        if overlap > 0.8:  # High word overlap
                            is_unique = False
                            break
                
                if is_unique:
                    unique_sentences.append(sentence)
        
        return unique_sentences
    
    @staticmethod
    def merge_chunks(chunks: List[Chunk]) -> Chunk:
        """
        Merge a list of similar chunks, preserving all important information.
        Returns a new merged chunk.
        """
        if not chunks:
            return None
            
        # Find primary chunk (highest information score)
        primary_chunk = max(chunks, key=lambda c: c.information_score)
        primary_chunk.is_primary = True
        
        # Start with primary chunk text
        merged_text = primary_chunk.text
        merged_from = [chunk.id for chunk in chunks]
        source_files = {chunk.source_file for chunk in chunks}
        
        # Extract and add unique information from other chunks
        for chunk in chunks:
            if chunk.id == primary_chunk.id:
                continue
                
            unique_sentences = ChunkMerger.get_unique_sentences(merged_text, chunk.text)
            
            if unique_sentences:
                # Add unique sentences at the end
                merged_text += "\n\n--- Additional Information ---\n"
                merged_text += " ".join(unique_sentences)
        
        # Create merged chunk
        merged_chunk = Chunk(
            id=f"merged_{hashlib.md5(merged_text[:100].encode()).hexdigest()}",
            text=merged_text,
            source_file=f"MERGED({','.join(sorted(source_files))})",
            position=primary_chunk.position,
            information_score=primary_chunk.information_score,
            merged_from=merged_from,
            is_primary=True,
            metadata={
                "merged_from_chunks": [chunk.id for chunk in chunks],
                "merged_from_files": list(source_files),
                "primary_chunk_id": primary_chunk.id
            }
        )
        
        return merged_chunk
    
    @staticmethod
    def merge_similar_clusters(clusters: List[List[Chunk]]) -> List[Chunk]:
        """
        Process all clusters and merge similar chunks.
        Returns a list of merged chunks.
        """
        logger.info(f"Merging {len(clusters)} clusters of similar chunks")
        merged_chunks = []
        
        for i, cluster in enumerate(tqdm(clusters, desc="Merging clusters")):
            merged_chunk = ChunkMerger.merge_chunks(cluster)
            if merged_chunk:
                merged_chunks.append(merged_chunk)
        
        logger.info(f"Created {len(merged_chunks)} merged chunks")
        return merged_chunks


class SmartDeduplicator:
    """
    Main class for smart chunk-level deduplication.
    Preserves important information while removing redundancy.
    """
    
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        api_key: Optional[str] = None,
    ):
        """Initialize the deduplicator with settings."""
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_threshold = similarity_threshold
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "chunks").mkdir(exist_ok=True)
        (self.output_dir / "deduplicated").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)
        
        # Initialize services
        self.embeddings_service = EmbeddingsService(api_key=api_key)
        
        # Data
        self.all_files = []
        self.all_chunks = []
        self.original_chunks_by_file = {}
        self.deduplicated_chunks = []
        
        logger.info(
            f"Initialized deduplicator with: "
            f"chunk_size={chunk_size}, "
            f"chunk_overlap={chunk_overlap}, "
            f"similarity_threshold={similarity_threshold}"
        )
    
    def process_files(self) -> None:
        """Process all files in the input directory."""
        logger.info(f"Processing files from {self.input_dir}")
        
        # Find all files
        for file_path in self.input_dir.glob("**/*"):
            if not file_path.is_file():
                continue
                
            if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
                
            self.all_files.append(file_path)
        
        logger.info(f"Found {len(self.all_files)} files to process")
        
        # Process each file
        for file_path in tqdm(self.all_files, desc="Processing files"):
            try:
                self._process_file(file_path)
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
        
        logger.info(f"Created {len(self.all_chunks)} chunks from {len(self.all_files)} files")
    
    def _process_file(self, file_path: Path) -> None:
        """Process a single file and create chunks."""
        rel_path = file_path.relative_to(self.input_dir)
        
        # Extract and clean text
        text = TextProcessor.extract_text(file_path)
        if not text:
            logger.warning(f"No text extracted from {file_path}")
            return
            
        text = TextProcessor.clean_text(text)
        
        # Split into chunks
        chunk_texts = TextProcessor.split_into_chunks(
            text, self.chunk_size, self.chunk_overlap
        )
        
        # Create chunk objects
        file_chunks = []
        for i, chunk_text in enumerate(chunk_texts):
            chunk = Chunk(
                id=f"{str(rel_path)}_{i}",
                text=chunk_text,
                source_file=str(rel_path),
                position=i,
            )
            file_chunks.append(chunk)
            self.all_chunks.append(chunk)
        
        self.original_chunks_by_file[str(rel_path)] = file_chunks
    
    def deduplicate(self) -> None:
        """
        Perform intelligent deduplication of chunks.
        Preserves important information while removing redundancy.
        """
        if not self.all_chunks:
            logger.warning("No chunks to deduplicate")
            return
        
        # Step 1: Generate embeddings for all chunks
        self.embeddings_service.generate_embeddings(self.all_chunks)
        
        # Step 2: Analyze information content in chunks
        InformationAnalyzer.analyze_chunks(self.all_chunks)
        
        # Step 3: Find similar chunks
        similar_clusters = SimilarityCalculator.find_similar_chunks(
            self.all_chunks, self.similarity_threshold
        )
        
        # Step 4: Merge similar chunks preserving information
        merged_chunks = ChunkMerger.merge_similar_clusters(similar_clusters)
        
        # Step 5: Combine merged chunks with non-duplicated chunks
        # First, identify all chunks that were merged
        merged_chunk_ids = set()
        for cluster in similar_clusters:
            for chunk in cluster:
                merged_chunk_ids.add(chunk.id)
        
        # Combine merged chunks with unique chunks
        self.deduplicated_chunks = []
        self.deduplicated_chunks.extend(merged_chunks)
        
        for chunk in self.all_chunks:
            if chunk.id not in merged_chunk_ids:
                self.deduplicated_chunks.append(chunk)
        
        logger.info(
            f"Deduplication complete: {len(self.all_chunks)} original chunks → "
            f"{len(self.deduplicated_chunks)} deduplicated chunks "
            f"({len(merged_chunks)} merged, {len(self.deduplicated_chunks) - len(merged_chunks)} unchanged)"
        )
    
    def save_results(self) -> None:
        """Save deduplicated chunks and reports."""
        # Save original chunks
        logger.info("Saving original chunks")
        chunks_dir = self.output_dir / "chunks" / "original"
        chunks_dir.mkdir(exist_ok=True, parents=True)
        
        for file_path, chunks in self.original_chunks_by_file.items():
            output_path = chunks_dir / f"{file_path.replace('/', '_')}.jsonl"
            
            with open(output_path, "w", encoding="utf-8") as f:
                for chunk in chunks:
                    # Convert embedding to list for JSON serialization
                    chunk_data = {
                        "id": chunk.id,
                        "text": chunk.text,
                        "source_file": chunk.source_file,
                        "position": chunk.position,
                        "information_score": chunk.information_score,
                    }
                    f.write(json.dumps(chunk_data) + "\n")
        
        # Save deduplicated chunks
        logger.info("Saving deduplicated chunks")
        deduplicated_dir = self.output_dir / "chunks" / "deduplicated"
        deduplicated_dir.mkdir(exist_ok=True, parents=True)
        
        deduplicated_path = deduplicated_dir / "deduplicated_chunks.jsonl"
        with open(deduplicated_path, "w", encoding="utf-8") as f:
            for chunk in self.deduplicated_chunks:
                # Convert embedding to list for JSON serialization
                chunk_data = {
                    "id": chunk.id,
                    "text": chunk.text,
                    "source_file": chunk.source_file,
                    "position": chunk.position,
                    "information_score": chunk.information_score,
                    "is_primary": chunk.is_primary,
                    "merged_from": chunk.merged_from,
                    "metadata": chunk.metadata,
                }
                f.write(json.dumps(chunk_data) + "\n")
        
        # Create deduplicated text files (for human review)
        logger.info("Creating deduplicated text files")
        text_dir = self.output_dir / "deduplicated"
        text_dir.mkdir(exist_ok=True, parents=True)
        
        # Group deduplicated chunks by source file
        deduplicated_by_file = defaultdict(list)
        for chunk in self.deduplicated_chunks:
            if "MERGED" in chunk.source_file:
                # Create a separate file for merged chunks
                merged_file = f"MERGED_{chunk.id}.txt"
                deduplicated_by_file[merged_file].append(chunk)
            else:
                deduplicated_by_file[chunk.source_file].append(chunk)
        
        # Write deduplicated files
        for file_path, chunks in deduplicated_by_file.items():
            # Sort chunks by position
            chunks.sort(key=lambda c: c.position)
            
            output_path = text_dir / file_path.replace('/', '_')
            
            with open(output_path, "w", encoding="utf-8") as f:
                for chunk in chunks:
                    if chunk.merged_from and len(chunk.merged_from) > 1:
                        f.write(f"\n\n--- MERGED CHUNK (from {len(chunk.merged_from)} sources) ---\n\n")
                    f.write(chunk.text)
                    f.write("\n\n")
        
        # Generate summary report
        logger.info("Generating summary report")
        report_path = self.output_dir / "reports" / "deduplication_report.json"
        
        # Calculate statistics
        total_original_text = sum(len(chunk.text) for chunk in self.all_chunks)
        total_deduplicated_text = sum(len(chunk.text) for chunk in self.deduplicated_chunks)
        text_reduction = total_original_text - total_deduplicated_text
        
        merged_chunks = [chunk for chunk in self.deduplicated_chunks if chunk.merged_from and len(chunk.merged_from) > 1]
        
        # Build report
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "input_directory": str(self.input_dir),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "similarity_threshold": self.similarity_threshold,
            "stats": {
                "total_files": len(self.all_files),
                "total_original_chunks": len(self.all_chunks),
                "total_deduplicated_chunks": len(self.deduplicated_chunks),
                "merged_chunks": len(merged_chunks),
                "unchanged_chunks": len(self.deduplicated_chunks) - len(merged_chunks),
                "total_original_text": total_original_text,
                "total_deduplicated_text": total_deduplicated_text,
                "text_reduction": text_reduction,
                "reduction_percentage": (text_reduction / total_original_text * 100) if total_original_text > 0 else 0
            },
            "merged_chunks": [
                {
                    "id": chunk.id,
                    "merged_from": chunk.merged_from,
                    "source_files": chunk.metadata.get("merged_from_files", []),
                    "information_score": chunk.information_score
                }
                for chunk in merged_chunks
            ]
        }
        
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        
        # Generate human-readable summary
        summary_path = self.output_dir / "reports" / "deduplication_summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"Smart Chunk Deduplication Summary: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Input Directory: {self.input_dir}\n")
            f.write(f"Chunk Size: {self.chunk_size}, Chunk Overlap: {self.chunk_overlap}\n")
            f.write(f"Similarity Threshold: {self.similarity_threshold}\n\n")
            
            f.write(f"Files Processed: {len(self.all_files)}\n")
            f.write(f"Original Chunks: {len(self.all_chunks)}\n")
            f.write(f"Deduplicated Chunks: {len(self.deduplicated_chunks)}\n")
            f.write(f"Merged Chunks: {len(merged_chunks)}\n")
            f.write(f"Unchanged Chunks: {len(self.deduplicated_chunks) - len(merged_chunks)}\n\n")
            f.write(f"Text Reduction: {text_reduction} characters ({report['stats']['reduction_percentage']:.2f}%)\n\n")

            # List top merged chunks
            if merged_chunks:
                f.write("Top Merged Chunks (by information score):\n")
                top_merged = sorted(merged_chunks, key=lambda c: c.information_score, reverse=True)[:10]
                for i, chunk in enumerate(top_merged, 1):
                    source_files = chunk.metadata.get("merged_from_files", [])
                    f.write(f"  {i}. Merged from {len(chunk.merged_from)} chunks across {len(source_files)} files\n")
                    f.write(f"     Source files: {', '.join(source_files)}\n")
                    f.write(f"     Information score: {chunk.information_score:.4f}\n")
                    f.write(f"     First 100 chars: {chunk.text[:100]}...\n\n")

        logger.info(f"Reports saved to {self.output_dir}/reports")

    def export_for_rag(self) -> None:
        """Export deduplicated chunks in a format suitable for RAG systems."""
        logger.info("Exporting deduplicated chunks for RAG system")

        # Create rag directory
        rag_dir = self.output_dir / "rag"
        rag_dir.mkdir(exist_ok=True, parents=True)

        # Export chunks in a clean format
        rag_path = rag_dir / "deduplicated_chunks.jsonl"
        with open(rag_path, "w", encoding="utf-8") as f:
            for chunk in self.deduplicated_chunks:
                # Create a clean record for RAG
                rag_record = {
                    "chunk_id": chunk.id,
                    "text": chunk.text,
                    "source": chunk.source_file,
                    "metadata": {
                        "merged": len(chunk.merged_from) > 1,
                        "sources": chunk.metadata.get("merged_from_files", [chunk.source_file]),
                        "information_score": chunk.information_score,
                    }
                }
                f.write(json.dumps(rag_record) + "\n")

        # Create a README explaining the format
        readme_path = rag_dir / "README.txt"
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write("Deduplicated Chunks for RAG System\n")
            f.write("=================================\n\n")
            f.write("This directory contains deduplicated chunks ready for use in a RAG system.\n\n")
            f.write("File Format:\n")
            f.write("  - deduplicated_chunks.jsonl: One JSON record per line with the following structure:\n")
            f.write("    - chunk_id: Unique identifier for the chunk\n")
            f.write("    - text: The deduplicated text content\n")
            f.write("    - source: Source file or indication of merged sources\n")
            f.write("    - metadata: Additional information about the chunk\n")
            f.write("      - merged: Boolean indicating if this is a merged chunk\n")
            f.write("      - sources: List of source files this chunk contains information from\n")
            f.write("      - information_score: A score indicating information density\n\n")
            f.write("Usage Recommendations:\n")
            f.write("  1. Index these chunks in your vector database\n")
            f.write("  2. Consider weighting chunks by information_score during retrieval\n")
            f.write("  3. Use the metadata.sources field to provide attribution in responses\n")

        logger.info(f"RAG-ready data exported to {rag_dir}")

    def run(self) -> None:
        """Run the complete deduplication process."""
        start_time = time.time()
        logger.info("Starting smart chunk deduplication")

        # Process files
        self.process_files()

        # Deduplicate
        self.deduplicate()

        # Save results
        self.save_results()

        # Export for RAG
        self.export_for_rag()

        elapsed_time = time.time() - start_time
        logger.info(f"Deduplication completed in {elapsed_time:.2f} seconds")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Smart chunk-level deduplication for RAG systems")

    parser.add_argument("--input_dir", required=True, help="Directory containing input files")
    parser.add_argument("--output_dir", required=True, help="Directory for output files and reports")
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE,
                       help=f"Size of chunks (default: {DEFAULT_CHUNK_SIZE})")
    parser.add_argument("--chunk_overlap", type=int, default=DEFAULT_CHUNK_OVERLAP,
                       help=f"Overlap between chunks (default: {DEFAULT_CHUNK_OVERLAP})")
    parser.add_argument("--similarity_threshold", type=float, default=DEFAULT_SIMILARITY_THRESHOLD,
                       help=f"Threshold for similarity detection (default: {DEFAULT_SIMILARITY_THRESHOLD})")
    parser.add_argument("--api_key", help="OpenAI API key (optional, can use OPENAI_API_KEY env var)")

    args = parser.parse_args()

    try:
        # Initialize deduplicator
        deduplicator = SmartDeduplicator(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            similarity_threshold=args.similarity_threshold,
            api_key=args.api_key,
        )

        # Run deduplication
        deduplicator.run()

    except Exception as e:
        logger.error(f"Error during deduplication: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
