import re
import nltk
import numpy as np
from typing import List, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity


class SemanticChunker:
    def __init__(
        self,
        target_chunk_size: int = 400,
        overlap_ratio: float = 0.1,
        embedding_service=None,
    ):
        self.target_chunk_size = target_chunk_size
        self.overlap_ratio = overlap_ratio
        self.embedding_service = embedding_service

    def chunk_text(self, text: str, doc_id: str) -> List["Chunk"]:
        sentences = nltk.sent_tokenize(text)
        if len(sentences) < 2:
            return self._create_single_chunk(text, doc_id, 0)

        sentence_embeddings = self._get_sentence_embeddings(sentences)
        boundaries = self._find_semantic_boundaries(
            sentence_embeddings, sentences
        )
        chunks = self._create_chunks_from_boundaries(
            sentences, boundaries, doc_id
        )

        return chunks

    def _get_sentence_embeddings(self, sentences: List[str]) -> np.ndarray:
        if not self.embedding_service:
            return np.random.rand(len(sentences), 1536)

        try:
            embeddings = self.embedding_service._embed_texts(sentences)
            return embeddings
        except Exception:
            return np.random.rand(len(sentences), 1536)

    def _find_semantic_boundaries(
        self, embeddings: np.ndarray, sentences: List[str]
    ) -> List[int]:
        boundaries = [0]

        for i in range(1, len(embeddings) - 1):
            sim_prev = cosine_similarity([embeddings[i]], [embeddings[i - 1]])[
                0
            ][0]
            sim_next = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[
                0
            ][0]

            if sim_prev < 0.7 and sim_next < 0.7:
                boundaries.append(i)

            current_length = sum(
                len(sentences[j]) for j in range(boundaries[-1], i + 1)
            )
            if current_length > self.target_chunk_size * 1.5:
                boundaries.append(i)

        boundaries.append(len(sentences))
        return boundaries

    def _create_chunks_from_boundaries(
        self, sentences: List[str], boundaries: List[int], doc_id: str
    ) -> List["Chunk"]:
        chunks = []
        overlap_sentences = (
            int(len(sentences) * self.overlap_ratio)
            if len(sentences) > 5
            else 1
        )

        for i in range(len(boundaries) - 1):
            start_idx = max(
                0, boundaries[i] - (overlap_sentences if i > 0 else 0)
            )
            end_idx = min(
                len(sentences),
                boundaries[i + 1]
                + (overlap_sentences if i < len(boundaries) - 2 else 0),
            )

            chunk_sentences = sentences[start_idx:end_idx]
            chunk_text = " ".join(chunk_sentences)

            if chunk_text.strip():
                chunk = self._create_chunk(
                    chunk_text, doc_id, i, boundaries, start_idx, end_idx
                )
                chunks.append(chunk)

        return chunks

    def _create_chunk(
        self,
        text: str,
        doc_id: str,
        chunk_index: int,
        boundaries: List[int],
        start_idx: int,
        end_idx: int,
    ) -> "Chunk":
        chunk_id = f"{doc_id}_{chunk_index:04d}"
        boundary_score = (
            1.0
            if chunk_index == 0 or chunk_index == len(boundaries) - 2
            else 0.8
        )
        info_score = self._calculate_information_score(text)

        return Chunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            chunk_index=chunk_index,
            text=text,
            metadata={
                "doc_id": doc_id,
                "chunk_index": chunk_index,
                "char_count": len(text),
                "word_count": len(text.split()),
                "sentence_range": f"{start_idx}-{end_idx}",
                "semantic_chunking": True,
            },
            information_score=info_score,
            semantic_boundary_score=boundary_score,
        )

    def _create_single_chunk(
        self, text: str, doc_id: str, chunk_index: int
    ) -> List["Chunk"]:
        chunk_id = f"{doc_id}_{chunk_index:04d}"
        chunk = Chunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            chunk_index=chunk_index,
            text=text,
            metadata={
                "doc_id": doc_id,
                "chunk_index": chunk_index,
                "char_count": len(text),
                "word_count": len(text.split()),
                "semantic_chunking": True,
                "single_chunk": True,
            },
            information_score=self._calculate_information_score(text),
            semantic_boundary_score=1.0,
        )
        return [chunk]

    def _calculate_information_score(self, text: str) -> float:
        if not text.strip():
            return 0.0

        length_score = min(1.0, len(text) / 500)

        entity_pattern = r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b"
        entities = len(re.findall(entity_pattern, text))
        entity_score = min(1.0, entities / 10)

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
            "schedule",
        ]
        keyword_matches = sum(
            1 for keyword in edu_keywords if keyword in text.lower()
        )
        keyword_score = min(1.0, keyword_matches / 5)

        structure_score = (
            0.2
            if any(
                indicator in text for indicator in ["-", "•", "1.", "2.", "\n"]
            )
            else 0.0
        )

        return (
            0.3 * length_score
            + 0.3 * entity_score
            + 0.25 * keyword_score
            + 0.15 * structure_score
        )


class Chunk:
    def __init__(
        self,
        chunk_id: str,
        doc_id: str,
        chunk_index: int,
        text: str,
        metadata: Dict[str, Any] = None,
        embedding: np.ndarray = None,
        information_score: float = 0.0,
        semantic_boundary_score: float = 0.0,
        is_primary: bool = False,
        source_file: str = "",
    ):
        self.chunk_id = chunk_id
        self.doc_id = doc_id
        self.chunk_index = chunk_index
        self.text = text
        self.metadata = metadata or {}
        self.embedding = embedding
        self.information_score = information_score
        self.semantic_boundary_score = semantic_boundary_score
        self.is_primary = is_primary
        self.source_file = source_file

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "metadata": self.metadata,
            "information_score": self.information_score,
            "semantic_boundary_score": self.semantic_boundary_score,
            "source_file": self.source_file,
        }
