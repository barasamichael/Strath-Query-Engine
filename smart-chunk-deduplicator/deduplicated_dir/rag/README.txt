Deduplicated Chunks for RAG System
=================================

This directory contains deduplicated chunks ready for use in a RAG system.

File Format:
  - deduplicated_chunks.jsonl: One JSON record per line with the following structure:
    - chunk_id: Unique identifier for the chunk
    - text: The deduplicated text content
    - source: Source file or indication of merged sources
    - metadata: Additional information about the chunk
      - merged: Boolean indicating if this is a merged chunk
      - sources: List of source files this chunk contains information from
      - information_score: A score indicating information density

Usage Recommendations:
  1. Index these chunks in your vector database
  2. Consider weighting chunks by information_score during retrieval
  3. Use the metadata.sources field to provide attribution in responses
