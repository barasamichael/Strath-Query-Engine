# Embeddings-Based RAG Deduplication Tool

A professional-grade deduplication tool for Retrieval-Augmented Generation (RAG) systems that uses OpenAI embeddings to detect both exact and semantic duplicates in your knowledge base.

## Overview

Duplicate content in RAG systems can cause significant problems:
- Biases retrieval results toward repeated information
- Wastes API tokens and increases costs
- Reduces the diversity and quality of responses
- Slows down indexing and retrieval operations

This tool addresses these issues with a sophisticated, embeddings-based approach that leverages the same embedding models as production RAG systems.

## Features

- **Multi-level duplication detection**:
  - Exact hash-based detection for bit-perfect duplicates
  - Normalized hash detection for formatting differences
  - Embeddings-based semantic similarity for conceptual duplicates

- **High-quality similarity detection**:
  - Uses OpenAI's text-embedding-ada-002 model
  - Same embedding model as production RAG systems
  - Configurable similarity thresholds
  - Optional chunking for partial document duplicates

- **Performance optimization**:
  - Efficient API usage with batch processing
  - Two-phase approach (hash first, then embeddings)
  - Only processes non-duplicate files with the API
  - Parallel processing where appropriate

- **Comprehensive output**:
  - Detailed JSON reports with similarity metrics
  - Human-readable summaries
  - Optional diff reports to see exactly what differs
  - Deduplicated dataset generation

## Installation

### Prerequisites

- Python 3.7 or higher
- OpenAI API key with embedding access

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/rag-embeddings-deduplicator.git
cd rag-embeddings-deduplicator

# Install dependencies
pip install openai numpy tqdm

# Set API key (recommended to use environment variables)
export OPENAI_API_KEY=your_api_key_here
```

## Usage

### Basic Usage

```bash
python embeddings_deduplicator.py --input_dir /path/to/data --output_dir /path/to/output
```

### All Options

```bash
python embeddings_deduplicator.py \
  --input_dir /path/to/data \
  --output_dir /path/to/output \
  --api_key your_api_key_here \
  --model text-embedding-ada-002 \
  --similarity_threshold 0.92 \
  --chunk_size 1000 \
  --chunk_overlap 200 \
  --batch_size 20 \
  --use_chunking \
  --action all
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--input_dir` | Directory containing files to deduplicate | (Required) |
| `--output_dir` | Directory for output files and reports | (Required) |
| `--api_key` | OpenAI API key (overrides environment variable) | OPENAI_API_KEY env var |
| `--model` | OpenAI embedding model to use | text-embedding-ada-002 |
| `--similarity_threshold` | Threshold for similarity detection (0.0-1.0) | 0.92 |
| `--chunk_size` | Size of chunks for similarity detection | 1000 |
| `--chunk_overlap` | Overlap between chunks | 200 |
| `--batch_size` | Batch size for API calls | 20 |
| `--use_chunking` | Enable chunking for similarity detection | False |
| `--action` | Action: 'report', 'deduplicate', 'diffs', or 'all' | report |

## Output

The tool produces several outputs based on the chosen action:

### For all modes ('report'):
- `deduplication_report.json`: Detailed JSON report with all duplicates found
- `deduplication_summary.txt`: Human-readable summary of duplication analysis

### When using 'deduplicate' action:
- `/deduplicated`: Directory containing deduplicated files
- `recommended_removals.txt`: List of files recommended for removal

### When using 'diffs' action:
- `/diffs`: Directory containing diff files between similar document pairs

## Implementation Details

### Embeddings-Based Similarity Detection

The core of this tool uses embedding vectors to detect semantic similarity between documents:

1. **Document processing**:
   - Clean and normalize text
   - Split into chunks (optional)
   - Generate embeddings using OpenAI API

2. **Similarity calculation**:
   - Cosine similarity between document embeddings
   - Document-level or chunk-level comparison
   - Threshold-based duplicate identification

3. **Optimization techniques**:
   - Two-phase approach: hash-based first, then embeddings
   - Only process unique documents with expensive API calls
   - Batch processing to minimize API requests

## Examples

### Generate a Basic Report

```bash
python embeddings_deduplicator.py --input_dir ./data/raw --output_dir ./reports
```

### Create a Deduplicated Dataset

```bash
python embeddings_deduplicator.py --input_dir ./data/raw --output_dir ./processed --action deduplicate
```

### Detect Partial Document Duplicates

```bash
python embeddings_deduplicator.py --input_dir ./data/raw --output_dir ./reports --use_chunking
```

### Generate Full Analysis with Diffs

```bash
python embeddings_deduplicator.py --input_dir ./data/raw --output_dir ./analysis --action all
```

## Integration with Existing Systems

While designed as a standalone tool, this utility can be easily integrated into your RAG pipeline:

```python
# Example integration
import subprocess
import os

# Step 1: Run deduplication before indexing
os.environ["OPENAI_API_KEY"] = "your_api_key_here"
subprocess.run([
    "python", "embeddings_deduplicator.py",
    "--input_dir", "./data/raw",
    "--output_dir", "./data/processed",
    "--action", "deduplicate",
    "--use_chunking"
])

# Step 2: Process deduplicated data with your RAG system
# ... your existing RAG code here ...
```

## Tips for Best Results

- **Adjust similarity threshold** based on your needs:
  - Higher values (0.95+) for strict similarity detection
  - Lower values (0.85-0.90) to catch more potential duplicates

- **Enable chunking** for long documents or when looking for partial duplicates

- **Use the 'diffs' action** to inspect differences before removing files

- **Run periodically** as your knowledge base grows

## Costs and Performance

- **API Usage**: Each document requires one embedding API call
- **Cost Estimate**: Approximately $0.0001 per 1K tokens (~750 words)
- **Optimization**: Hash-based first pass reduces API calls significantly
- **Large Datasets**: For datasets >1000 documents, consider running in batches

## License

MIT License
