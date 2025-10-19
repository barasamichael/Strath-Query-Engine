# KnowStrath CLI Documentation

## Overview

The `cli.py` script provides a command-line interface for managing and interacting with the KnowStrath system, a Retrieval Augmented Generation (RAG) system for Strathmore University. This CLI enables document processing, querying, and system maintenance.

## Installation

Before using the CLI, ensure you have:
1. Set up the KnowStrath environment
2. Installed the required dependencies
3. Set the OpenAI API key:
   ```bash
   export OPENAI_API_KEY=your_key_here
   ```

## Basic Usage

Run the CLI with the following format:
```bash
python cli.py [COMMAND] [OPTIONS]
```

To see all available commands:
```bash
python cli.py --help
```

## Command Reference

### Document Management

#### `process-document`
Process a single document, creating chunks, embeddings, and indexing it in the vector database.

```bash
python cli.py process-document PATH_TO_FILE
```

#### `process-folder`
Process all documents in a specified folder.

```bash
python cli.py process-folder PATH_TO_FOLDER
```

#### `process-all-documents`
Process all documents in the configured raw directory.

```bash
python cli.py process-all-documents
```

#### `delete-document`
Remove a document and its associated chunks from the system.

```bash
python cli.py delete-document DOCUMENT_ID
```

#### `update-document`
Update an existing document with a new version.

```bash
python cli.py update-document PATH_TO_UPDATED_FILE
```

#### `list-documents`
List all processed documents in the system.

```bash
python cli.py list-documents
# For detailed information
python cli.py list-documents --details
```

#### `document-info`
Get detailed information about a specific document.

```bash
python cli.py document-info DOCUMENT_ID
```

### Vector Database Management

#### `initialize-collection`
Initialize or recreate the vector database collection.

```bash
python cli.py initialize-collection
# To recreate an existing collection
python cli.py initialize-collection --recreate
```

#### `rebuild-index`
Rebuild the vector database index with all processed documents.

```bash
python cli.py rebuild-index
```

### Deduplication Management

#### `deduplication-status`
Show the status of deduplication in the system.

```bash
python cli.py deduplication-status
```

#### `run-deduplication`
Run the deduplication process on existing chunks.

```bash
python cli.py run-deduplication
```

#### `embed-deduplicated`
Generate embeddings for deduplicated chunks.

```bash
python cli.py embed-deduplicated
```

#### `index-deduplicated`
Index deduplicated chunks in the vector database.

```bash
python cli.py index-deduplicated
```

### Query System

#### `query`
Test a query against the RAG system.

```bash
python cli.py query "Your query text here"
# With options
python cli.py query "Your query text here" --top-k 10 --multi-query
```

#### `interactive`
Start an interactive query session.

```bash
python cli.py interactive
# With options
python cli.py interactive --top-k 20 --multi-query
```

## Workflow Examples

### Adding New Documents

```bash
# Process a single document
python cli.py process-document data/raw/new/handbook.pdf

# Process all documents in a folder
python cli.py process-folder data/raw/new_batch/

# Process all documents in the raw directory
python cli.py process-all-documents
```

### Managing Documents

```bash
# List all documents
python cli.py list-documents

# Get detailed information about a document
python cli.py document-info 12ba56666fd8dee15d7608e4566597d0

# Update a document
python cli.py update-document data/raw/updated/handbook_v2.pdf

# Delete a document
python cli.py delete-document 12ba56666fd8dee15d7608e4566597d0
```

### Optimizing the System

```bash
# Run deduplication
python cli.py run-deduplication

# Generate embeddings for deduplicated chunks
python cli.py embed-deduplicated

# Index deduplicated chunks
python cli.py index-deduplicated

# Rebuild the vector database index
python cli.py rebuild-index
```

### Querying

```bash
# Run a single query
python cli.py query "What are the tuition payment deadlines?"

# Start an interactive session
python cli.py interactive
```

## Advanced Usage

### Controlling Deduplication

The system can deduplicate similar chunks across documents to improve efficiency:

```bash
# Check deduplication status
python cli.py deduplication-status

# Complete deduplication workflow
python cli.py run-deduplication
python cli.py embed-deduplicated
python cli.py index-deduplicated
```

### Multi-Query Approach

For better retrieval quality, use the multi-query approach:

```bash
python cli.py query "What is the attendance policy?" --multi-query
```

## Troubleshooting

- If you see API key errors, ensure your OpenAI API key is set correctly in your environment
- For processing errors, check file formats (PDF, TXT, DOCX, HTML, and MD are supported)
- If queries return irrelevant information, try rebuilding the vector database or adjusting the `top-k` parameter
