# KnowStrath API Documentation

## Overview

The KnowStrath API provides programmatic access to the Strathmore University Retrieval Augmented Generation (RAG) system. This RESTful API enables applications to query the knowledge base, manage documents, and interact with the system.

## Base URL

```
http://localhost:8000
```

## Authentication

All API requests require authentication using an API key.

- Include the API key in the `Authorization` header:
  ```
  Authorization: Bearer your_api_key_here
  ```

## API Endpoints

### Health Check

#### `GET /health`

Check if the API is running.

**Response:**
```json
{
  "status": "healthy"
}
```

### Querying

#### `POST /query`

Query the RAG system with a natural language question.

**Request Body:**
```json
{
  "query": "What are the tuition payment deadlines?",
  "top_k": 5,
  "doc_id": "optional_document_id"
}
```

**Parameters:**
- `query` (required): The question to ask
- `top_k` (optional): Number of chunks to retrieve (default: 5)
- `doc_id` (optional): Limit search to a specific document

**Response:**
```json
{
  "response": "Tuition payment deadlines are as follows...",
  "intent_type": "factual_query",
  "topic": "fees",
  "confidence": 0.85,
  "token_usage": {
    "prompt_tokens": 350,
    "completion_tokens": 120,
    "total_tokens": 470
  }
}
```

#### `POST /multi-query-search`

Perform a multi-query search for improved retrieval.

**Request Body:**
```json
{
  "query": "What is the attendance policy?",
  "top_k": 5,
  "doc_id": "optional_document_id"
}
```

**Response:**
```json
{
  "query": "What is the attendance policy?",
  "chunks": [
    {
      "text": "The attendance policy requires students to...",
      "score": 0.92,
      "doc_id": "12ba56666fd8dee15d7608e4566597d0",
      "chunk_id": "12ba56666fd8dee15d7608e4566597d0_0004"
    },
    // Additional chunks
  ],
  "count": 5
}
```

### Document Management

#### `POST /documents/upload`

Upload and process a new document.

**Request:**
- Content-Type: `multipart/form-data`
- Form Field: `file` (The document file to upload)

**Response:**
```json
{
  "doc_id": "12ba56666fd8dee15d7608e4566597d0",
  "file_name": "student_handbook.pdf",
  "doc_type": "pdf",
  "num_chunks": 156,
  "success": true,
  "message": "Document successfully processed and indexed"
}
```

#### `DELETE /documents/{doc_id}`

Delete a document and its associated chunks.

**Path Parameters:**
- `doc_id` (required): The ID of the document to delete

**Response:**
```json
{
  "doc_id": "12ba56666fd8dee15d7608e4566597d0",
  "file_name": "student_handbook.pdf",
  "doc_type": "pdf",
  "num_chunks": 156,
  "success": true,
  "message": "Document successfully deleted"
}
```

#### `GET /documents`

List all processed documents.

**Response:**
```json
{
  "documents": [
    {
      "doc_id": "12ba56666fd8dee15d7608e4566597d0",
      "file_name": "student_handbook.pdf",
      "doc_type": "pdf",
      "num_chunks": 156,
      "file_path": "/path/to/original/file.pdf",
      "processed_path": "/path/to/processed/file.txt",
      "chunks_path": "/path/to/chunks/file_chunks.jsonl"
    },
    // Additional documents
  ],
  "count": 3
}
```

#### `GET /documents/{doc_id}`

Get detailed information about a specific document.

**Path Parameters:**
- `doc_id` (required): The ID of the document to retrieve

**Response:**
```json
{
  "doc_id": "12ba56666fd8dee15d7608e4566597d0",
  "file_name": "student_handbook.pdf",
  "doc_type": "pdf",
  "num_chunks": 156,
  "file_path": "/path/to/original/file.pdf",
  "processed_path": "/path/to/processed/file.txt",
  "chunks_path": "/path/to/chunks/file_chunks.jsonl",
  "success": true
}
```

#### `POST /documents/update`

Update an existing document with a new version.

**Request:**
- Content-Type: `multipart/form-data`
- Form Field: `file` (The updated document file)

**Response:**
```json
{
  "doc_id": "12ba56666fd8dee15d7608e4566597d0",
  "file_name": "student_handbook.pdf",
  "doc_type": "pdf",
  "num_chunks": 162,
  "success": true,
  "message": "Document successfully updated and indexed"
}
```

#### `POST /rebuild-index`

Rebuild the vector database index with all current documents.

**Response:**
```json
{
  "success": true,
  "message": "Vector database index rebuilt successfully with 3 documents",
  "indexed_documents": 3,
  "total_documents": 3
}
```

## Error Handling

The API returns appropriate HTTP status codes:

- `200 OK`: Request successful
- `400 Bad Request`: Invalid request parameters
- `401 Unauthorized`: Missing or invalid API key
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server-side error

Error responses include a detailed message:

```json
{
  "detail": "Error message explaining what went wrong"
}
```

## Examples

### Python Example: Querying the System

```python
import requests

API_URL = "http://localhost:8000"
API_KEY = "your_api_key_here"
headers = {"Authorization": f"Bearer {API_KEY}"}

def ask_question(query):
    response = requests.post(
        f"{API_URL}/query",
        headers=headers,
        json={"query": query, "top_k": 5}
    )
    return response.json()

answer = ask_question("What are the requirements for graduation?")
print(answer["response"])
```

### Python Example: Document Management

```python
import requests

API_URL = "http://localhost:8000"
API_KEY = "your_api_key_here"
headers = {"Authorization": f"Bearer {API_KEY}"}

# Upload a document
def upload_document(file_path):
    with open(file_path, "rb") as file:
        files = {"file": file}
        response = requests.post(
            f"{API_URL}/documents/upload",
            headers=headers,
            files=files
        )
    return response.json()

# List all documents
def list_documents():
    response = requests.get(
        f"{API_URL}/documents",
        headers=headers
    )
    return response.json()

# Delete a document
def delete_document(doc_id):
    response = requests.delete(
        f"{API_URL}/documents/{doc_id}",
        headers=headers
    )
    return response.json()

# Upload example
result = upload_document("path/to/document.pdf")
print(f"Document uploaded with ID: {result['doc_id']}")
```

## Supported Document Formats

The API supports the following document formats:
- PDF (.pdf)
- Plain Text (.txt)
- Word Documents (.docx, .doc)
- HTML (.html)
- Markdown (.md)

## Rate Limiting

Please be mindful of system resources. The API may implement rate limiting for high-volume requests.
