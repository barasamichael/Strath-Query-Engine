# Production Deployment Guide - KnowStrath

## Pre-Deployment Checklist

### 1. Install ChromaDB and Dependencies

```bash
# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install production requirements
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### 2. Set Environment Variables

Create a `.env` file in the project root:

```bash
# OpenAI API Key (REQUIRED)
OPENAI_API_KEY=sk-your-actual-api-key-here

# API Configuration
HOST=0.0.0.0
PORT=8000
DEDUPLICATION_ENABLED=true

# Generate a secure API key for production
# Use: python -c "import secrets; print(secrets.token_urlsafe(32))"
API_KEY=your-secure-random-key-here
```

### 3. Update config.yaml for Production

```yaml
environment: production

llm:
  provider: openai
  model: gpt-3.5-turbo
  temperature: 0.1
  max_tokens: 1000

embedding:
  model: text-embedding-ada-002
  dimension: 1536

vector_db:
  type: chromadb
  location: local
  collection_name: strathmore_handbook

chunking:
  chunk_size: 500
  chunk_overlap: 50

deduplication:
  enabled: true
  similarity_threshold: 0.92
  information_weight: 0.1

api:
  host: 0.0.0.0
  port: 8000
  debug: false  # MUST be false in production
  api_key: "${API_KEY}"  # Will be loaded from environment
```

## Deployment Steps

### Step 1: Process Documents

```bash
# Process all documents with deduplication
python cli.py process-all-documents

# Verify deduplication status
python cli.py deduplication-status

# Generate embeddings for deduplicated chunks (if not already done)
python cli.py embed-deduplicated
```

### Step 2: Initialize ChromaDB

```bash
# Initialize collection (will use existing if available)
python cli.py initialize-collection

# Index deduplicated chunks
python cli.py index-deduplicated

# Verify indexing
python cli.py query "test query" --top-k 3
```

### Step 3: Test the System

```bash
# Test with CLI
python cli.py interactive

# Example queries to test:
# - "What are the admission requirements?"
# - "Tell me about student housing"
# - "How do I pay my fees?"
```

### Step 4: Start Production Server

```bash
# Start with uvicorn (production ASGI server)
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4

# Or with gunicorn (more robust)
gunicorn api.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## Cost Optimization Features

### 1. Embedding Cache

The system now caches embeddings and only regenerates when:
- Embeddings file doesn't exist
- Source file has changed (detected by hash)
- Embeddings file is corrupted

**Benefits:**
- Prevents redundant API calls
- Saves significant costs on re-processing
- Faster reindexing

**Cache Location:** `data/embeddings/embeddings_metadata.json`

### 2. Deduplication

Reduces embedding costs by:
- Merging similar chunks (default threshold: 0.92)
- Reducing total chunks by ~20-40%
- Preserving all important information

**Cost Savings Example:**
- Original: 1000 chunks × $0.0001 = $0.10
- Deduplicated: 700 chunks × $0.0001 = $0.07
- **Savings: 30%**

### 3. Batch Processing

- Processes embeddings in batches of 100
- Implements retry logic (3 attempts)
- Continues on failure instead of stopping

## Error Handling Improvements

### 1. Service-Level Error Handling

All services now raise custom exceptions:
- `VectorDBError` - Database operations
- `EmbeddingError` - Embedding generation
- `DocumentProcessingError` - Document processing

### 2. Graceful Degradation

- Failed documents don't stop entire batch
- Failed embeddings use fallback
- Search errors return empty results instead of crashing

### 3. Atomic File Operations

All file writes use temporary files then atomic rename:
```python
temp_file.write(data)
temp_file.replace(original_file)
```

## ChromaDB Advantages

### Why ChromaDB?

1. **Persistent Storage** - Data survives restarts
2. **Better Performance** - Optimized for similarity search
3. **Scalability** - Handles millions of vectors
4. **Production Ready** - Used by many companies
5. **Easy Backup** - Single directory to backup

### Storage Location

```
database/
  └── chroma_db/
      ├── chroma.sqlite3  # Metadata
      └── [uuid]/         # Vector data
```

### Backup Strategy

```bash
# Backup ChromaDB
tar -czf chroma_backup_$(date +%Y%m%d).tar.gz database/chroma_db/

# Backup embeddings cache
tar -czf embeddings_backup_$(date +%Y%m%d).tar.gz data/embeddings/

# Restore
tar -xzf chroma_backup_YYYYMMDD.tar.gz
```

## Monitoring and Maintenance

### Health Check

```bash
# Via CLI
curl http://localhost:8000/health

# Expected response:
{
  "status": "healthy",
  "services": {
    "vector_db": "healthy",
    "embedding_service": "healthy",
    "intent_recognizer": "healthy"
  },
  "collection_stats": {
    "name": "strathmore_handbook",
    "count": 700,
    "dimension": 1536
  }
}
```

### Statistics

```bash
# Get system stats
curl -H "Authorization: Bearer your-api-key" \
  http://localhost:8000/stats

# Response includes:
# - Collection size
# - Total embeddings
# - Cached files
```

### Log Monitoring

```bash
# Monitor logs in production
tail -f /var/log/knowstrath/app.log

# Key things to watch:
# - Embedding failures
# - Search errors
# - API errors
# - Token usage
```

## Performance Tuning

### Memory Management

ChromaDB uses memory-mapped files. For large datasets:

```python
# config.yaml
vector_db:
  batch_size: 100  # Adjust based on RAM
```

### Query Performance

```yaml
# For faster queries
chunking:
  chunk_size: 400  # Smaller chunks = faster search
  
# For better accuracy
chunking:
  chunk_size: 600  # Larger chunks = more context
```

### Embedding Batch Size

```python
# In embeddings.py, adjust batch_size
embed_batch(texts, batch_size=50)  # Lower for stability
embed_batch(texts, batch_size=200)  # Higher for speed
```

## Security Considerations

### 1. API Key Management

```bash
# Generate strong API key
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Store in .env, never commit
echo ".env" >> .gitignore
```

### 2. Rate Limiting

Add to `api/main.py`:

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/query")
@limiter.limit("10/minute")
async def query(...):
    ...
```

### 3. Input Validation

Already implemented:
- Query length limits
- top_k bounds
- Empty query rejection

## Troubleshooting

### Issue: "Embeddings not found"

```bash
# Clear cache and regenerate
rm data/embeddings/embeddings_metadata.json
python cli.py embed-deduplicated
```

### Issue: "ChromaDB collection empty"

```bash
# Rebuild index
python cli.py initialize-collection --recreate
python cli.py index-deduplicated
```

### Issue: "High API costs"

```bash
# Check cache status
ls -lh data/embeddings/*.npz

# Verify deduplication
python cli.py deduplication-status

# Clear cache only if necessary
```

### Issue: "Slow queries"

```bash
# Check collection size
python cli.py query "test" --top-k 1

# Reduce top_k in queries
# Consider reindexing with smaller chunks
```

## Migration from Old System

```bash
# 1. Backup old database
cp database/vector_store/*.pkl backup/

# 2. Install ChromaDB
pip install chromadb==0.4.18

# 3. Recreate collection
python cli.py initialize-collection --recreate

# 4. Reindex (will use cached embeddings)
python cli.py index-deduplicated

# 5. Test
python cli.py interactive
```

## Production Deployment Options

### Option 1: Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm

COPY . .

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Option 2: Systemd Service

```ini
# /etc/systemd/system/knowstrath.service
[Unit]
Description=KnowStrath RAG API
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/knowstrath
Environment="PATH=/opt/knowstrath/venv/bin"
ExecStart=/opt/knowstrath/venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

### Option 3: Cloud Platforms

**AWS EC2:**
- Instance: t3.medium (2vCPU, 4GB RAM)
- Storage: 20GB SSD
- Use Application Load Balancer

**Google Cloud Run:**
```bash
gcloud run deploy knowstrath \
  --source . \
  --memory 2Gi \
  --cpu 2
```

## Cost Analysis

### Embedding Costs (OpenAI)

| Scenario | Chunks | Cost per Index | Monthly* |
|----------|--------|----------------|----------|
| Initial | 1000 | $0.10 | $0.10 |
| With Cache | 1000 | $0.00 | $0.00 |
| With Dedup | 700 | $0.07 | $0.07 |
| Re-index | 700 | $0.00 | $0.00 |

*Assuming monthly re-indexing

### Query Costs (GPT-3.5-Turbo)

| Queries/Day | Cost/Query | Daily | Monthly |
|-------------|-----------|-------|---------|
| 100 | $0.002 | $0.20 | $6.00 |
| 500 | $0.002 | $1.00 | $30.00 |
| 1000 | $0.002 | $2.00 | $60.00 |

**Total Monthly Cost:** ~$6-60 depending on usage

## Support and Maintenance

### Weekly Tasks
- Review error logs
- Check API costs
- Monitor query performance

### Monthly Tasks
- Update documents if needed
- Review and update deduplication threshold
- Backup ChromaDB and embeddings

### Quarterly Tasks
- Update dependencies
- Performance audit
- Cost optimization review
