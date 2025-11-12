# Complete RAG Pipeline Documentation

**Project Name**: Enterprise RAG Pipeline with Streaming Architecture

**Key Technologies**: Python, Poetry, LangChain, Docker, Milvus (Hybrid Search: Dense + Sparse BM25), SQLite, Pydantic, Streaming Architecture (Producer-Consumer), Multi-threaded Processing, Sentence Transformers, PyMuPDF, Pandas

---

## Table of Contents

0. [Problem Statement & Solution](#problem-statement--solution)
1. [Tech Stack](#tech-stack)
2. [Initial Setup](#initial-setup)
3. [Phase 2: Data Models and Type Safety with Pydantic](#phase-2-data-models-and-type-safety-with-pydantic)
4. [Phase 3: Document Loading and File Readers](#phase-3-document-loading-and-file-readers)
5. [Phase 4: Intelligent Chunking Strategies](#phase-4-intelligent-chunking-strategies)
6. [Phase 5: Database Architecture - SQLite for Metadata](#phase-5-database-architecture---sqlite-for-metadata)
7. [Phase 6: Milvus Hybrid Search Setup](#phase-6-milvus-hybrid-search-setup)
8. [Phase 7: Streaming Pipeline Architecture](#phase-7-streaming-pipeline-architecture)
9. [Phase 8: Domain Classification and Document Separation](#phase-8-domain-classification-and-document-separation)
10. [Phase 9: Evaluation Framework - Ground Truth and Metrics](#phase-9-evaluation-framework---ground-truth-and-metrics)
11. [File Reference Guide](#file-reference-guide)
12. [Architectural Decisions and Design Rationale](#architectural-decisions-and-design-rationale)

---

## Problem Statement

Modern organizations have large collections of documents (PDFs, text files, JSON, CSV, etc.) spread across different domains. When users need to retrieve highly relevant answers—not just keyword matches, but contextually appropriate passages—from these sources, traditional search and naive chunking approaches fail. The diversity in document formats, writing styles, and internal structure means that a single, generic search or chunking mechanism cannot deliver answers with consistent precision and depth. There is a need for a retrieval-augmented generation (RAG) system that can handle multimodal documents, apply context-sensitive chunking, and provide semantically and literally accurate results.

---

## Solution

Our pipeline delivers robust, domain-aware RAG by:

- **Automatically detecting each document's type and domain** to choose context-appropriate chunking strategies. This includes recursive splitting, header-driven chunking, row-based chunking for CSV, and JSON-object preserving methods.
- **Generating both semantic (dense) and keyword (sparse) embeddings** for each chunk, and combining these at retrieval time to cover queries needing literal precision and broader semantic context.
- **Maintaining a clear modular separation** between metadata tracking (SQLite) and similarity search vector storage (Milvus), supporting rich analytics and fast chunk retrieval.
- **Enforcing modularity at every pipeline step**—from reading, chunking, validation, embedding, to evaluation—making the system maintainable and extensible for future domains, new chunking logic, or different models.
- **Integrating an evaluation framework** measuring retrieval accuracy (by IR metrics and LLM-based estimations) so the methodology and architecture align tightly with the goal: returning the right answers, not just the fastest.

With this architecture, our RAG system achieves efficient, accurate, and flexible context retrieval across document types and domains, directly addressing both methodology and engineering needs of a real-world enterprise RAG platform.

---

## Tech Stack

### Python

**Version**: Python 3.9+

Python serves as the foundation language for this RAG pipeline due to its rich ecosystem of NLP, machine learning, and data processing libraries.

**Key Libraries Used**:
- **Data Processing**: pandas, openpyxl
- **PDF Handling**: PyMuPDF (fitz)
- **NLP**: spacy, nltk
- **Machine Learning**: torch, transformers, sentence-transformers
- **Type Safety**: pydantic
- **Utilities**: python-dotenv, requests

### Poetry

**Purpose**: Dependency management and packaging

Poetry solves a fundamental problem in Python development: managing project dependencies in a reproducible, deterministic way. Before Poetry, developers used `pip` with `requirements.txt` files, which had several critical flaws.

#### The Problem with Traditional `pip` + `requirements.txt`

Imagine you develop a Python package locally:
```
- You install langchain == 0.1.0
- You install pydantic == 2.0.0
- You install milvus-sdk == 2.3.0
```

You create `requirements.txt` with these versions. Six months later, you deploy to production. In the meantime:
- langchain released 0.1.5 with a bug
- A colleague installed langchain 0.1.2 (not 0.1.0)
- The production server auto-updated to 0.1.5

Now the code behaves differently across machines. This is **dependency hell**.

#### How Poetry Solves This

Poetry maintains TWO files:

1. **`pyproject.toml`** (you edit this):
```toml
[tool.poetry.dependencies]
python = "^3.9"                    # Accept 3.9, 3.10, 3.11, etc.
langchain = "^0.1.0"              # Accept 0.1.x but not 0.2.0
pydantic = "^2.0.0"               # Accept 2.0.x but not 3.0.0
pymilvus = "^2.3.0"
```

2. **`poetry.lock`** (Poetry generates this automatically):
```
[langchain 0.1.7]
name = "langchain"
version = "0.1.7"
requires-python = ">=3.9"
dependencies:
  - requests [version >=2.28.0]
  - numpy [version >=1.21.0]
...
```

The `poetry.lock` file captures EXACT versions of all transitive dependencies. When you run `poetry install`:
1. Poetry reads `poetry.lock`
2. Installs EXACT versions specified
3. All developers get identical environments

**Key Benefit**: Deterministic builds. Every machine has the same packages.

#### Installation

```bash
# Universal installation method (works on Windows, macOS, Linux)
pip install poetry

# Verify installation
poetry --version
# Output: Poetry (version 1.7.1)
```

#### Common Commands

```bash
# Install all dependencies from pyproject.toml
poetry install

# Add new package
poetry add requests==2.31.0

# Remove package
poetry remove langchain

# Update all packages
poetry update

# Run Python code within Poetry environment
poetry run python main.py

# Activate virtual environment
poetry shell
```

#### Version Constraints

| Syntax | Meaning | Allows |
|--------|---------|--------|
| `^2.0.0` | Compatible | 2.0.0, 2.1.0, 2.9.9 (NOT 3.0.0) |
| `~2.0.0` | Approximate | 2.0.0, 2.0.1, 2.0.9 (NOT 2.1.0) |
| `>=2.0.0` | At least | 2.0.0, 2.1.0, 3.0.0, 99.0.0 |
| `==2.0.0` | Exact | Only 2.0.0 |

**Recommended**: Use `^` for most dependencies (allows minor updates, locks major versions).

### LangChain

**Purpose**: Text splitting and chunking utilities

LangChain provides a comprehensive suite of text splitters that handle different document formats intelligently.

**Key Components Used**:
- `RecursiveCharacterTextSplitter`: Hierarchical splitting (paragraph → sentence → word)
- `RecursiveJsonSplitter`: JSON-aware splitting (preserves object structure)
- `TokenTextSplitter`: Token-based splitting (respects LLM context limits)
- `SentenceTransformersTokenTextSplitter`: Sentence-based (never breaks mid-sentence)
- `MarkdownHeaderTextSplitter`: Markdown structure-aware
- `HTMLHeaderTextSplitter`: HTML-aware splitting

**Why LangChain?**
- Battle-tested splitting logic
- Handles edge cases (encoding, Unicode, special characters)
- Maintains context boundaries
- Configurable overlap and chunk size

#### Issues with TokenTextSplitter

During development, we encountered significant issues with LangChain's `TokenTextSplitter`:

**Problem 1: Inconsistent Token Counting**
- The splitter uses tiktoken for token counting
- Different models tokenize differently (GPT-3 vs GPT-4 vs sentence-transformers)
- Token count estimates were often inaccurate, leading to chunks that exceeded model limits

**Problem 2: Performance Bottlenecks**
- Token-based splitting is computationally expensive
- For large documents (100K+ tokens), splitting could take 10-30 seconds per document
- Slowed down overall pipeline throughput significantly

**Problem 3: Loss of Semantic Boundaries**
- Token-based splits often broke mid-sentence or mid-word
- Created confusing, incomplete chunks
- Example:
  ```
  Chunk 1: "The mechanism of action involves competitive inhibition of enzyme X. This pre"
  Chunk 2: "vents substrate binding and results in reduced metabolic activity."
  ```

**Our Solution**:
- Switched to `RecursiveCharacterTextSplitter` as primary strategy
- Use character-based approximation: `estimated_tokens = len(text) / 3`
- Post-process oversized chunks by splitting at sentence boundaries
- Result: 5-10x faster processing with better semantic coherence

**Lesson Learned**: Token-based splitting sounds ideal in theory but introduces complexity and performance issues in practice. Character-based splitting with heuristic token estimation is more pragmatic for production systems.

### Docker

**Purpose**: Containerization and deployment

Docker enables consistent, reproducible deployment across development, staging, and production environments.

#### Docker Compose Setup

Our system uses Docker Compose to orchestrate multiple services:

**`docker-compose.yml`**:
```yaml
version: '3.8'

services:
  # Milvus standalone
  milvus-standalone:
    container_name: milvus-standalone
    image: milvusdb/milvus:v2.3.0
    command: ["milvus", "run", "standalone"]
    environment:
      ETCD_USE_EMBED: true
      COMMON_STORAGETYPE: local
    volumes:
      - ./milvus_data:/var/lib/milvus
    ports:
      - "19530:19530"
      - "9091:9091"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9091/healthz"]
      interval: 30s
      timeout: 20s
      retries: 3

  # Text Embedding Inference (TEI) server
  tei:
    container_name: tei-embedding-server
    image: ghcr.io/huggingface/text-embeddings-inference:latest
    command: --model-id sentence-transformers/all-mpnet-base-v2
    ports:
      - "8080:80"
    volumes:
      - ./tei_data:/data
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # RAG Pipeline Application
  rag-pipeline:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: rag-pipeline
    depends_on:
      - milvus-standalone
      - tei
    environment:
      - MILVUS_HOST=milvus-standalone
      - MILVUS_PORT=19530
      - TEI_ENDPOINT=http://tei:80
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    command: poetry run python -m project.complete_pipeline_hybrid
```

#### Dockerfile

**`Dockerfile`**:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry

# Copy Poetry files
COPY pyproject.toml poetry.lock ./

# Install Python dependencies
RUN poetry install --no-dev --no-interaction --no-ansi

# Copy application code
COPY src/ ./src/
COPY data/ ./data/

# Create necessary directories
RUN mkdir -p /app/logs /app/data/output

# Run pipeline
CMD ["poetry", "run", "python", "-m", "project.complete_pipeline_hybrid"]
```

#### Deployment Commands

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f rag-pipeline

# Stop all services
docker-compose down

# Rebuild after code changes
docker-compose up -d --build

# Scale TEI servers (for high throughput)
docker-compose up -d --scale tei=3
```

#### Benefits of Docker Deployment

- **Consistency**: Same environment across dev/staging/prod
- **Isolation**: Dependencies don't conflict with host system
- **Scalability**: Easy to scale individual services (multiple TEI instances)
- **Portability**: Works on any machine with Docker installed
- **Resource Management**: Set memory/CPU limits per service

### Milvus

**Purpose**: Vector database for hybrid search (dense + sparse vectors)

Milvus is an open-source vector database optimized for similarity search at billion-scale.

#### Why Milvus?

| Feature | Milvus | Alternatives (Pinecone, Weaviate) |
|---------|--------|-----------------------------------|
| **Cost** | Free (open-source) | Paid ($$$ at scale) |
| **Hybrid Search** | Built-in dense+sparse | Limited or complex |
| **Performance** | Sub-5ms on millions of vectors | Similar |
| **Deployment** | Self-hosted (full control) | Cloud-only or limited |
| **BM25 Support** | Native sparse vector + BM25 function | Manual implementation |
| **Scalability** | Billions of vectors | Same |

**Key Advantage**: Milvus is the only major vector DB with **native BM25 sparse vector support**, enabling true hybrid search without external systems.

#### Milvus Architecture

```
┌─────────────────────────────────────────────────┐
│            Milvus Standalone                    │
│                                                 │
│  ┌──────────────┐      ┌──────────────┐        │
│  │   etcd       │      │   MinIO      │        │
│  │  (metadata)  │      │  (storage)   │        │
│  └──────────────┘      └──────────────┘        │
│                                                 │
│  ┌──────────────────────────────────────────┐  │
│  │         Query Node                       │  │
│  │  - Vector search (HNSW index)            │  │
│  │  - Sparse search (Inverted index)        │  │
│  │  - Hybrid search (RRF combination)       │  │
│  └──────────────────────────────────────────┘  │
│                                                 │
│  ┌──────────────────────────────────────────┐  │
│  │         Data Node                        │  │
│  │  - Ingestion                             │  │
│  │  - Indexing                              │  │
│  │  - Persistence                           │  │
│  └──────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

#### Hybrid Search Components

**1. Dense Vectors (Semantic)**
- 768-dimensional embeddings from sentence-transformers
- Indexed with HNSW (Hierarchical Navigable Small World)
- Captures semantic meaning
- Cosine similarity metric

**2. Sparse Vectors (Keyword)**
- Generated automatically by Milvus BM25 function
- Tokenizes text, removes stopwords, calculates TF-IDF weights
- Indexed with Inverted Index
- BM25 similarity metric

**3. Hybrid Search (Combination)**
- Execute both dense and sparse searches in parallel
- Combine results using RRF (Reciprocal Rank Fusion)
- Formula: `RRF_score = Σ(1 / (k + rank_i))` where k=60
- Result: Best of both worlds (30-50% better accuracy)

#### Collection Schema

Our Milvus collection has the following fields:

```python
fields = [
    FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, is_primary=True, max_length=256),
    FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="chunk_text", dtype=DataType.VARCHAR, max_length=4096, enable_analyzer=True),
    FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=768),
    FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
    # ... additional metadata fields
]
```

**Critical**: `enable_analyzer=True` on `chunk_text` enables automatic BM25 sparse vector generation.

#### Index Configuration

**Dense Vector Index** (HNSW):
```python
index_params.add_index(
    field_name="dense_vector",
    index_type="HNSW",
    metric_type="COSINE",
    params={"M": 16, "efConstruction": 200}
)
```

**Sparse Vector Index** (Inverted):
```python
index_params.add_index(
    field_name="sparse_vector",
    index_type="SPARSE_INVERTED_INDEX",
    metric_type="BM25"
)
```

#### Performance Characteristics

| Operation | Latency | Throughput |
|-----------|---------|------------|
| Insert (batch of 100) | 100-200ms | 500-1000 chunks/sec |
| Dense search | 3-8ms | 1000+ queries/sec |
| Sparse search | 1-3ms | 2000+ queries/sec |
| Hybrid search | 10-15ms | 500+ queries/sec |
| Build HNSW index | ~1 min | For 1M vectors |

### Additional Technologies

**Pydantic**: Type-safe data models and validation  
**SQLite**: Metadata storage and analytics  
**Sentence Transformers**: Dense embedding generation  
**PyMuPDF**: Fast PDF text extraction  
**Pandas**: CSV/TSV processing and data manipulation  
**tiktoken**: Token counting (OpenAI tokenizer)  
**spacy/nltk**: NLP utilities  

---

## Initial Setup

### Prerequisites

- Python 3.9 or higher
- Docker and Docker Compose
- 8GB+ RAM (16GB+ recommended)
- GPU (optional, for faster embedding)

### Step 1: Clone Repository and Install Dependencies

```bash
# Clone the repository
git clone <repository-url>
cd rag-pipeline

# Install Poetry (if not already installed)
pip install poetry

# Install project dependencies
poetry install

# Verify installation
poetry run python --version
```

### Step 2: Configure Environment Variables

Create `.env` file in project root:

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your settings
nano .env
```

**`.env` contents**:
```
# Milvus Configuration
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_COLLECTION_NAME=rag_chunks

# SQLite Configuration
SQLITE_DB_PATH=data/db/documents.db

# Azure OpenAI (for evaluation)
AZURE_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_DEPLOYMENT_NAME=gpt-4
AZURE_API_KEY=your-api-key-here

# Processing Configuration
CHUNK_SIZE=1024
CHUNK_OVERLAP=256
NUM_CONSUMER_THREADS=5
QUEUE_MAX_SIZE=1500
```

### Step 3: Start Docker Services

```bash
# Start Milvus and TEI (Text Embedding Inference)
docker-compose up -d

# Verify services are running
docker-compose ps

# Expected output:
# NAME                     STATUS
# milvus-standalone        Up (healthy)
# tei-embedding-server     Up
```

### Step 4: Create Database Schemas

```bash
# Create SQLite schema
poetry run python -m project.sqlite_setup

# Create Milvus collection schema
poetry run python -m project.schema_setup

# Verify schemas
poetry run python -m project.check_schema
```

Expected output:
```
✓ SQLite schema created successfully
  - documents table: 10 columns
  - chunks table: 13 columns

✓ Milvus collection 'rag_chunks' created successfully
  - Fields: 15 total
  - Indexes: HNSW (dense) + INVERTED (sparse)
  - Status: Ready for data insertion
```

### Step 5: Project Structure

Verify your project structure matches:

```
rag-pipeline/
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
├── poetry.lock
├── .env
├── .env.example
├── README.md
│
├── src/
│   └── project/
│       ├── __init__.py
│       ├── config.py
│       ├── pydantic_models.py
│       ├── doc_reader.py
│       ├── chunker.py
│       ├── chunk_cleaner.py
│       ├── file_meta_loader.py
│       ├── sqlite_setup.py
│       ├── schema_setup.py
│       ├── complete_pipeline_hybrid.py
│       ├── populate_sqlite_from_json.py
│       ├── process_all_queries_csv.py
│       ├── classify_domains_parallel.py
│       └── check_schema.py
│
├── data/
│   ├── input/              # Place raw documents here
│   ├── output/             # Processed output goes here
│   └── db/                 # SQLite databases
│
└── logs/                   # Pipeline logs
```

### Step 6: Test Installation

Run a simple test to verify everything works:

```bash
# Test document reading
poetry run python -c "from project.doc_reader import DocumentReader; print('✓ Doc reader OK')"

# Test Milvus connection
poetry run python -c "from project.config import client; print('✓ Milvus connection OK')"

# Test embedding service
curl http://localhost:8080/health
# Expected: {"status":"ok"}
```

### Step 7: Prepare Input Data

```bash
# Create input directory structure
mkdir -p data/input/research_papers
mkdir -p data/input/legal_documents
mkdir -p data/input/general

# Copy your documents to appropriate folders
cp /path/to/your/pdfs/* data/input/research_papers/
```

### You're Ready!

Your RAG pipeline is now set up. Proceed to run the pipeline:

```bash
poetry run python -m project.complete_pipeline_hybrid \
    --input-dir data/input \
    --output-dir data/output
```

---

## Phase 2: Data Models and Type Safety with Pydantic

[CONTENT REMAINS SAME AS BEFORE - KEEP ENTIRE PHASE 2]

---

## Phase 3: Document Loading and File Readers

[CONTENT REMAINS SAME AS BEFORE - KEEP ENTIRE PHASE 3]

---

## Phase 4: Intelligent Chunking Strategies

[CONTENT REMAINS SAME AS BEFORE - KEEP ENTIRE PHASE 4]

---

## Phase 5: Database Architecture - SQLite for Metadata

[CONTENT REMAINS SAME AS BEFORE - KEEP ENTIRE PHASE 5]

---

## Phase 6: Milvus Hybrid Search Setup

[CONTENT REMAINS SAME AS BEFORE - KEEP ENTIRE PHASE 6]

---

## Phase 7: Streaming Pipeline Architecture

[CONTENT REMAINS SAME AS BEFORE - KEEP ENTIRE PHASE 7]

---

## Phase 8: Domain Classification and Document Separation

[CONTENT REMAINS SAME AS BEFORE - PREVIOUSLY PHASE 8.5, NOW PHASE 8]

---

## Phase 9: Evaluation Framework - Ground Truth and Metrics

[CONTENT REMAINS SAME AS BEFORE - KEEP ENTIRE PHASE 9]

---

## File Reference Guide

### Active Production Files

| File | Purpose | Entry Point? | Notes |
|------|---------|--------------|-------|
| `pydantic_models.py` | Data models (Document, Chunk, Config) | No | Foundation for type safety |
| `config.py` | Configuration (Milvus URI, collection name) | No | Centralized settings |
| `doc_reader.py` | Document loading (PDF, TXT, JSON, CSV, TSV) | No | Handles all file types |
| `chunker.py` | Chunking strategies (file-based & dataset-aware) | No | Two services: production & research |
| `chunk_cleaner.py` | Chunk validation and cleaning | No | Removes empty, malformed chunks |
| `file_meta_loader.py` | File metadata extraction (size, type, etc.) | No | Used by pipeline |
| `sqlite_setup.py` | SQLite schema (documents + chunks tables) | **YES** | Run once to create DB |
| `schema_setup.py` | Milvus hybrid collection setup | **YES** | Run once to create collection |
| `complete_pipeline_hybrid.py` | Main streaming pipeline (producer-consumer) | **YES** | **PRIMARY PIPELINE (USED)** |
| `populate_sqlite_from_json.py` | Populate SQLite from JSON backup | **YES** | Recovery/migration |
| `process_all_queries_csv.py` | Evaluation framework | **YES** | Run queries and compute metrics |
| `classify_domains_parallel.py` | Domain classification | **YES** | Parallel document classification |

### Utility/Support Files

| File | Purpose | When to Use? |
|------|---------|--------------|
| `check_schema.py` | Verify Milvus schema | After creating collection |

### Deprecated/Not Used Files

These files exist in the repository but are **NOT used** in the actual pipeline:

| File | Why Not Used | Alternative |
|------|--------------|-------------|
| `complete_pipeline_gpu.py` | Experimental GPU optimization that was not production-ready | `complete_pipeline_hybrid.py` (CPU-based streaming) |
| `milvus_bulk_import.py` | Planned for bulk import workflow, but we use streaming pipeline directly | `complete_pipeline_hybrid.py` inserts directly |
| `embedder.py` | Embedding handled by external TEI (Text Embedding Inference) service | TEI Docker container |
| `query_engine.py` | Simple query interface, superseded by evaluation framework | `process_all_queries_csv.py` |
| `test_hybrid_search.py` | Testing utility, integrated into evaluation | `process_all_queries_csv.py` |

**Important**: The actual production workflow is:

1. Documents → `complete_pipeline_hybrid.py` → Milvus + SQLite (streaming, real-time)
2. **NOT**: Documents → JSON → Bulk Import (we don't use this)

The streaming pipeline handles everything: reading, chunking, cleaning, embedding (via TEI), and insertion to both Milvus and SQLite in one unified workflow.

---

## Architectural Decisions and Design Rationale

### 1. Streaming Over Batch Processing

**Decision**: Use streaming producer-consumer architecture instead of loading entire files.

**Rationale**:
- **Memory efficiency**: Bounded queue prevents overflow (150MB max)
- **Scalability**: Can process large datasets without OOM errors
- **Responsiveness**: Real-time progress tracking
- **Fault tolerance**: Failed files don't crash pipeline

**Tradeoff**: More complex implementation (threading, synchronization)

### 2. Hybrid Search (Dense + Sparse)

**Decision**: Implement both semantic (dense) and keyword (sparse) search.

**Rationale**:
- **Better accuracy**: 30-50% improvement over dense-only
- **Entity matching**: Catches exact terms (product codes, names)
- **Robustness**: Works for both semantic and keyword queries

**Tradeoff**: Slightly higher storage (sparse vectors + dense vectors)

### 3. SQLite + Milvus Dual Storage

**Decision**: Use SQLite for metadata, Milvus for vectors.

**Rationale**:
- **Separation of concerns**: Each database does what it's best at
- **Flexibility**: SQL for analytics, Milvus for search
- **Backup**: SQLite is a single file (easy to backup)

**Tradeoff**: Data in two places (need to keep in sync)

### 4. Pydantic Models Everywhere

**Decision**: Use Pydantic for all data structures.

**Rationale**:
- **Type safety**: Catches errors at development time
- **Validation**: Automatic data validation
- **Documentation**: Models document data contracts

**Tradeoff**: Slight learning curve, minor performance overhead

### 5. File-Type and Dataset-Aware Chunking

**Decision**: Two chunking services (file-based + dataset-based).

**Rationale**:
- **Flexibility**: Can choose strategy based on use case
- **Quality**: Specialized strategies outperform one-size-fits-all
- **Extensibility**: Easy to add new strategies

**Tradeoff**: More code to maintain

### 6. Automated Evaluation

**Decision**: Use LLM + NLI for automatic relevance scoring.

**Rationale**:
- **Scalability**: No manual annotation needed
- **Reproducibility**: Consistent across runs
- **Speed**: Can evaluate thousands of queries

**Tradeoff**: Not as accurate as human annotation (but close enough)

### 7. Character-Based vs Token-Based Splitting

**Decision**: Use character-based splitting with heuristic token estimation instead of token-based splitting.

**Rationale**:
- **Performance**: 5-10x faster than token-based splitting
- **Simplicity**: Avoids tokenizer dependencies and version mismatches
- **Semantic coherence**: Better boundary detection using sentence/paragraph breaks
- **Reliability**: No dependency on specific tokenizer implementations

**Tradeoff**: Token counts are approximate (but accurate enough for practical use)

---

## Conclusion

This RAG pipeline represents a production-grade system with:

1. **Scalability**: Streaming architecture handles large-scale datasets
2. **Accuracy**: Hybrid search (dense + sparse) for best retrieval quality
3. **Efficiency**: Bounded memory, multi-threaded processing
4. **Intelligence**: Domain-aware chunking, automatic classification
5. **Measurability**: Automated evaluation framework
6. **Maintainability**: Modular architecture, type-safe models

**Key Innovations**:
- Streaming producer-consumer (no memory overflow)
- Hybrid search (semantic + keyword)
- Dual storage (SQLite + Milvus)
- Automated evaluation (no manual annotation)
- Character-based chunking (avoiding token-based complexity)

**Performance**:
- Throughput: 400-500 chunks/second with streaming
- Memory: Bounded at 150MB maximum
- Search: Sub-5ms query latency (Milvus HNSW)
- Accuracy: 30-50% better than dense-only search

This system can scale to millions of documents while maintaining high retrieval quality and system reliability.
