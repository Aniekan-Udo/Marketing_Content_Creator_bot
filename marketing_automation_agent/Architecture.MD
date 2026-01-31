# Marketing Content Generation Agent - Architecture Documentation

**Project Type:** AI Agent System for Marketing Content Generation  
**Tech Stack:** Python, CrewAI, LlamaIndex, FastAPI/Flask, PostgreSQL  
**Last Updated:** January 28, 2026

---

## Executive Summary

This is a **multi-agent AI system** that generates brand-aligned marketing content (blogs, social posts, ads) using:
- **CrewAI** for agent orchestration
- **RAG (Retrieval-Augmented Generation)** for brand voice learning
- **Hybrid learning system** combining automatic metrics + human feedback
- **PostgreSQL** for persistent storage and learning

**Current Complexity Level:** HIGH
- 3 main files, ~2000+ lines of code
- 5 AI agents in sequential pipeline
- Multiple databases (vector stores + relational)
- Real-time feedback loops
- Thread-safe caching systems

---

## System Architecture Overview

### High-Level Flow

```
User Upload Brand Docs → RAG Indexing → Agent Pipeline → Content Generation → Human Feedback → Learning Storage
```

### Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      Flask Web API (app.py)                  │
│  /api/upload | /api/generate | /api/feedback | /api/status  │
└───────────────────────┬─────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
        ▼               ▼               ▼
┌──────────────┐ ┌─────────────┐ ┌──────────────┐
│  PostgreSQL  │ │   Vector    │ │   Learning   │
│   Database   │ │   Stores    │ │   Memory     │
│   (db.py)    │ │  (PGVector) │ │  (PostgreSQL)│
└──────────────┘ └─────────────┘ └──────────────┘
        │               │               │
        └───────────────┼───────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │   CrewAI Agent Pipeline       │
        │       (deployer.py)           │
        └───────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
        ▼               ▼               ▼
  ┌─────────┐    ┌──────────┐    ┌──────────┐
  │Research │    │Creative  │    │  Brand   │
  │ Agent   │───▶│Strategist│───▶│ Analyst  │
  └─────────┘    └──────────┘    └──────────┘
                                       │
                                       ▼
                               ┌──────────────┐
                               │Writer Agent  │
                               └──────┬───────┘
                                      │
                                      ▼
                              ┌────────────────┐
                              │Reviewer Agent  │
                              │(Hybrid Scoring)│
                              └────────────────┘
```

---

## Core Components

### 1. **Web API Layer** (`app.py`)

**Purpose:** HTTP interface for all interactions

**Key Endpoints:**
- `POST /api/upload` - Upload brand documents
- `POST /api/generate` - Trigger content generation
- `GET /api/status` - Poll generation progress
- `POST /api/feedback/submit` - Submit human feedback
- `GET /api/feedback/stats/<business_id>` - Get learning statistics

**Design Pattern:** RESTful API with thread-based async execution

**Key Features:**
- Thread-safe status tracking per user
- Global caching (KB instances, learning memory)
- Dynamic user creation (no pre-registration needed)
- Error handling with detailed JSON responses

**Critical Sections:**
```python
# Global caches (thread-safe with locks in deployer.py)
status_store = {}              # {user_id: status_dict}
kb_instance_cache = {}         # {business_id: kb_instance}
learning_memory_cache = {}     # {business_id: memory_instance}
generation_storage_cache = {}  # {business_id: storage_instance}
```

---

### 2. **Agent Orchestration** (`deployer.py`)

**Purpose:** Multi-agent content generation pipeline using CrewAI

**Architecture:** Sequential 5-agent pipeline

#### Agent Pipeline

**Agent 1: Research Analyst**
- **Role:** Gather factual information, trends, data
- **Tools:** `TavilySearchTool` (web search)
- **Output:** 10-15 diverse factual findings
- **LLM:** Analytical (temp=0.5)

**Agent 2: Creative Content Strategist**
- **Role:** Find unique angles that match brand voice
- **Tools:** `KBQueryTool`, `LearningMemoryTool`
- **Output:** 2-3 creative angles with recommendations
- **LLM:** Creative (temp=0.8)
- **Critical:** Reviews past learnings BEFORE proposing angles

**Agent 3: Brand Voice Pattern Analyst**
- **Role:** Extract measurable metrics from brand docs
- **Tools:** `KBQueryTool`, `BrandMetricsTool`
- **Output:** Structural metrics + qualitative patterns
- **LLM:** Analytical (temp=0.5)
- **Provides:** Exact targets (sentence length, paragraph structure, signature phrases)

**Agent 4: Brand Voice Content Writer**
- **Role:** Write content matching exact brand metrics
- **Tools:** `KBQueryTool`, `BrandMetricsTool`
- **Output:** Full content + self-check metadata
- **LLM:** Creative (temp=0.8)
- **Constraints:** Must hit structural targets from Agent 3

**Agent 5: Hybrid Quality Enforcer (Reviewer)**
- **Role:** Score content using automatic metrics + LLM judgment
- **Tools:** `ContentScoringTool`, `SaveLearningTool`
- **Output:** JSON with scores, decision, issues, learning saved
- **LLM:** Analytical (temp=0.5)
- **Scoring:** 50% automatic structural + 50% LLM qualitative

**Decision Rules:**
- ≥9.0: Approved ✅
- 8.0-8.9: Approved with suggestions ✅
- 7.0-7.9: Revision needed 🔄
- <7.0: Major revision ❌

---

### 3. **RAG System** (`Marketing_Rag_System`)

**Purpose:** Semantic search over brand documents

**Technology Stack:**
- **Embeddings:** FastEmbed (BAAI/bge-small-en-v1.5, 384 dims)
- **Vector Store:** PGVector (PostgreSQL extension)
- **Index:** HNSW for fast similarity search
- **Parser:** SemanticSplitter (blogs) or SimpleNodeParser (social/ads)

**Key Features:**
- Business-specific tables: `rag_data_{business_id}_{content_type}`
- Hybrid search (vector + BM25)
- Thread-safe initialization with locks
- Persistent storage (survives restarts)

**Data Flow:**
```
Brand Docs → SimpleDirectoryReader → Parser → Embeddings → PGVector → Query Engine
```

**Critical Configuration:**
```python
HNSW Parameters:
- m=16 (connections per layer)
- ef_construction=64 (indexing quality)
- ef_search=40 (query quality)
- distance=cosine

Chunk Sizes:
- Blog: Semantic splitting (adaptive)
- Social: 256 tokens, 50 overlap
- Ads: 256 tokens, 50 overlap
```

---

### 4. **Learning Systems**

#### A. **Brand Metrics Analyzer**

**Purpose:** Extract concrete, measurable patterns from brand docs

**What It Measures:**
- Average sentence length (words)
- Sentences per paragraph
- Bullet point usage
- Numbered list usage
- Common phrases (2-5 word sequences appearing 2+ times)

**Output Example:**
```python
{
    'target_sentence_length': 15.3,
    'target_sentences_per_para': 3.2,
    'uses_bullets': True,
    'signature_phrases': ['customer experience', 'innovative solutions', ...]
}
```

**Scoring Method:**
- Compares generated content against extracted metrics
- Returns deviation scores (0-10 scale)
- Fully objective (no LLM guessing)

#### B. **Brand Learning Memory**

**Purpose:** Store and retrieve learnings from human + auto feedback

**Storage:** PostgreSQL table `brand_learning_{business_id}`

**Schema:**
```sql
CREATE TABLE brand_learning_{business_id} (
    id SERIAL PRIMARY KEY,
    content_type VARCHAR(50),        -- blog/social/ad
    learning_type VARCHAR(50),       -- creative_variation
    pattern_data JSONB,              -- {angle, what_worked, what_failed}
    auto_score FLOAT,                -- Automatic score
    human_approved BOOLEAN,          -- Human verdict
    human_score FLOAT,               -- Human score
    human_feedback TEXT,             -- Human comments
    created_at TIMESTAMP
)
```

**Retrieval Priority:**
1. Human-approved patterns (highest priority)
2. High auto-scores (≥8.0) without human feedback
3. Sorted by score DESC, then recency

**Usage:**
- Agent 2 (Strategist) queries this BEFORE proposing angles
- Agent 5 (Reviewer) saves learnings AFTER scoring

#### C. **Reviewer Learning (Generation Storage)**

**Purpose:** Track agent predictions vs human feedback for model improvement

**Storage:** PostgreSQL table `reviewer_learning`

**Schema:**
```sql
CREATE TABLE reviewer_learning (
    id SERIAL PRIMARY KEY,
    generation_id VARCHAR(100) UNIQUE,  -- UUID
    business_id VARCHAR(100),
    topic VARCHAR(500),
    content_type VARCHAR(50),
    generated_content TEXT,
    creative_angle VARCHAR(200),
    
    -- Agent predictions
    agent_auto_score DECIMAL(3,1),
    agent_auto_approved BOOLEAN,
    
    -- Human feedback
    has_human_feedback BOOLEAN DEFAULT FALSE,
    human_approved BOOLEAN,
    human_score DECIMAL(3,1),
    human_feedback TEXT,
    
    -- Learning signal
    agent_correct BOOLEAN,  -- Did agent predict correctly?
    
    created_at TIMESTAMP
)
```

**Critical Feature:**
```python
agent_correct = (agent_auto_approved == human_approved)
```

This enables:
- Measuring agent accuracy over time
- Training reward models in future
- Identifying systematic errors

---

### 5. **Database Layer** (`db.py`)

**Purpose:** Relational data storage using SQLAlchemy ORM

**Database:** PostgreSQL

**Core Tables:**

**Users**
- Primary key for all business entities
- Tracks subscription tier, usage limits
- Created dynamically on first API call

**BrandDocuments**
- Tracks uploaded files
- Links to vector store tables
- Status tracking (uploaded/processed/error)

**Generations**
- Historical record of all generations
- Stores content, scores, timestamps
- Links to feedback

**GenerationFeedback**
- User ratings and comments
- Issue tracking (tone, structure, etc.)

**ReviewerLearning**
- Agent learning data (described above)
- Enables future model training

**Design Patterns:**
- Foreign keys with CASCADE deletes
- JSONB for flexible metadata
- Indexed by business_id + timestamps
- Server-side defaults for timestamps

---

## Data Flow Diagrams

### Content Generation Flow

```
1. User uploads brand docs
        │
        ▼
2. RAG system indexes docs into PGVector
   - Extracts embeddings
   - Stores in rag_data_{business_id}_{type}
        │
        ▼
3. User triggers /api/generate
   - Creates status tracker
   - Launches agent pipeline in thread
        │
        ▼
4. Agent Pipeline Executes
   ┌──────────────────────────────────┐
   │ Research → Creative → Analyst    │
   │     ↓          ↓         ↓        │
   │         Writer                    │
   │            ↓                      │
   │         Reviewer                  │
   │  (Auto-scores + saves learning)  │
   └──────────────────────────────────┘
        │
        ▼
5. Result stored in:
   - status_store (for /api/status polling)
   - reviewer_learning table (with generation_id)
        │
        ▼
6. User reviews content via /feedback UI
        │
        ▼
7. Human feedback saved to:
   - brand_learning_{business_id} (for future generations)
   - reviewer_learning (updates agent_correct flag)
```

### Feedback Loop Flow

```
Generation → Auto-Score (8.5/10) → Human Review
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    │                     │                     │
                Approve (9/10)      Reject (5/10)       Neutral (7/10)
                    │                     │                     │
                    ▼                     ▼                     ▼
         brand_learning table      brand_learning table    (no save)
         {                         {
           approved: true            approved: false
           human_score: 9.0          human_score: 5.0
           angle: "X"                angle: "Y"
         }                         }
                    │                     │
                    └─────────────────────┘
                              │
                              ▼
              reviewer_learning.agent_correct
              = (agent_approved == human_approved)
                              │
                              ▼
              Future generations learn from this
```

---

## Thread Safety & Concurrency

### Critical Race Conditions

**Problem:** Multiple users uploading/generating simultaneously

**Solutions Implemented:**

1. **Per-Business Locks** (`deployer.py`)
```python
_init_locks: Dict[str, Lock] = {}
_locks_lock = Lock()

def get_init_lock(business_id):
    with _locks_lock:
        if business_id not in _init_locks:
            _init_locks[business_id] = Lock()
        return _init_locks[business_id]
```

2. **RAG Index Initialization Lock**
```python
with lock:
    # Only one thread builds index per business_id
    index = rag_system.build_marketing_index()
```

3. **Thread-Safe Caching**
```python
if cache_key in self._kb_cache:
    return self._kb_cache[cache_key]

with lock:
    # Double-check pattern
    if cache_key in self._kb_cache:
        return self._kb_cache[cache_key]
    # ... build KB ...
    self._kb_cache[cache_key] = kb
```

4. **Database Session Management**
```python
with session_maker() as session:
    # Auto-commits on exit
    # Auto-rollbacks on exception
```

---

## File Structure & Naming Conventions

### Directory Layout

```
project_root/
├── deployer.py              # Agent pipeline + RAG + learning
├── app.py                   # Flask API + endpoints
├── db.py                    # SQLAlchemy models
├── .env                     # Environment variables
├── index.html               # Frontend UI
├── templates/
│   └── feedback.html        # Feedback UI
├── brand_blogs/             # Uploaded brand docs
│   └── {business_id}/
│       ├── doc1.txt
│       └── doc2.md
├── brand_social/
│   └── {business_id}/
├── brand_ads/
│   └── {business_id}/
└── requirements.txt
```

### Naming Conventions

**Database Tables:**
- Pattern: `{table_type}_{business_id}`
- Examples:
  - `rag_data_BAND_Foods_blog`
  - `brand_learning_BAND_Foods`
  - Sanitization: Replace `-` and `.` with `_`

**API Endpoints:**
- Pattern: `/api/{resource}/{action}`
- Examples:
  - `/api/upload`
  - `/api/feedback/submit`
  - `/api/feedback/stats/<business_id>`

**Tool Names:**
- Pattern: `{Purpose}{Type}Tool`
- Examples:
  - `KBQueryTool`
  - `TavilySearchTool`
  - `ContentScoringTool`

---

## Configuration & Environment

### Required Environment Variables

```bash
# LLM API
GROQ_API_KEY=gsk_xxx              # Groq API for Llama 3.3 70B

# Database
POSTGRES_URI=postgresql://...     # Sync connection
POSTGRES_ASYNC_URI=postgresql://...  # Async connection (for PGVector)

# Optional
TAVILY_API_KEY=tvly-xxx           # Web search (optional)
LOG_FORMAT=console                # or JSON
```

### Model Configuration

**Primary LLM:** Groq Llama-3.3-70B-Versatile
- Creative agents: temperature=0.8
- Analytical agents: temperature=0.5

**Embeddings:** FastEmbed BAAI/bge-small-en-v1.5
- Dimension: 384
- Speed: ~1000 docs/sec
- Quality: Good for semantic search

---

## Performance Characteristics

### Timing Benchmarks (Typical)

```
RAG Index Build: 5-15 seconds (for 10-20 docs)
Agent Pipeline: 30-60 seconds (5 agents sequential)
  - Research: 5-10s
  - Creative Strategy: 8-12s
  - Brand Analysis: 5-8s
  - Writing: 10-15s
  - Review + Scoring: 5-10s

Total Generation Time: ~45 seconds average
```

### Bottlenecks

1. **Sequential Agent Pipeline**
   - Can't parallelize due to dependencies
   - Each agent waits for previous output

2. **LLM API Latency**
   - Groq API: ~2-5s per call
   - 5 agents × ~3 calls each = ~15 API calls

3. **Vector Index Building**
   - First-time index: 10-20s
   - Cached index load: <1s

### Scalability Limits

**Current Limits:**
- Concurrent users: ~10-20 (thread-based)
- Database connections: 20 pool + 10 overflow
- Vector store: Tested up to 1000 docs/business

**To Scale Further:**
- Move to async (FastAPI + async SQLAlchemy)
- Add Redis for status tracking
- Implement queue system (Celery/RQ)
- Shard vector stores by business_id

---

## Error Handling Strategy

### Levels of Error Handling

**1. API Level** (`app.py`)
```python
@app.errorhandler(Exception)
def handle_error(error):
    return jsonify({
        "status": "error",
        "error": str(error),
        "traceback": traceback.format_exc()
    }), 500
```

**2. Agent Level** (`deployer.py`)
```python
def run_pipeline():
    try:
        # ... agent execution ...
    except Exception as e:
        status["overall"] = "error"
        status["result"] = {"error": str(e)}
```

**3. RAG Level**
```python
@retry(stop=stop_after_attempt(3), 
       wait=wait_exponential(multiplier=1, min=2, max=10))
def build_marketing_index(self):
    # Auto-retries on failure
```

### Fallback Strategies

**Missing Brand Docs:**
```python
fallbacks = {
    "blog": "Conversational yet professional tone...",
    "social": "Engaging, brief, personality-driven",
    "ad": "Clear value proposition, compelling CTA"
}
```

**Empty Vector Store:**
- Logs warning
- Uses fallback metrics
- Generation continues with generic voice

**Network Failures:**
- Web search: Returns "unavailable" message
- API: Retries 3 times with exponential backoff

---

## Security Considerations

### Current Security Posture

**✅ Implemented:**
- SQL injection protection (SQLAlchemy parameterized queries)
- File upload sanitization (`secure_filename()`)
- Business-specific data isolation (separate folders/tables)
- CORS enabled for frontend

**⚠️ Missing (for production):**
- Authentication/authorization
- API rate limiting
- Input validation on all endpoints
- File upload size limits
- Malicious file scanning
- Encrypted secrets storage

### Recommended Additions

1. **Authentication:** JWT tokens or session-based
2. **Rate Limiting:** Flask-Limiter (60 req/min per IP)
3. **Input Validation:** Pydantic models for all API inputs
4. **File Scanning:** ClamAV or similar before processing
5. **Secrets:** Use environment-specific vaults (AWS Secrets Manager, etc.)

---

## Testing Strategy (Currently Missing)

### Recommended Test Structure

```
tests/
├── unit/
│   ├── test_rag_system.py
│   ├── test_metrics_analyzer.py
│   └── test_learning_memory.py
├── integration/
│   ├── test_agent_pipeline.py
│   └── test_api_endpoints.py
└── e2e/
    └── test_full_generation_flow.py
```

### Critical Tests Needed

1. **RAG System**
   - Index building with sample docs
   - Query retrieval accuracy
   - Thread safety under load

2. **Metrics Analyzer**
   - Sentence/paragraph counting accuracy
   - Phrase extraction correctness
   - Scoring algorithm validation

3. **Agent Pipeline**
   - Mock LLM responses
   - Test each agent in isolation
   - Test full pipeline integration

4. **API Endpoints**
   - Upload → Index → Generate flow
   - Concurrent user handling
   - Error scenarios

---

## Future Enhancements

### High Priority

1. **Async/Await Refactor**
   - Convert to FastAPI
   - Use asyncio for concurrent operations
   - Reduces total generation time by 40%

2. **Queue System**
   - Celery or RQ for background tasks
   - Better scalability for multiple users
   - Progress tracking via Redis

3. **Agent Improvement Learning**
   - Use `reviewer_learning.agent_correct` data
   - Fine-tune scoring thresholds
   - Build reward model for RLHF

### Medium Priority

4. **Caching Layer**
   - Redis for KB queries
   - Cache common research queries
   - Reduce LLM API costs

5. **Advanced RAG**
   - Reranking (Cohere, etc.)
   - Query expansion
   - Contextual compression

6. **Multi-format Support**
   - PDF parsing (pypdf2, pdfplumber)
   - DOCX parsing (python-docx)
   - HTML extraction

### Low Priority

7. **UI Improvements**
   - Real-time WebSocket updates
   - Rich text editor
   - Visual diff for revisions

8. **Analytics Dashboard**
   - Generation success rates
   - Agent accuracy trends
   - Cost tracking

---

## Troubleshooting Guide

### Common Issues

**1. "No brand docs found"**
- **Cause:** Files uploaded to wrong folder
- **Fix:** Check `brand_{type}s/{business_id}/` exists
- **Verify:** `ls -la brand_blogs/BAND_Foods/`

**2. "Index build failed"**
- **Cause:** PostgreSQL connection issue or empty folder
- **Fix:** Check `POSTGRES_ASYNC_URI` in `.env`
- **Verify:** `psql $POSTGRES_URI -c "SELECT 1"`

**3. "Agent timeout"**
- **Cause:** Groq API rate limit or slow response
- **Fix:** Check `GROQ_API_KEY` quota
- **Workaround:** Increase timeout in requests

**4. "Empty KB response"**
- **Cause:** Vector store empty or query mismatch
- **Fix:** Rebuild index with valid documents
- **Verify:** Query vector store directly

**5. "Thread safety error"**
- **Cause:** Race condition in KB cache
- **Fix:** Ensure locks are used in `build_marketing_index()`

### Debug Mode

Enable detailed logging:
```python
# deployer.py
logger.setLevel(logging.DEBUG)

# app.py
app.run(debug=True)
```

Check logs:
```bash
tail -f app.log
grep "ERROR" app.log
```

---

## Maintenance Tasks

### Daily
- Monitor error rates in logs
- Check disk space (vector stores grow)

### Weekly
- Review human feedback submissions
- Analyze agent accuracy trends
- Clean up old generations (if storage limited)

### Monthly
- Update dependencies (`pip list --outdated`)
- Backup PostgreSQL database
- Review and archive old learning data

### Quarterly
- Performance benchmarking
- Security audit
- Cost analysis (LLM API usage)

---

## Cost Analysis

### Current Costs (per 1000 generations)

**LLM API (Groq):**
- Model: Llama-3.3-70B
- Cost: ~$0.50-$1.00 per 1000 generations
- Tokens: ~15 API calls × 2000 tokens avg = 30K tokens/generation

**Database:**
- PostgreSQL: $20-50/month (small instance)
- Vector store size: ~100MB per 100 docs

**Total Monthly (1000 gens/month):**
- LLM: $0.50-1.00
- Database: $20-50
- **Total: ~$21-51/month**

### Cost Optimization

1. **Cache KB queries** → 30% reduction
2. **Batch generations** → 20% reduction
3. **Use smaller model for research** → 40% reduction

---

## Glossary

**RAG:** Retrieval-Augmented Generation - combines document search with LLM generation

**HNSW:** Hierarchical Navigable Small World - fast similarity search algorithm

**PGVector:** PostgreSQL extension for vector storage and search

**CrewAI:** Framework for orchestrating multiple AI agents

**Signature Phrases:** Common 2-5 word sequences extracted from brand docs

**Hybrid Scoring:** Combination of automatic metrics (50%) + LLM judgment (50%)

**agent_correct:** Boolean indicating if agent's approval prediction matched human's

**generation_id:** UUID tracking individual content generations

---

## Quick Reference

### Key Files
- `deployer.py` - Agent pipeline (800+ lines)
- `app.py` - Flask API (600+ lines)
- `db.py` - Database models (400+ lines)

### Key Classes
- `Marketing_Rag_System` - RAG indexing
- `BrandMetricsAnalyzer` - Structural analysis
- `BrandLearningMemory` - Pattern storage
- `ContentCrewFactory` - Agent orchestration

### Key Functions
- `get_crew()` - Main entry point
- `build_marketing_index()` - RAG setup
- `save_human_feedback()` - Learning storage

### Database Tables
- `users` - Business entities
- `brand_documents` - Uploaded files
- `generations` - Historical generations
- `reviewer_learning` - Agent learning data
- `brand_learning_{id}` - Pattern memory

---

## Document Version History

- **v1.0** (2026-01-28) - Initial architecture documentation