# Refactoring & Improvement Roadmap

## Current Assessment

### Complexity Score: 7/10

**What's Good:**
- Clear separation of concerns (API, Core Logic, Database)
- Thread-safe caching implemented
- Hybrid learning system (auto + human)
- Comprehensive error handling

**What Needs Work:**
- Monolithic files (800+ lines each)
- No async/await (blocks on I/O)
- Missing test coverage
- Tight coupling in some areas
- Configuration hardcoded

---

## Phase 1: Immediate Improvements (Week 1-2)

### 1.1 Split `deployer.py` into Modules

**Current:** 800+ lines in single file

**Target Structure:**
```
core/
├── __init__.py
├── agents/
│   ├── __init__.py
│   ├── researcher.py
│   ├── strategist.py
│   ├── analyst.py
│   ├── writer.py
│   └── reviewer.py
├── tools/
│   ├── __init__.py
│   ├── kb_query.py
│   ├── tavily_search.py
│   ├── learning_memory.py
│   ├── metrics.py
│   └── scoring.py
├── rag/
│   ├── __init__.py
│   ├── rag_system.py
│   ├── embeddings.py
│   └── indexing.py
├── learning/
│   ├── __init__.py
│   ├── metrics_analyzer.py
│   ├── learning_memory.py
│   └── feedback.py
└── crew_factory.py
```

**Benefits:**
- Easier to test individual components
- Clearer dependencies
- Better code navigation
- Enables parallel development

**Migration Steps:**

```python
# Step 1: Create core/__init__.py
from .crew_factory import ContentCrewFactory, get_crew
from .rag import Marketing_Rag_System
from .learning import BrandLearningMemory, BrandMetricsAnalyzer

__all__ = [
    'ContentCrewFactory',
    'get_crew',
    'Marketing_Rag_System',
    'BrandLearningMemory',
    'BrandMetricsAnalyzer'
]

# Step 2: Move agents to separate files
# core/agents/researcher.py
from crewai import Agent, Task

def create_researcher_agent(llm, tools):
    return Agent(
        role="Research Analyst",
        goal="Find current, factual information...",
        backstory="...",
        tools=tools,
        llm=llm,
        verbose=True
    )

def create_research_task(agent):
    return Task(
        description="...",
        expected_output="...",
        agent=agent
    )

# Step 3: Update app.py imports
from core import get_crew
from core.learning import save_human_feedback, get_feedback_stats
```

### 1.2 Extract Configuration

**Current:** Environment variables scattered, magic numbers

**Target:** Centralized config management

```python
# core/config.py
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # API Keys
    groq_api_key: str
    tavily_api_key: Optional[str] = None
    
    # Database
    postgres_uri: str
    postgres_async_uri: str
    
    # RAG Settings
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    
    # HNSW Index
    hnsw_m: int = 16
    hnsw_ef_construction: int = 64
    hnsw_ef_search: int = 40
    
    # Content Type Chunk Sizes
    blog_chunk_size: int = 512
    social_chunk_size: int = 256
    ad_chunk_size: int = 256
    chunk_overlap: int = 50
    
    # Agent Settings
    creative_temperature: float = 0.8
    analytical_temperature: float = 0.5
    
    # Scoring Thresholds
    excellent_threshold: float = 9.0
    good_threshold: float = 8.0
    revision_threshold: float = 7.0
    
    # Retrieval Settings
    blog_top_k: int = 5
    social_top_k: int = 3
    ad_top_k: int = 3
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Singleton instance
settings = Settings()

# Usage:
from core.config import settings
llm = LLM(model="groq/llama-3.3-70b", temperature=settings.creative_temperature)
```

### 1.3 Add Basic Tests

**Start with critical paths:**

```python
# tests/test_metrics_analyzer.py
import pytest
from core.learning import BrandMetricsAnalyzer

def test_sentence_length_calculation():
    analyzer = BrandMetricsAnalyzer("test_business", "blog")
    
    # Mock brand docs
    test_text = """
    This is sentence one. This is sentence two with more words.
    This is sentence three.
    """
    
    metrics = analyzer._analyze_text(test_text)
    
    assert 'avg_sentence_length' in metrics
    assert metrics['sentence_count'] == 3
    assert 5 <= metrics['avg_sentence_length'] <= 7

def test_signature_phrase_extraction():
    analyzer = BrandMetricsAnalyzer("test_business", "blog")
    
    test_text = """
    Customer experience is key. Great customer experience matters.
    Customer experience drives loyalty.
    """
    
    metrics = analyzer._analyze_text(test_text)
    
    assert 'customer experience' in metrics['common_phrases']

# tests/test_rag_system.py
import pytest
from core.rag import Marketing_Rag_System
from llama_index.core import Document

def test_index_building(tmp_path):
    # Create test documents
    test_file = tmp_path / "test.txt"
    test_file.write_text("This is test brand content.")
    
    rag = Marketing_Rag_System(
        data_path=str(tmp_path),
        business_id="test_business",
        content_type="blog"
    )
    
    index = rag.build_marketing_index()
    assert index is not None

# tests/test_api.py
import pytest
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health_endpoint(client):
    rv = client.get('/api/health')
    assert rv.status_code == 200
    data = rv.get_json()
    assert data['status'] == 'healthy'

def test_upload_requires_business_id(client):
    rv = client.post('/api/upload', data={})
    assert rv.status_code == 400
    data = rv.get_json()
    assert 'business_id' in data['error']
```

**Run tests:**
```bash
pip install pytest pytest-cov
pytest tests/ -v --cov=core --cov=app
```

---

## Phase 2: Async Refactor (Week 3-4)

### 2.1 Convert to FastAPI

**Why:** Non-blocking I/O, better performance, automatic OpenAPI docs

**Migration Strategy:**

```python
# app_async.py (new file)
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio

app = FastAPI(title="Marketing Content Generator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class GenerateRequest(BaseModel):
    business_id: str
    topic: str
    format: str = "Blog Article"
    voice: str = "formal"

class GenerateResponse(BaseModel):
    status: str
    message: str
    generation_id: str

class StatusResponse(BaseModel):
    overall: str
    agents: dict
    result: dict | None

# Endpoints
@app.post("/api/generate", response_model=GenerateResponse)
async def generate_content(
    request: GenerateRequest,
    background_tasks: BackgroundTasks
):
    user = get_or_create_user(request.business_id)
    generation_id = str(uuid.uuid4())
    
    # Run in background
    background_tasks.add_task(
        run_generation_pipeline,
        generation_id=generation_id,
        business_id=request.business_id,
        topic=request.topic,
        format_type=request.format,
        voice=request.voice
    )
    
    return GenerateResponse(
        status="success",
        message="Generation started",
        generation_id=generation_id
    )

@app.get("/api/status", response_model=StatusResponse)
async def get_status(business_id: str):
    status = get_user_status(business_id)
    return StatusResponse(**status)

# Background task
async def run_generation_pipeline(
    generation_id: str,
    business_id: str,
    topic: str,
    format_type: str,
    voice: str
):
    try:
        # Use async crew (if CrewAI supports it, or use asyncio.to_thread)
        crew, kb, learning, metrics = await asyncio.to_thread(
            get_crew,
            business_id=business_id,
            topic=topic,
            format_type=format_type,
            voice=voice
        )
        
        result = await asyncio.to_thread(
            crew.kickoff,
            inputs={"topic": topic, "format": format_type, "voice": voice}
        )
        
        # Save result
        await save_generation_async(generation_id, result)
        
    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        await save_error_async(generation_id, str(e))
```

### 2.2 Async Database Access

```python
# db_async.py
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

engine = create_async_engine(
    os.getenv("POSTGRES_ASYNC_URI"),
    echo=False,
    pool_size=20
)

async_session_maker = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

async def save_generation_async(generation_id: str, result: str):
    async with async_session_maker() as session:
        generation = ReviewerLearning(
            generation_id=generation_id,
            business_id=...,
            generated_content=...,
            # ...
        )
        session.add(generation)
        await session.commit()
```

---

## Phase 3: Advanced Improvements (Week 5-8)

### 3.1 Add Redis for Caching & Status

**Why:** In-memory speed, pub/sub for real-time updates

```python
# core/cache.py
import redis
import json

class RedisCache:
    def __init__(self):
        self.redis = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=6379,
            decode_responses=True
        )
    
    def set_status(self, business_id: str, status: dict):
        self.redis.setex(
            f"status:{business_id}",
            300,  # 5 min TTL
            json.dumps(status)
        )
    
    def get_status(self, business_id: str) -> dict:
        data = self.redis.get(f"status:{business_id}")
        return json.loads(data) if data else None
    
    def cache_kb_query(self, business_id: str, query: str, result: str):
        key = f"kb:{business_id}:{hash(query)}"
        self.redis.setex(key, 3600, result)  # 1 hour
    
    def get_kb_query(self, business_id: str, query: str) -> str | None:
        key = f"kb:{business_id}:{hash(query)}"
        return self.redis.get(key)

# Usage in app
cache = RedisCache()

@app.get("/api/status")
async def get_status(business_id: str):
    status = cache.get_status(business_id)
    if not status:
        raise HTTPException(404, "No active generation")
    return status
```

### 3.2 Add WebSocket for Real-Time Updates

```python
# app_async.py
from fastapi import WebSocket
from typing import Dict

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, business_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[business_id] = websocket
    
    def disconnect(self, business_id: str):
        self.active_connections.pop(business_id, None)
    
    async def send_status(self, business_id: str, status: dict):
        if business_id in self.active_connections:
            await self.active_connections[business_id].send_json(status)

manager = ConnectionManager()

@app.websocket("/ws/status/{business_id}")
async def websocket_status(websocket: WebSocket, business_id: str):
    await manager.connect(business_id, websocket)
    try:
        while True:
            # Keep connection alive
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        manager.disconnect(business_id)

# In generation pipeline
async def update_agent_status(business_id: str, agent: str, status: str):
    status_dict = get_user_status(business_id)
    status_dict[agent]["status"] = status
    await manager.send_status(business_id, status_dict)
```

### 3.3 Add Celery for Background Tasks

**Why:** Distributed task queue, better scalability

```python
# celery_app.py
from celery import Celery

celery = Celery(
    'content_generator',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/1'
)

@celery.task
def generate_content_task(
    generation_id: str,
    business_id: str,
    topic: str,
    format_type: str,
    voice: str
):
    crew, kb, learning, metrics = get_crew(
        business_id=business_id,
        topic=topic,
        format_type=format_type,
        voice=voice
    )
    
    result = crew.kickoff(inputs={
        "topic": topic,
        "format": format_type,
        "voice": voice
    })
    
    save_generation(generation_id, result)
    return {"status": "completed", "generation_id": generation_id}

# In app_async.py
from celery_app import generate_content_task

@app.post("/api/generate")
async def generate_content(request: GenerateRequest):
    generation_id = str(uuid.uuid4())
    
    # Enqueue task
    generate_content_task.delay(
        generation_id=generation_id,
        business_id=request.business_id,
        topic=request.topic,
        format_type=request.format,
        voice=request.voice
    )
    
    return {"generation_id": generation_id, "status": "queued"}
```

---

## Phase 4: Production Readiness (Week 9-12)

### 4.1 Add Comprehensive Monitoring

```python
# core/monitoring.py
from prometheus_client import Counter, Histogram, Gauge
import time

# Metrics
generation_counter = Counter(
    'generations_total',
    'Total generations',
    ['business_id', 'content_type', 'status']
)

generation_duration = Histogram(
    'generation_duration_seconds',
    'Generation duration',
    ['business_id', 'content_type']
)

agent_accuracy = Gauge(
    'agent_accuracy_percent',
    'Agent approval accuracy',
    ['business_id']
)

# Usage
class MonitoredCrewFactory:
    def create_crew(self):
        start = time.time()
        try:
            crew = super().create_crew()
            duration = time.time() - start
            
            generation_duration.labels(
                business_id=self.business_id,
                content_type=self.content_type
            ).observe(duration)
            
            generation_counter.labels(
                business_id=self.business_id,
                content_type=self.content_type,
                status='success'
            ).inc()
            
            return crew
        except Exception as e:
            generation_counter.labels(
                business_id=self.business_id,
                content_type=self.content_type,
                status='error'
            ).inc()
            raise

# Expose metrics
from prometheus_client import make_asgi_app

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)
```

### 4.2 Add Rate Limiting

```python
# app_async.py
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/api/generate")
@limiter.limit("10/minute")
async def generate_content(request: Request, data: GenerateRequest):
    # ... existing logic ...
    pass
```

### 4.3 Add Health Checks

```python
@app.get("/health/ready")
async def readiness_check():
    checks = {
        "database": await check_database(),
        "redis": await check_redis(),
        "vector_store": await check_vector_store(),
        "llm_api": await check_llm_api()
    }
    
    all_healthy = all(checks.values())
    
    return {
        "status": "ready" if all_healthy else "not_ready",
        "checks": checks
    }

@app.get("/health/live")
async def liveness_check():
    return {"status": "alive"}

async def check_database():
    try:
        async with async_session_maker() as session:
            await session.execute("SELECT 1")
        return True
    except:
        return False
```

---

## Phase 5: Optimization (Ongoing)

### 5.1 Query Optimization

```python
# Batch KB queries
async def batch_kb_queries(queries: List[str], business_id: str):
    # Instead of 4 separate queries, combine similar ones
    combined_query = " ".join(queries)
    result = await query_kb_async(combined_query, business_id)
    return result

# Cache expensive operations
@lru_cache(maxsize=100)
def get_brand_metrics(business_id: str, content_type: str):
    return BrandMetricsAnalyzer(business_id, content_type).metrics
```

### 5.2 Reduce LLM API Calls

```python
# Combine similar agent tasks
class CombinedAnalyst(Agent):
    """Combines Creative + Brand analysis in one LLM call"""
    
    def analyze(self, research, kb):
        prompt = f"""
        Given research: {research}
        
        Task 1: Propose 2 creative angles
        Task 2: Extract brand metrics
        
        Return JSON: {{"angles": [...], "metrics": {{...}}}}
        """
        # Single API call instead of 2
        result = self.llm(prompt)
        return parse_combined_result(result)
```

### 5.3 Parallel Agent Execution (where possible)

```python
async def run_independent_tasks():
    # Research and Brand Analysis don't depend on each other
    research, brand_metrics = await asyncio.gather(
        research_agent.run(),
        brand_analyst.run()
    )
    
    # Then run dependent tasks
    creative_angles = await creative_agent.run(research, brand_metrics)
    content = await writer_agent.run(creative_angles, brand_metrics)
    return content
```

---

## Architecture Decision Records (ADRs)

Start documenting key decisions:

```markdown
# ADR-001: Use Groq for LLM API

**Date:** 2026-01-28
**Status:** Accepted

## Context
Need fast, reliable LLM for 5-agent pipeline (15+ API calls per generation)

## Decision
Use Groq with Llama-3.3-70B

## Consequences
- Very fast inference (< 1s per call)
- Good quality for marketing content
- Affordable pricing
- Rate limits (may need queuing)
- Vendor lock-in

## Alternatives Considered
- OpenAI GPT-4: Too expensive
- Anthropic Claude: Slower, more expensive
- Local model: Infrastructure complexity
```

---

## Recommended Reading Order

1. **ARCHITECTURE.md** - Start here for big picture
2. **DIAGRAMS.md** - Visual understanding
3. **This file** - Implementation roadmap
4. **Code** - Now dive into actual implementation

---

## Quick Wins Checklist

Start with these for immediate improvement:

- [ ] Split deployer.py into modules
- [ ] Extract config.py
- [ ] Add health check endpoint
- [ ] Write 5 critical tests
- [ ] Add logging levels (DEBUG, INFO, ERROR)
- [ ] Document environment variables in README
- [ ] Add input validation with Pydantic
- [ ] Implement rate limiting (basic)
- [ ] Add monitoring endpoint (/metrics)
- [ ] Create docker-compose.yml for local dev

---

## Resources

**Learning:**
- FastAPI: https://fastapi.tiangolo.com/
- CrewAI: https://docs.crewai.com/
- LlamaIndex: https://docs.llamaindex.ai/
- SQLAlchemy Async: https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html

**Tools:**
- pytest: Testing framework
- black: Code formatting
- mypy: Type checking
- ruff: Fast linting

**Monitoring:**
- Prometheus: Metrics collection
- Grafana: Visualization
- Sentry: Error tracking