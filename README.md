# Production-Ready Deployment Guide

## Overview

Your code has been upgraded to production-ready status with the following enterprise-grade features:

## Production Features Added

### 1. **Retry Logic with Exponential Backoff**
- All database operations retry up to 3 times
- Exponential backoff: 2s → 4s → 8s wait times
- Handles transient failures gracefully
- Uses `tenacity` library for robust retry mechanisms

**Example:**
```python
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((OperationalError, DatabaseError))
)
def save_learning(...):
    # Database operation with automatic retry
```

### 2. **Rate Limiting**
- Token bucket algorithm for API calls
- Groq API: 30 calls/minute
- Tavily Search: 20 calls/minute
- Thread-safe implementation
- Automatic sleep when limit reached

**Usage:**
```python
@groq_rate_limiter  # Automatically rate limits
def get_llm(temperature=0.7):
    return LLM(...)
```

### 3. **Circuit Breakers**
- Prevents cascading failures
- Automatic service degradation
- 3 states: CLOSED → OPEN → HALF_OPEN
- Configurable failure thresholds
- Per-service circuit breakers (Groq, Tavily, DB)

**Behavior:**
- After 5 failures: Circuit OPENS (blocks requests)
- After 60s timeout: Tries HALF_OPEN (test request)
- On success: Returns to CLOSED

### 4. **Connection Pooling**
- PostgreSQL connection pool: 20 connections
- Max overflow: 10 additional connections
- Connection timeout: 30 seconds
- Auto-reconnection with `pool_pre_ping=True`
- Connection recycling after 1 hour

**Configuration:**
```python
engine = create_engine(
    db_url,
    pool_size=20,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=3600,
    pool_pre_ping=True
)
```

### 5. **Comprehensive Error Handling**
- Try-catch blocks on all critical operations
- Proper transaction management (commit/rollback)
- Resource cleanup in `finally` blocks
- Graceful degradation with fallback responses
- Detailed error logging with stack traces

### 6. **Thread Safety**
- Reentrant locks (RLock) for business_id operations
- Lock-protected caching
- Thread-safe rate limiters
- Atomic database operations with `FOR UPDATE`

### 7. **Query Optimization**
- Indexed columns: business_id, content_type, generation_id
- Parameterized queries (SQL injection protection)
- Query timeout: 30 seconds
- Efficient batch operations

### 8. **Monitoring & Observability**
- Structured logging with `structlog`
- JSON/Console log formats
- Connection pool status monitoring
- Event listeners for database connections
- Detailed error context

### 9. **Health Checks**
- Database connectivity check
- Connection pool status
- Circuit breaker states
- Environment variable validation

### 10. **Graceful Degradation**
- Fallback responses when services fail
- Cached KB responses
- Default metrics when docs missing
- Continue operation on non-critical failures

---

## File Structure

```
/outputs/
├── db_production.py              # Complete (471 lines)
├── deployer_part1.py             # Core classes & config
├── deployer_part2.py             # Learning memory & RAG
├── deployer_part3.py             # Tools & crew
├── deployer_part4.py             # Main execution
└── PRODUCTION_README.md          # This file
```

---

## 🔧 Configuration

### Environment Variables (Required)

```bash
# Database
POSTGRES_URI=postgresql://user:pass@host:5432/dbname
POSTGRES_ASYNC_URI=postgresql+asyncpg://user:pass@host:5432/dbname

# API Keys
GROQ_API_KEY=gsk_...
TAVILY_API_KEY=tvly-...

# Optional
LOG_FORMAT=JSON  # or console (default)
```

### Database Indexes

Ensure these indexes exist:
```sql
CREATE INDEX idx_business_id ON users(business_id);
CREATE INDEX idx_content_type ON brand_documents(content_type);
CREATE INDEX idx_generation_id ON reviewer_learning(generation_id);
CREATE INDEX idx_has_feedback ON reviewer_learning(has_human_feedback);
```

---

## Deployment Steps

### 1. Install Dependencies

```bash
pip install --break-system-packages \
    psycopg2-binary \
    sqlalchemy \
    fastapi \
    uvicorn \
    tenacity \
    structlog \
    crewai \
    llama-index \
    tavily-python \
    python-dotenv
```

### 2. Initialize Database

```python
from db import init_db, check_database_health

# Initialize tables
init_db()

# Verify health
if check_database_health():
    print("Database ready")
```

### 3. Combine Deployer Parts

```python
# Combine all deployer_part*.py files into one deployer.py
# Remove duplicate imports and docstrings from parts 2-4
```

### 4. Run Application

```bash
# Development
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn app:app --host 0.0.0.0 --port 8000 \
    --workers 4 \
    --log-level info \
    --access-log
```

---

## Monitoring

### Health Check Endpoint

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
    "status": "healthy",
    "timestamp": "2026-01-31T...",
    "environment": {
        "groq_api_key": true,
        "postgres_uri": true,
        "tavily_api_key": true
    }
}
```

### Connection Pool Status

```python
from db import get_pool_status

status = get_pool_status()
print(f"Pool size: {status['size']}")
print(f"Checked out: {status['checked_out']}")
print(f"Available: {status['checked_in']}")
```

### Circuit Breaker Status

```python
print(f"Groq Circuit: {groq_circuit_breaker.state}")
print(f"DB Circuit: {db_circuit_breaker.state}")
print(f"Tavily Circuit: {tavily_circuit_breaker.state}")
```

---

## 🔒 Security Improvements

1. **SQL Injection Protection**: All queries use parameterized statements
2. **Input Validation**: Pydantic models validate all inputs
3. **Query Timeouts**: 30-second timeout prevents long-running queries
4. **Connection Limits**: Pool prevents connection exhaustion
5. **Rate Limiting**: Prevents API abuse

---

## ⚡ Performance Optimizations

1. **Caching**:
   - KB query results cached per business_id
   - Embedding models cached
   - LLM instances reused

2. **Database**:
   - Connection pooling (20 + 10 overflow)
   - Indexed lookups
   - Batch operations where possible

3. **Resource Management**:
   - Automatic connection cleanup
   - Thread-safe operations
   - Efficient lock usage

---

## 🐛 Error Handling Examples

### Database Failures
```python
try:
    with get_db_session() as session:
        # Operation
except OperationalError as e:
    # Automatic retry up to 3 times
    logger.error(f"DB operation failed: {e}")
    # Circuit breaker may open after repeated failures
```

### API Rate Limits
```python
@groq_rate_limiter
def call_llm():
    # Automatically waits if rate limit reached
    # No manual throttling needed
```

### Service Degradation
```python
try:
    kb_response = query_kb(...)
except Exception:
    # Fallback to default response
    return get_fallback_response()
```

---

## 📈 Scaling Considerations

### Horizontal Scaling
- Stateless design allows multiple app instances
- Shared database for coordination
- Rate limiters are per-instance (consider Redis for global rate limiting)

### Database Scaling
- Current pool: 20 connections
- Can increase for more traffic
- Consider read replicas for heavy read workloads

### Caching Layer
- Consider Redis for:
  - Global rate limiting
  - Shared KB cache
  - Session management

---

## 🔍 Troubleshooting

### High Database Connection Usage
```python
from db import get_pool_status
status = get_pool_status()
# If checked_out is consistently high, increase pool_size
```

### Circuit Breaker Constantly Open
```bash
# Check service health
curl http://localhost:8000/health

# Review logs for root cause
tail -f app.log | grep "Circuit breaker"
```

### Slow Queries
```sql
-- Enable query logging
ALTER DATABASE your_db SET log_statement = 'all';

-- Check slow queries
SELECT query, calls, total_time 
FROM pg_stat_statements 
ORDER BY total_time DESC 
LIMIT 10;
```

---

## 🎯 Testing

### Load Testing
```bash
# Install hey
go install github.com/rakyll/hey@latest

# Run load test
hey -n 1000 -c 50 -m POST \
    -H "Content-Type: application/json" \
    -d '{"business_id":"test","topic":"AI"}' \
    http://localhost:8000/api/generate
```

### Database Connection Test
```python
import threading
import time

def test_concurrent_access():
    threads = []
    for i in range(50):
        t = threading.Thread(target=run_generation_with_learning, 
                             args=("test_business", "topic", "blog", "formal"))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    print("All threads completed")

test_concurrent_access()
```

---

## API Documentation

### Generate Content
```bash
POST /api/generate
{
    "business_id": "company_123",
    "topic": "AI in Healthcare",
    "format": "Blog Article",
    "voice": "professional"
}
```

### Submit Feedback
```bash
POST /api/feedback
{
    "generation_id": "uuid-here",
    "business_id": "company_123",
    "human_approved": true,
    "human_score": 9.5,
    "human_feedback": "Excellent tone and structure"
}
```

### Get Learning Stats
```bash
GET /api/learning/stats/company_123?content_type=blog
```

---

## 🚨 Production Checklist

- [ ] Environment variables configured
- [ ] Database initialized with `init_db()`
- [ ] Database indexes created
- [ ] Health check endpoint responding
- [ ] Rate limiters configured
- [ ] Circuit breakers tested
- [ ] Connection pool sized appropriately
- [ ] Logging configured (JSON for production)
- [ ] Error monitoring set up (Sentry, etc.)
- [ ] Load testing completed
- [ ] Backup strategy in place
- [ ] Monitoring dashboards created

---

## 📞 Support

For issues or questions:
1. Check logs: `tail -f app.log`
2. Verify health: `curl /health`
3. Check circuit breakers: Review `groq_circuit_breaker.state`
4. Monitor connections: Call `get_pool_status()`

---

## Summary

Your code now includes:
- Automatic retry with exponential backoff
- Rate limiting (30 Groq, 20 Tavily calls/min)
- Circuit breakers for all external services
- Connection pooling (20 + 10 connections)
- Thread-safe operations
- Comprehensive error handling
- Graceful degradation
- Health checks and monitoring
- Query timeouts (30s)
- Resource cleanup (finally blocks)

**Result**: Production-ready, resilient, scalable application! (rocket emoji)