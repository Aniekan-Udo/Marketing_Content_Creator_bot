# api.py - PRODUCTION-READY FastAPI APPLICATION

"""
Production-ready FastAPI application with:
- Rate limiting per endpoint
- Comprehensive error handling
- Request/response logging
- Health checks and monitoring
- Graceful shutdown
- CORS configuration
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
from datetime import datetime
from contextlib import asynccontextmanager
import os
import time
import structlog
import psycopg2
from dotenv import load_dotenv

load_dotenv()

# Database imports
# Database imports
try:
    from .db import (
        session_maker, 
        init_db, 
        User, 
        BrandDocument, 
        ReviewerLearning,
        check_database_health,
        get_pool_status
    )
    # Business logic imports
    from .deployer import (
        run_generation_with_learning,
        update_generation_with_human_feedback,
        verify_learning_loop,
        BrandLearningMemory,
        groq_circuit_breaker,
        tavily_circuit_breaker,
        db_circuit_breaker,
        logger
    )
except ImportError:
    from db import (
        session_maker, 
        init_db, 
        User, 
        BrandDocument, 
        ReviewerLearning,
        check_database_health,
        get_pool_status
    )
    # Business logic imports
    from deployer import (
        run_generation_with_learning,
        update_generation_with_human_feedback,
        verify_learning_loop,
        BrandLearningMemory,
        groq_circuit_breaker,
        tavily_circuit_breaker,
        db_circuit_breaker,
        logger
    )

# ===== RATE LIMITING =====

from collections import defaultdict
from threading import Lock

class EndpointRateLimiter:
    """
    Simple in-memory rate limiter for API endpoints.
    For production, consider Redis-based rate limiting.
    """
    
    def __init__(self):
        self.requests = defaultdict(list)
        self.lock = Lock()
    
    def is_allowed(self, identifier: str, max_requests: int, window: int = 60) -> bool:
        """
        Check if request is allowed.
        
        Args:
            identifier: IP address or API key
            max_requests: Max requests in window
            window: Time window in seconds
            
        Returns:
            bool: True if allowed
        """
        with self.lock:
            now = time.time()
            
            # Clean old requests
            self.requests[identifier] = [
                timestamp for timestamp in self.requests[identifier]
                if now - timestamp < window
            ]
            
            # Check limit
            if len(self.requests[identifier]) >= max_requests:
                return False
            
            # Add new request
            self.requests[identifier].append(now)
            return True

rate_limiter = EndpointRateLimiter()


async def rate_limit_dependency(request: Request):
    """FastAPI dependency for rate limiting"""
    identifier = request.client.host
    
    if not rate_limiter.is_allowed(identifier, max_requests=10, window=60):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later."
        )


# ===== REQUEST LOGGING MIDDLEWARE =====

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("Starting BrandGuard AI API")
    try:
        init_db()
        logger.info("✅ Database initialized")
        
        # Check database health
        if check_database_health():
            logger.info("✅ Database health check passed")
        else:
            logger.warning("⚠️ Database health check failed")
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}", exc_info=True)
        raise

    yield  # Application runs

    # Shutdown
    logger.info("Shutting down BrandGuard AI API")
    logger.info(f"Final pool status: {get_pool_status()}")


# ===== FASTAPI APP =====

app = FastAPI(
    title="BrandGuard AI API",
    description="Production-ready marketing content generation with brand learning",
    version="2.0.0",
    lifespan=lifespan
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for your domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests with timing"""
    start_time = time.time()
    
    # Log request
    logger.info(
        "request_started",
        method=request.method,
        path=request.url.path,
        client=request.client.host if request.client else "unknown"
    )
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Log response
        logger.info(
            "request_completed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=round(process_time * 1000, 2)
        )
        
        response.headers["X-Process-Time"] = str(process_time)
        return response
        
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(
            "request_failed",
            method=request.method,
            path=request.url.path,
            error=str(e),
            duration_ms=round(process_time * 1000, 2),
            exc_info=True
        )
        raise


# Static files
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


# ===== PYDANTIC MODELS WITH VALIDATION =====

class GenerateRequest(BaseModel):
    business_id: str
    topic: str
    format_type: str = Field(default="Blog Article", alias="format")
    voice: str = "formal"
    
    @field_validator('business_id')
    @classmethod
    def validate_business_id(cls, v):
        if not v or len(v) < 3:
            raise ValueError("business_id must be at least 3 characters")
        return v
    
    @field_validator('topic')
    @classmethod
    def validate_topic(cls, v):
        if not v or len(v) < 5:
            raise ValueError("topic must be at least 5 characters")
        return v


class FeedbackRequest(BaseModel):
    generation_id: str
    business_id: str
    human_approved: bool
    human_score: float = Field(ge=0, le=10)
    human_feedback: str
    
    @field_validator('human_feedback')
    @classmethod
    def validate_feedback(cls, v):
        if not v or len(v) < 10:
            raise ValueError("feedback must be at least 10 characters")
        return v


class GenerateResponse(BaseModel):
    status: str
    generation_id: str
    content: str
    auto_score: float
    creative_angle: str
    topic: str
    format_type: str
    business_id: str
    timestamp: str
    processing_time_ms: Optional[float] = None


class FeedbackResponse(BaseModel):
    success: bool
    message: str
    generation_id: str
    agent_accuracy: Optional[float] = None


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    database: dict
    circuit_breakers: dict
    connection_pool: dict


# ===== ROUTES =====

@app.get("/")
async def serve_index():
    """Serve the main frontend"""
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {
        "name": "BrandGuard AI API",
        "version": "2.0.0",
        "status": "operational",
        "docs": "/docs"
    }


@app.get("/feedback")
async def serve_feedback_page():
    """Serve the feedback page"""
    if os.path.exists("feedback.html"):
        return FileResponse("feedback.html")
    return JSONResponse(status_code=404, content={"message": "Feedback page not found"})


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Comprehensive health check endpoint.
    Checks database, circuit breakers, and connection pool.
    """
    timestamp = datetime.utcnow().isoformat()
    
    try:
        # Check database
        try:
            db_healthy = check_database_health()
            db_status = {
                "healthy": db_healthy,
                "pool": get_pool_status()
            }
        except Exception as e:
            db_status = {
                "healthy": False,
                "error": str(e),
                "pool": {"status": "error"}
            }
        
        # Check circuit breakers
        try:
            circuit_status = {
                "groq": getattr(groq_circuit_breaker, 'state', 'unknown'),
                "tavily": getattr(tavily_circuit_breaker, 'state', 'unknown'),
                "database": getattr(db_circuit_breaker, 'state', 'unknown')
            }
        except Exception:
            circuit_status = {"status": "error"}
        
        # Overall status
        overall_healthy = db_status.get("healthy", False) and all(
            state == "closed" for state in circuit_status.values()
            if isinstance(state, str)
        )
        
        return {
            "status": "healthy" if overall_healthy else "degraded",
            "timestamp": timestamp,
            "database": db_status,
            "circuit_breakers": circuit_status,
            "connection_pool": get_pool_status()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=200, # Still return 200 but with error info
            content={
                "status": "degraded",
                "timestamp": timestamp,
                "error": str(e),
                "database": {"healthy": False},
                "circuit_breakers": {},
                "connection_pool": {}
            }
        )


@app.post("/api/upload", dependencies=[Depends(rate_limit_dependency)])
async def upload_documents(
    files: List[UploadFile] = File(...),
    business_id: str = Form(...),
    content_type: str = Form(...)
):
    """
    Upload brand documents with validation and error handling.
    
    - **files**: List of files to upload
    - **business_id**: Business identifier
    - **content_type**: Type (blog/social/ad)
    """
    try:
        # Validate content_type
        if content_type not in ['blog', 'social', 'ad']:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid content_type: {content_type}. Must be blog/social/ad"
            )
        
        # Validate file count
        if len(files) > 50:
            raise HTTPException(
                status_code=400,
                detail="Maximum 50 files per upload"
            )

        # Get or create user
        with session_maker() as session:
            user = session.query(User).filter_by(business_id=business_id).first()
            if not user:
                user = User(
                    email=f"{business_id}@brandguard.ai",
                    business_id=business_id,
                    business_name=business_id,
                    industry='Marketing'
                )
                session.add(user)
                session.commit()
                session.refresh(user)

        # Create directory
        folder_mapping = {
            'blog': f'brand_blogs/{business_id}',
            'social': f'brand_social/{business_id}',
            'ad': f'brand_ads/{business_id}'
        }

        target_folder = folder_mapping[content_type]
        os.makedirs(target_folder, exist_ok=True)

        uploaded_files = []
        total_size = 0

        # Save files
        with session_maker() as session:
            for file in files:
                if file.filename:
                    # Validate file size (10MB limit)
                    content = await file.read()
                    if len(content) > 10 * 1024 * 1024:
                        raise HTTPException(
                            status_code=400,
                            detail=f"File {file.filename} exceeds 10MB limit"
                        )
                    
                    total_size += len(content)
                    
                    filename = file.filename
                    filepath = os.path.join(target_folder, filename)

                    # Save file
                    with open(filepath, 'wb') as f:
                        f.write(content)

                    file_size = os.path.getsize(filepath)

                    # Save to database
                    doc = BrandDocument(
                        user_id=user.id,
                        business_id=business_id,
                        filename=filename,
                        content_type=content_type,
                        file_path=filepath,
                        file_size_bytes=file_size,
                        status='uploaded'
                    )
                    session.add(doc)
                    uploaded_files.append(filename)

            session.commit()

        logger.info(f"✅ Uploaded {len(uploaded_files)} files ({total_size/1024:.1f}KB) for {business_id}")

        return {
            "success": True,
            "message": f"Uploaded {len(uploaded_files)} files",
            "files": uploaded_files,
            "business_id": business_id,
            "content_type": content_type,
            "total_size_kb": round(total_size / 1024, 2)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.post("/api/generate", response_model=GenerateResponse, dependencies=[Depends(rate_limit_dependency)])
async def generate_content(request: GenerateRequest):
    """
    Generate marketing content with comprehensive error handling.
    
    Rate limited to 10 requests per minute per IP.
    """
    start_time = time.time()
    
    try:
        logger.info(f"🚀 Generation request: {request.business_id} - {request.topic}")
        
        result_dict, generation_id = run_generation_with_learning(
            business_id=request.business_id,
            topic=request.topic,
            format_type=request.format_type,
            voice=request.voice
        )
        
        processing_time = round((time.time() - start_time) * 1000, 2)
        
        logger.info(f"✅ Generation complete: {generation_id} ({processing_time}ms)")
        
        return GenerateResponse(
            status="success",
            generation_id=generation_id,
            content=result_dict["content"],
            auto_score=result_dict["auto_score"],
            creative_angle=result_dict["creative_angle"],
            topic=request.topic,
            format_type=request.format_type,
            business_id=request.business_id,
            timestamp=datetime.utcnow().isoformat(),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        
        # Check if circuit breaker opened
        if "Circuit breaker OPEN" in str(e):
            raise HTTPException(
                status_code=503,
                detail="Service temporarily unavailable. Please try again in a moment."
            )
        
        raise HTTPException(
            status_code=500, 
            detail=f"Generation failed: {str(e)[:200]}"
        )


@app.post("/api/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    """
    Submit human feedback with validation and error handling.
    """
    try:
        logger.info(f"📝 Feedback for {request.generation_id}")

        success = update_generation_with_human_feedback(
            generation_id=request.generation_id,
            business_id=request.business_id,
            human_approved=request.human_approved,
            human_score=request.human_score,
            human_feedback=request.human_feedback
        )

        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Generation {request.generation_id} not found"
            )

        memory = BrandLearningMemory(business_id=request.business_id)
        stats = memory.get_learning_stats("blog")

        return FeedbackResponse(
            success=True,
            message="Feedback saved successfully",
            generation_id=request.generation_id,
            agent_accuracy=stats.get('agent_accuracy', 0.0)
        )

    except HTTPException:
        raise
    except psycopg2.OperationalError as e:
        logger.error(f"Database connection failed: {e}")
        raise HTTPException(
            status_code=503, 
            detail="Database temporarily unavailable"
        )
    except Exception as e:
        logger.error(f"Feedback submission failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Feedback failed: {str(e)[:200]}"
        )


@app.get("/api/learning/{business_id}/stats")
async def get_learning_stats(business_id: str, content_type: str = "blog"):
    """Get learning statistics for a business"""
    try:
        memory = BrandLearningMemory(business_id=business_id)
        stats = memory.get_learning_stats(content_type)
        
        return {
            "business_id": business_id,
            "content_type": content_type,
            "stats": stats
        }

    except Exception as e:
        logger.error(f"Stats retrieval failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Stats retrieval failed: {str(e)[:200]}"
        )


@app.get("/api/learning/verify/{business_id}")
async def verify_learning(business_id: str, content_type: str = "blog"):
    """Verify learning system is working"""
    try:
        stats = verify_learning_loop(business_id, content_type)

        diagnostics = []
        if stats.get('total_generations', 0) == 0:
            diagnostics.append({
                "level": "error",
                "message": "No learnings saved"
            })
        elif stats.get('agent_accuracy', 0) < 50 and stats.get('human_feedback_count', 0) > 5:
            diagnostics.append({
                "level": "warning",
                "message": "Agent accuracy low"
            })
        else:
            diagnostics.append({
                "level": "success",
                "message": "Learning system operational"
            })

        return {
            "business_id": business_id,
            "content_type": content_type,
            "stats": stats,
            "diagnostics": diagnostics
        }

    except Exception as e:
        logger.error(f"Verification failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Verification failed: {str(e)[:200]}"
        )


# ===== ERROR HANDLERS =====

@app.exception_handler(429)
async def rate_limit_handler(request: Request, exc: HTTPException):
    """Custom rate limit error response"""
    return JSONResponse(
        status_code=429,
        content={
            "error": "rate_limit_exceeded",
            "message": "Too many requests. Please try again in 60 seconds.",
            "retry_after": 60
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: Exception):
    """Custom 500 error response"""
    logger.error(f"Internal error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "An internal error occurred. Please try again later.",
            "timestamp": datetime.utcnow().isoformat()
        }
    )


# ===== STARTUP INFO =====

if __name__ == "__main__":
    import uvicorn

    print("\n" + "="*60)
    print("🎯 BrandGuard AI - Production-Ready API")
    print("="*60)
    print("✅ Features:")
    print("  - Rate limiting (10 req/min per IP)")
    print("  - Circuit breakers (Groq, Tavily, DB)")
    print("  - Database: NullPool (Neon-optimized)")
    print("  - Automatic retry with exponential backoff")
    print("  - Comprehensive error handling")
    print("  - Request/response logging")
    print("  - Health checks")
    print("="*60)
    print("🌐 Endpoints:")
    print("  GET  /health              - Health check")
    print("  POST /api/generate        - Generate content")
    print("  POST /api/feedback        - Submit feedback")
    print("  POST /api/upload          - Upload docs")
    print("  GET  /api/learning/stats  - Learning stats")
    print("  GET  /docs                - API documentation")
    print("="*60 + "\n")

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Set to True for development
        log_level="info",
        access_log=True
    )