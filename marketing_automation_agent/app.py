# api.py - FastAPI routes that call deployer.py business logic

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import os
import uuid
from datetime import datetime
import structlog

# Database imports
from db import session_maker, init_db, User, BrandDocument, ReviewerLearning

# Import business logic from deployer
from deployer import (
    run_generation_with_learning,
    update_generation_with_human_feedback,
    verify_learning_loop,
    BrandLearningMemory,
    BrandMetricsAnalyzer,
    logger
)

# ===== FASTAPI APP =====
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting BrandGuard AI API")
    try:
        init_db()
        logger.info("✅ Database initialized")
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        raise

    yield  # Application runs

    # Shutdown
    logger.info("Shutting down BrandGuard AI API")


# FastAPI app initialization:
app = FastAPI(
    title="BrandGuard AI API",
    description="Marketing content generation with brand learning",
    version="2.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# ===== PYDANTIC MODELS =====

class GenerateRequest(BaseModel):
    business_id: str
    topic: str
    format_type: str = Field(default="Blog Article", alias="format")
    voice: str = "formal"


class FeedbackRequest(BaseModel):
    generation_id: str
    business_id: str
    human_approved: bool
    human_score: float = Field(ge=0, le=10)
    human_feedback: str


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


class FeedbackResponse(BaseModel):
    success: bool
    message: str
    generation_id: str
    agent_accuracy: Optional[float] = None


class LearningStatsResponse(BaseModel):
    total_generations: int
    human_feedback_count: int
    approved_count: int
    approval_rate: float
    avg_auto_score: float
    avg_human_score: float
    agent_accuracy: float


# ===== HELPER FUNCTIONS =====

def get_or_create_user(business_id: str, email: str = None) -> User:
    """Get or create user in database"""
    with session_maker() as session:
        user = session.query(User).filter_by(business_id=business_id).first()

        if user:
            return user

        email = email or f"{business_id}@brandguard.ai"
        user = User(
            email=email,
            business_id=business_id,
            business_name=business_id,
            industry='Marketing'
        )
        session.add(user)
        session.commit()
        session.refresh(user)

        logger.info(f"Created new user: {business_id}")
        return user


# ===== ROUTES =====

# Keep an API-info endpoint on a different path
@app.get("/api-info")
async def api_info():
    """API metadata"""
    return {
        "name": "BrandGuard AI API",
        "version": "2.0.0",
        "status": "operational",
        "endpoints": {
            "generate": "/api/generate",
            "feedback": "/api/feedback",
            "stats": "/api/learning/stats/{business_id}",
            "verify": "/api/learning/verify/{business_id}",
            "upload": "/api/upload"
        }
    }


# Root now only serves index.html
@app.get("/", include_in_schema=False)
async def serve_index():
    """Serve the main frontend"""
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    else:
        return {
            "name": "BrandGuard AI API",
            "version": "2.0.0",
            "status": "operational",
            "message": "index.html not found. Place it in the same directory as app.py",
            "endpoints": {
                "docs": "/docs",
                "generate": "/api/generate",
                "feedback": "/api/feedback",
                "stats": "/api/learning/stats/{business_id}",
            }
        }


@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "environment": {
            "groq_api_key": bool(os.getenv("GROQ_API_KEY")),
            "postgres_uri": bool(os.getenv("POSTGRES_URI")),
            "tavily_api_key": bool(os.getenv("TAVILY_API_KEY"))
        }
    }


@app.post("/api/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
    business_id: str = Form(...),
    content_type: str = Form(...)
):
    """
    Upload brand documents

    - **files**: List of files to upload
    - **business_id**: Business identifier
    - **content_type**: Type (blog/social/ad)
    """
    try:
        if content_type not in ['blog', 'social', 'ad']:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid content_type: {content_type}. Must be blog/social/ad"
            )

        # Get or create user
        user = get_or_create_user(business_id)

        # Create directory
        folder_mapping = {
            'blog': f'brand_blogs/{business_id}',
            'social': f'brand_social/{business_id}',
            'ad': f'brand_ads/{business_id}'
        }

        target_folder = folder_mapping[content_type]
        os.makedirs(target_folder, exist_ok=True)

        uploaded_files = []

        # Save files
        with session_maker() as session:
            for file in files:
                if file.filename:
                    filename = file.filename
                    filepath = os.path.join(target_folder, filename)

                    # Save file
                    content = await file.read()
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

        logger.info(f"✅ Uploaded {len(uploaded_files)} files for {business_id}")

        return {
            "success": True,
            "message": f"Uploaded {len(uploaded_files)} files",
            "files": uploaded_files,
            "business_id": business_id,
            "content_type": content_type
        }

    except Exception as e:
        logger.error(f"Upload failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate", response_model=GenerateResponse)
async def generate_content(request: GenerateRequest):
    """Generate marketing content"""
    try:
        logger.info(f"🚀 Generation request: {request.business_id} - {request.topic}")
        
        
        result_dict, generation_id = run_generation_with_learning(
            business_id=request.business_id,
            topic=request.topic,
            format_type=request.format_type,
            voice=request.voice
        )
        
        logger.info(f"✅ Generation complete: {generation_id}")
        
        return GenerateResponse(
            status="success",
            generation_id=generation_id,
            content=result_dict["content"],
            auto_score=result_dict["auto_score"],
            creative_angle=result_dict["creative_angle"],
            topic=request.topic,
            format_type=request.format_type,
            business_id=request.business_id,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
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
                detail=f"Generation {request.generation_id} not found for business {request.business_id}"
            )

        memory = BrandLearningMemory(business_id=request.business_id)
        stats = memory.get_learning_stats("blog")

        return FeedbackResponse(
            success=True,
            message="Feedback saved successfully",
            generation_id=request.generation_id,
            agent_accuracy=stats.get('agent_accuracy', 0.0)  # ✅ Default value
        )

    except HTTPException:
        raise
    except psycopg2.OperationalError as e:
        logger.error(f"Database connection failed: {e}")
        raise HTTPException(status_code=503, detail="Database temporarily unavailable")
    except Exception as e:
        logger.error(f"Feedback submission failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/learning/stats/{business_id}", response_model=LearningStatsResponse)
async def get_learning_stats(business_id: str, content_type: str = "blog"):
    """
    Get learning statistics for a business
    """
    try:
        memory = BrandLearningMemory(business_id=business_id)
        stats = memory.get_learning_stats(content_type)

        return LearningStatsResponse(**stats)

    except Exception as e:
        logger.error(f"Stats retrieval failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/learning/verify/{business_id}")
async def verify_learning(business_id: str, content_type: str = "blog"):
    """
    Verify learning system is working
    """
    try:
        stats = verify_learning_loop(business_id, content_type)

        diagnostics = []

        if stats.get('total_generations', 0) == 0:
            diagnostics.append({
                "level": "error",
                "message": "No learnings saved! Auto-save not working."
            })
        elif stats.get('agent_accuracy', 0) < 50 and stats.get('human_feedback_count', 0) > 5:
            diagnostics.append({
                "level": "warning",
                "message": "Agent accuracy low - review scoring criteria"
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
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/learning/summary/{business_id}")
async def get_learning_summary(business_id: str, content_type: str = "blog"):
    """
    Get human-readable learning summary
    """
    try:
        memory = BrandLearningMemory(business_id=business_id)
        summary = memory.get_learning_summary(content_type)

        approved = memory.get_approved_patterns(content_type, limit=5)
        rejected = memory.get_rejected_patterns(content_type, limit=3)

        return {
            "business_id": business_id,
            "content_type": content_type,
            "summary_text": summary,
            "approved_patterns": [
                {
                    "creative_angle": p.get('creative_angle'),
                    "score": p.get('human_score') or p.get('auto_score'),
                    "feedback": p.get('human_feedback'),
                    "approval_type": "human" if p.get('human_approved') else "auto"
                }
                for p in approved
            ],
            "rejected_patterns": [
                {
                    "creative_angle": p.get('creative_angle'),
                    "score": p.get('auto_score'),
                    "feedback": p.get('human_feedback')
                }
                for p in rejected
            ]
        }

    except Exception as e:
        logger.error(f"Summary retrieval failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/feedback/pending/{business_id}")
async def get_pending_feedback(business_id: str, limit: int = 10):
    """
    Get generations that need human feedback
    """
    try:
        with session_maker() as session:
            pending = session.query(ReviewerLearning).filter(
                ReviewerLearning.business_id == business_id,
                ReviewerLearning.has_human_feedback == False
            ).order_by(
                ReviewerLearning.created_at.desc()
            ).limit(limit).all()

            return {
                "business_id": business_id,
                "count": len(pending),
                "pending": [
                    {
                        "generation_id": p.generation_id,
                        "topic": p.topic,
                        "content_type": p.content_type,
                        "generated_content": p.generated_content,
                        "creative_angle": p.creative_angle,
                        "auto_score": float(p.agent_auto_score),
                        "created_at": p.created_at.isoformat()
                    }
                    for p in pending
                ]
            }

    except Exception as e:
        logger.error(f"Pending feedback retrieval failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/kb/stats")
async def kb_stats(business_id: str):
    """Get knowledge base statistics"""
    try:
        user = get_or_create_user(business_id)

        with session_maker() as session:
            docs = session.query(BrandDocument).filter_by(
                business_id=business_id
            ).all()

            stats = {
                "blog": len([d for d in docs if d.content_type == 'blog']),
                "social": len([d for d in docs if d.content_type == 'social']),
                "ad": len([d for d in docs if d.content_type == 'ad']),
            }

            stats["total"] = sum(stats.values())
            stats["ready"] = stats["total"] > 0
            stats["business_id"] = business_id

            return stats

    except Exception as e:
        logger.error(f"KB stats failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/feedback")
async def serve_feedback():
    """Serve the feedback page"""
    if os.path.exists("templates/feedback.html"):
        return FileResponse("templates/feedback.html")
    elif os.path.exists("feedback.html"):
        return FileResponse("feedback.html")
    else:
        raise HTTPException(status_code=404, detail="Feedback page not found")


# ===== RUN =====

if __name__ == "__main__":
    import uvicorn

    print("\n" + "="*60)
    print("🎯 BrandGuard AI - FastAPI Backend")
    print("="*60)
    print("✅ API running on: http://localhost:8000")
    print("📖 Docs: http://localhost:8000/docs")
    print("📋 OpenAPI: http://localhost:8000/redoc")
    print("="*60)
    print("\n📍 Endpoints:")
    print("  POST /api/generate        - Generate content")
    print("  POST /api/feedback        - Submit feedback")
    print("  GET  /api/learning/stats/{business_id}")
    print("  GET  /api/learning/verify/{business_id}")
    print("  GET  /api/learning/summary/{business_id}")
    print("  GET  /api/feedback/pending/{business_id}")
    print("  POST /api/upload          - Upload brand docs")
    print("  GET  /api/kb/stats        - KB statistics")
    print("="*60 + "\n")

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
