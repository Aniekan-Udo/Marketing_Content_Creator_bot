

import os
from datetime import datetime, timedelta
from contextlib import contextmanager
from typing import Optional

import structlog
from dotenv import load_dotenv
from sqlalchemy import create_engine, MetaData, String, Integer, Boolean, DateTime, Text, DECIMAL, ForeignKey, CheckConstraint, func, event
from sqlalchemy.orm import Mapped, mapped_column, sessionmaker, Session, DeclarativeBase, relationship
from sqlalchemy.pool import QueuePool
from sqlalchemy.dialects.postgresql import JSONB, INET
from sqlalchemy.exc import OperationalError, DatabaseError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from passlib.context import CryptContext
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

# Load environment variables
load_dotenv()

logger = structlog.get_logger(__name__)

# ===== BASE MODEL =====
class Model(DeclarativeBase):
    metadata = MetaData(naming_convention={
        "ix": "ix_%(column_0_label)s",
        "uq": "uq_%(table_name)s_%(column_0_name)s",
        "ck": "ck_%(table_name)s_%(constraint_name)s",
        "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
        "pk": "pk_%(table_name)s",
    })


# ===== MODELS =====
class User(Model):
    __tablename__ = 'users'
    
    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    business_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    business_name: Mapped[Optional[str]] = mapped_column(String(255))
    industry: Mapped[Optional[str]] = mapped_column(String(100))
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now(),
        nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )
    
    is_active: Mapped[bool] = mapped_column(Boolean, server_default='true', nullable=False)
    subscription_tier: Mapped[str] = mapped_column(String(50), server_default='free', nullable=False)
    # password_hash removed
    
    monthly_generation_limit: Mapped[int] = mapped_column(Integer, server_default='50', nullable=False)
    monthly_generations_used: Mapped[int] = mapped_column(Integer, server_default='0', nullable=False)
    
    settings: Mapped[dict] = mapped_column(JSONB, server_default='{}', nullable=False)
    
    # Relationships
    documents = relationship("BrandDocument", back_populates="user", cascade="all, delete-orphan")
    generations = relationship("ReviewerLearning", back_populates="user", cascade="all, delete-orphan")
    feedbacks = relationship("GenerationFeedback", back_populates="user", cascade="all, delete-orphan")
    api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f'User(id={self.id}, business_id={self.business_id}, email={self.email})'


class BrandDocument(Model):
    __tablename__ = 'brand_documents'
    
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    business_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    content_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    file_path: Mapped[Optional[str]] = mapped_column(Text)
    file_size_bytes: Mapped[Optional[int]] = mapped_column(Integer)
    file_content: Mapped[Optional[str]] = mapped_column(Text)  # Store actual file content
    
    status: Mapped[str] = mapped_column(String(50), server_default='pending', nullable=False)
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    
    vector_table_name: Mapped[Optional[str]] = mapped_column(String(255))
    document_count: Mapped[int] = mapped_column(Integer, server_default='1', nullable=False)
    chunk_count: Mapped[Optional[int]] = mapped_column(Integer)
    
    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    processed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    
    user = relationship("User", back_populates="documents")
    
    def __repr__(self):
        return f'BrandDocument(id={self.id}, filename={self.filename}, type={self.content_type})'


class GenerationFeedback(Model):
    __tablename__ = 'generation_feedback'
    
    id: Mapped[int] = mapped_column(primary_key=True)
    generation_id: Mapped[int] = mapped_column(ForeignKey('reviewer_learning.id', ondelete='CASCADE'), nullable=False, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    
    feedback_type: Mapped[str] = mapped_column(String(50), nullable=False)
    
    rating: Mapped[Optional[int]] = mapped_column(Integer)
    comment: Mapped[Optional[str]] = mapped_column(Text)
    
    issues: Mapped[Optional[dict]] = mapped_column(JSONB)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    
    generation = relationship("ReviewerLearning", back_populates="feedbacks")
    user = relationship("User", back_populates="feedbacks")
    
    __table_args__ = (
        CheckConstraint('rating >= 1 AND rating <= 5', name='valid_rating'),
    )
    
    def __repr__(self):
        return f'GenerationFeedback(id={self.id}, type={self.feedback_type}, rating={self.rating})'


class UsageAnalytics(Model):
    __tablename__ = 'usage_analytics'
    
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[Optional[int]] = mapped_column(ForeignKey('users.id', ondelete='CASCADE'), index=True)
    business_id: Mapped[Optional[str]] = mapped_column(String(100), index=True)
    
    event_type: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    event_data: Mapped[Optional[dict]] = mapped_column(JSONB)
    
    session_id: Mapped[Optional[str]] = mapped_column(String(255))
    ip_address: Mapped[Optional[str]] = mapped_column(INET)
    user_agent: Mapped[Optional[str]] = mapped_column(Text)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    
    def __repr__(self):
        return f'UsageAnalytics(id={self.id}, event={self.event_type})'


class APIKey(Model):
    __tablename__ = 'api_keys'
    
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    
    key_hash: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    key_prefix: Mapped[Optional[str]] = mapped_column(String(20))
    name: Mapped[Optional[str]] = mapped_column(String(100))
    
    scopes: Mapped[dict] = mapped_column(JSONB, server_default='["read", "write"]', nullable=False)
    
    is_active: Mapped[bool] = mapped_column(Boolean, server_default='true', nullable=False)
    last_used_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    
    rate_limit_per_minute: Mapped[int] = mapped_column(Integer, server_default='60', nullable=False)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    
    user = relationship("User", back_populates="api_keys")
    
    def __repr__(self):
        return f'APIKey(id={self.id}, prefix={self.key_prefix}, active={self.is_active})'


class ReviewerLearning(Model):
    __tablename__ = 'reviewer_learning'
    
    id: Mapped[int] = mapped_column(primary_key=True)
    generation_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    business_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    
    # Generation metadata 
    topic: Mapped[str] = mapped_column(String(500))
    content_type: Mapped[str] = mapped_column(String(50), index=True)
    format_type: Mapped[str] = mapped_column(String(100))
    generated_content: Mapped[str] = mapped_column(Text)
    creative_angle: Mapped[str] = mapped_column(String(200))
    
    # Agent scoring
    agent_auto_score: Mapped[float] = mapped_column(DECIMAL(3, 1))
    agent_confidence: Mapped[Optional[float]] = mapped_column(DECIMAL(3, 2))
    agent_auto_approved: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Human feedback
    has_human_feedback: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    human_approved: Mapped[Optional[bool]] = mapped_column(Boolean)
    human_score: Mapped[Optional[float]] = mapped_column(DECIMAL(3, 1))
    human_feedback: Mapped[Optional[str]] = mapped_column(Text)
    
    # Learning signals
    agent_correct: Mapped[Optional[bool]] = mapped_column(Boolean)
    features_used: Mapped[Optional[dict]] = mapped_column(JSONB)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )

    user = relationship("User", back_populates="generations")
    feedbacks = relationship("GenerationFeedback", back_populates="generation", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f'ReviewerLearning(generation_id={self.generation_id}, business={self.business_id})'


# ===== DATABASE ENGINE CONFIGURATION =====

def get_database_url() -> str:
    """Get database URL with validation"""
    db_url = os.getenv("POSTGRES_URI")
    if not db_url:
        raise ValueError("POSTGRES_URI environment variable not set")
    return db_url


def create_db_engine():
    """
    Create database engine with production settings:
    - Connection pooling with health checks
    - Automatic reconnection
    - Query timeout (set per-session for Neon compatibility)
    """
    db_url = get_database_url()
    
    # Check if using Neon pooler
    is_neon_pooler = 'pooler' in db_url and 'neon.tech' in db_url
    
    # Base connect_args
    connect_args = {
        "connect_timeout": 10,
        "sslmode": "require"  # Required for Neon
    }
    
    # Only add statement_timeout if NOT using Neon pooler
    if not is_neon_pooler:
        connect_args["options"] = "-c statement_timeout=30000"
    
    engine = create_engine(
        db_url,
        # Connection pool settings
        poolclass=QueuePool,
        pool_size=20,              # Maximum connections in pool
        max_overflow=10,           # Additional connections when pool is full
        pool_timeout=30,           # Seconds to wait for connection
        pool_recycle=3600,         # Recycle connections after 1 hour
        pool_pre_ping=True,        # Verify connections before using
        
        # Query settings
        echo=False,                # Set to True for debugging
        echo_pool=False,           # Pool logging
        
        # Performance settings
        connect_args=connect_args
    )
    
    # Add connection pool event listeners for monitoring
    @event.listens_for(engine, "connect")
    def receive_connect(dbapi_conn, connection_record):
        logger.debug("Database connection established")
        
        # Set statement timeout for Neon pooler connections
        if is_neon_pooler:
            cursor = dbapi_conn.cursor()
            try:
                cursor.execute("SET statement_timeout = '30s'")
            finally:
                cursor.close()
    
    @event.listens_for(engine, "checkout")
    def receive_checkout(dbapi_conn, connection_record, connection_proxy):
        logger.debug("Connection checked out from pool")
    
    @event.listens_for(engine, "checkin")
    def receive_checkin(dbapi_conn, connection_record):
        logger.debug("Connection returned to pool")
    
    return engine


# Create engine ONCE at module import
try:
    engine = create_db_engine()
    logger.info("Database engine created successfully")
except Exception as e:
    logger.error(f"Failed to create database engine: {e}")
    raise


# Create session factory
session_maker = sessionmaker(
    bind=engine,
    class_=Session,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False
)


# ===== DATABASE INITIALIZATION =====

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((OperationalError, DatabaseError)),
    reraise=True
)
def init_db():
    """
    Initialize database with retry logic.
    Creates all tables if they don't exist.
    Also ensures the 'file_content' column exists in 'brand_documents'.
    """
    try:
        logger.info("Initializing database...")
        Model.metadata.create_all(engine)
        
        # Ensure 'file_content' column exists (self-healing migration)
        try:
            from sqlalchemy import text
            with engine.begin() as conn:
                # Check if column exists
                check_sql = text("""
                    SELECT 1 
                    FROM information_schema.columns 
                    WHERE table_name='brand_documents' 
                    AND column_name='file_content'
                """)
                result = conn.execute(check_sql).fetchone()
                
                if not result:
                    logger.info("Adding missing 'file_content' column to 'brand_documents'...")
                    conn.execute(text("ALTER TABLE brand_documents ADD COLUMN file_content TEXT"))
                    logger.info("Successfully added 'file_content' column")


        except Exception as migration_error:
            logger.warning(f"Self-healing migration failed (non-critical): {migration_error}")

        logger.info("Database initialization completed")
        return True
    except Exception as e:
        logger.error(f"Database initialization failed: {e}", exc_info=True)
        raise


# ===== SESSION MANAGEMENT =====

@contextmanager
def get_db_session():
    """
    Context manager for database sessions with automatic cleanup.
    
    Usage:
        with get_db_session() as session:
            user = session.query(User).first()
    """
    session = session_maker()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database session error: {e}", exc_info=True)
        raise
    finally:
        session.close()


def get_session():
    """
    Dependency for getting database sessions (FastAPI compatible).
    
    Usage:
        @app.get("/users")
        def get_users(session: Session = Depends(get_session)):
            return session.query(User).all()
    """
    session = session_maker()
    try:
        yield session
    finally:
        session.close()


# ===== HEALTH CHECK =====

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=5),
    retry=retry_if_exception_type((OperationalError, DatabaseError))
)
def check_database_health() -> bool:
    """
    Check if database is accessible and healthy.
    
    Returns:
        bool: True if database is healthy
    """
    try:
        with get_db_session() as session:
            # Simple query to test connection
            session.execute(func.now())
        return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False


# ===== HELPER FUNCTIONS =====

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((OperationalError, DatabaseError))
)
def get_or_create_user(business_id: str, email: str = None, session: Session = None) -> User:
    """
    Get existing user or create new one with retry logic.
    
    Args:
        business_id: Business identifier
        email: User email (optional)
        session: Optional existing session
        
    Returns:
        User: User object
    """
    close_session = False
    
    if session is None:
        session = session_maker()
        close_session = True
    
    try:
        user = session.query(User).filter_by(business_id=business_id).first()
        
        if user:
            return user
        
        # Create new user
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
        
    except Exception as e:
        session.rollback()
        logger.error(f"Failed to get/create user: {e}", exc_info=True)
        raise
    finally:
        if close_session:
            session.close()


# Auth helpers removed (hash_password, verify_password, create_access_token, decode_access_token, get_current_user)


def get_pool_status() -> dict:
    """Get current connection pool status for monitoring"""
    pool = engine.pool
    return {
        "size": pool.size(),
        "checked_in": pool.checkedin(),
        "checked_out": pool.checkedout(),
        "overflow": pool.overflow(),
        "total": pool.size() + pool.overflow()
    }
