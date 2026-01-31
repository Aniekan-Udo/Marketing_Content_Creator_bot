import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, MetaData, String
from sqlalchemy.orm import Mapped, mapped_column, sessionmaker, Session
from typing import Optional
from datetime import datetime

from sqlalchemy import (
    String, Integer, Boolean, DateTime, Text, DECIMAL, 
    ForeignKey, CheckConstraint, func
)
from sqlalchemy.orm import (
    DeclarativeBase, relationship
)
from sqlalchemy.dialects.postgresql import JSONB, INET

# Load environment variables FIRST
load_dotenv()

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
    business_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
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
    user_id: Mapped[int] = mapped_column(ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    business_id: Mapped[str] = mapped_column(String(100), nullable=False)
    
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    content_type: Mapped[str] = mapped_column(String(50), nullable=False)
    file_path: Mapped[Optional[str]] = mapped_column(Text)
    file_size_bytes: Mapped[Optional[int]] = mapped_column(Integer)
    
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
    generation_id: Mapped[int] = mapped_column(ForeignKey('reviewer_learning.id', ondelete='CASCADE'), nullable=False)
    user_id: Mapped[int] = mapped_column(ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    
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
    user_id: Mapped[Optional[int]] = mapped_column(ForeignKey('users.id', ondelete='CASCADE'))
    business_id: Mapped[Optional[str]] = mapped_column(String(100))
    
    event_type: Mapped[str] = mapped_column(String(100), nullable=False)
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
    user_id: Mapped[int] = mapped_column(ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    
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
    generation_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)  # UUID
    business_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    
    # ✅ ADD THIS LINE:
    user_id: Mapped[int] = mapped_column(ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    
    # Generation metadata 
    topic: Mapped[str] = mapped_column(String(500))
    content_type: Mapped[str] = mapped_column(String(50))  # blog/social/ad
    format_type: Mapped[str] = mapped_column(String(100))  # Blog Article, etc
    generated_content: Mapped[str] = mapped_column(Text)
    creative_angle: Mapped[str] = mapped_column(String(200))
    
    # Agent scoring
    agent_auto_score: Mapped[float] = mapped_column(DECIMAL(3, 1))
    agent_confidence: Mapped[Optional[float]] = mapped_column(DECIMAL(3, 2))
    agent_auto_approved: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Human feedback
    has_human_feedback: Mapped[bool] = mapped_column(Boolean, default=False)
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

    # ✅ ADD THIS RELATIONSHIP:
    user = relationship("User", back_populates="generations")
    
    # ✅ REMOVE THIS (wrong relationship):
    # generation = relationship("Generation", back_populates="reviewer_learning")
    
    feedbacks = relationship("GenerationFeedback", back_populates="generation", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f'ReviewerLearning(generation_id={self.generation_id}, business={self.business_id})'

# ===== DATABASE ENGINE & SESSION =====
# Create engine ONCE at module import
engine = create_engine(
    os.getenv("POSTGRES_URI"),
    echo=False,  # Set to True for debugging
    pool_size=20,
    max_overflow=10,
    pool_pre_ping=True,
)

# Create session factory ONCE
session_maker = sessionmaker(
    bind=engine,
    class_=Session,
    expire_on_commit=False,
    autoflush=False,
)


def init_db():
    """Initialize database - run once at startup (SYNCHRONOUS)"""
    Model.metadata.create_all(engine)
    print("✅ Database tables created successfully")


def get_session():
    """Dependency for getting database sessions"""
    with session_maker() as session:
        yield session
