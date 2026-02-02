# deployer.py - HYBRID REFLECTION LEARNING SOLUTION


import os
import argparse
import threading
from pathlib import Path
from typing import Dict, Optional, Any, List
import json
import re
from datetime import datetime
from pydantic import Field, BaseModel

import structlog
from structlog import get_logger
from structlog.stdlib import LoggerFactory
from structlog.dev import ConsoleRenderer
from structlog.processors import TimeStamper, StackInfoRenderer, format_exc_info, add_log_level, JSONRenderer

from threading import Lock
from dotenv import load_dotenv

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool

from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SemanticSplitterNodeParser, SimpleNodeParser
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.fastembed import FastEmbedEmbedding

from tavily import TavilyClient
from tenacity import retry, stop_after_attempt, wait_exponential

# ===== LOGGING SETUP =====
structlog.configure(
    processors=[
        add_log_level,
        TimeStamper(fmt="iso"),
        StackInfoRenderer(),
        format_exc_info,
        JSONRenderer() if "JSON" in os.getenv("LOG_FORMAT", "console") else ConsoleRenderer()
    ],
    logger_factory=LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = get_logger("Marketing content creation bot")

# ===== ENVIRONMENT VARIABLES =====
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
POSTGRES_URI = os.getenv("POSTGRES_URI")
POSTGRES_ASYNC_URI = os.getenv("POSTGRES_ASYNC_URI")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if POSTGRES_URI:
    logger.info(f"Sync URI loaded")
else:
    logger.error("POSTGRES_URI is None or empty")
    raise ValueError("POSTGRES_URI not found in environment")

if POSTGRES_ASYNC_URI:
    logger.info(f"Async URI loaded successfully")
else:
    logger.error("POSTGRES_ASYNC_URI is None or empty")
    raise ValueError("POSTGRES_ASYNC_URI not found in environment")

if not GROQ_API_KEY:
    logger.error("GROQ_API_KEY not found")
    raise ValueError("GROQ_API_KEY not found")

if not TAVILY_API_KEY:
    logger.warning("TAVILY_API_KEY not found - web search disabled")

# ===== LOCKS FOR THREAD SAFETY =====
_init_locks: Dict[str, Lock] = {}
_locks_lock = Lock()

def get_init_lock(business_id):
    with _locks_lock:
        if business_id not in _init_locks:
            _init_locks[business_id] = Lock()
        return _init_locks[business_id]

# ===== LLM CONFIGURATION =====
def get_llm(temperature=0.7):
    """Get configured LLM instance. Higher temp (0.7-0.8) for creative diversity."""
    return LLM(
        model="groq/llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY,
        temperature=temperature
    )

# ===== BRAND METRICS ANALYZER (NEW!) =====
class BrandMetricsAnalyzer:
    """
    Analyzes brand documents to extract concrete, measurable metrics.
    Provides ground truth for automatic scoring.
    """
    
    def __init__(self, business_id: str, content_type: str):
        self.business_id = business_id
        self.content_type = content_type
        self.metrics = None
        self._load_metrics()
    
    def _load_brand_docs(self) -> List[str]:
        """Load raw brand documents"""
        data_path = os.path.abspath(f"brand_{self.content_type}s/{self.business_id}")
        
        if not os.path.exists(data_path):
            logger.warning(f"Brand docs path not found: {data_path}")
            return []
        
        docs = []
        for filename in os.listdir(data_path):
            filepath = os.path.join(data_path, filename)
            if os.path.isfile(filepath):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        docs.append(f.read())
                except Exception as e:
                    logger.warning(f"Could not read {filename}: {e}")
        
        return docs
    
    def _analyze_text(self, text: str) -> Dict[str, Any]:
        """Extract metrics from a single text"""
        # Sentence splitting (simple regex)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        
        # Word counts
        sentence_lengths = [len(s.split()) for s in sentences]
        avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0
        
        # Paragraph detection (double newlines or clear breaks)
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        # Sentences per paragraph
        sentences_per_para = []
        for para in paragraphs:
            para_sentences = re.split(r'[.!?]+', para)
            para_sentences = [s.strip() for s in para_sentences if s.strip() and len(s.strip()) > 10]
            if para_sentences:
                sentences_per_para.append(len(para_sentences))
        
        avg_sentences_per_para = sum(sentences_per_para) / len(sentences_per_para) if sentences_per_para else 0
        
        # Detect bullet points or numbered lists
        has_bullets = bool(re.search(r'^\s*[-*•]', text, re.MULTILINE))
        has_numbered_lists = bool(re.search(r'^\s*\d+\.', text, re.MULTILINE))
        
        # Extract common phrases (2-5 word sequences)
        words = text.lower().split()
        phrases = []
        for i in range(len(words) - 1):
            for length in [2, 3, 4, 5]:
                if i + length <= len(words):
                    phrase = ' '.join(words[i:i+length])
                    # Clean punctuation
                    phrase = re.sub(r'[^\w\s]', '', phrase).strip()
                    if phrase and len(phrase) > 5:
                        phrases.append(phrase)
        
        # Count phrase frequency
        from collections import Counter
        phrase_counts = Counter(phrases)
        common_phrases = [phrase for phrase, count in phrase_counts.most_common(30) if count >= 2]
        
        return {
            'sentence_count': len(sentences),
            'avg_sentence_length': avg_sentence_length,
            'paragraph_count': len(paragraphs),
            'avg_sentences_per_para': avg_sentences_per_para,
            'has_bullets': has_bullets,
            'has_numbered_lists': has_numbered_lists,
            'common_phrases': common_phrases[:20],
            'total_words': len(text.split())
        }
    
    def _load_metrics(self):
        """Load and cache brand metrics"""
        docs = self._load_brand_docs()
        
        if not docs:
            logger.warning(f"No brand docs found for {self.business_id}/{self.content_type}")
            self.metrics = self._get_fallback_metrics()
            return
        
        # Analyze all docs
        all_metrics = [self._analyze_text(doc) for doc in docs]
        
        # Aggregate metrics
        self.metrics = {
            'target_sentence_length': sum(m['avg_sentence_length'] for m in all_metrics) / len(all_metrics),
            'target_sentences_per_para': sum(m['avg_sentences_per_para'] for m in all_metrics) / len(all_metrics),
            'uses_bullets': any(m['has_bullets'] for m in all_metrics),
            'uses_numbered_lists': any(m['has_numbered_lists'] for m in all_metrics),
            'signature_phrases': self._consolidate_phrases([m['common_phrases'] for m in all_metrics]),
            'sample_count': len(docs)
        }
        
        logger.info(f"Metrics loaded: {self.business_id}/{self.content_type}")
        logger.info(f"   Target sentence length: {self.metrics['target_sentence_length']:.1f} words")
        logger.info(f"   Target sentences/para: {self.metrics['target_sentences_per_para']:.1f}")
        logger.info(f"   Signature phrases: {len(self.metrics['signature_phrases'])}")
    
    def _consolidate_phrases(self, phrase_lists: List[List[str]]) -> List[str]:
        """Find phrases common across multiple documents"""
        from collections import Counter
        all_phrases = [phrase for sublist in phrase_lists for phrase in sublist]
        phrase_counts = Counter(all_phrases)
        
        # Return phrases that appear in multiple docs
        return [phrase for phrase, count in phrase_counts.most_common(25) if count >= 2]
    
    def _get_fallback_metrics(self) -> Dict[str, Any]:
        """Fallback metrics if no docs available"""
        return {
            'target_sentence_length': 15.0,
            'target_sentences_per_para': 3.0,
            'uses_bullets': False,
            'uses_numbered_lists': False,
            'signature_phrases': [],
            'sample_count': 0
        }
    
    def score_content(self, content: str) -> Dict[str, Any]:
        """Score content against brand metrics"""
        content_metrics = self._analyze_text(content)
        
        # Sentence length score
        target_len = self.metrics['target_sentence_length']
        actual_len = content_metrics['avg_sentence_length']
        len_deviation = abs(actual_len - target_len) / target_len if target_len > 0 else 1.0
        sentence_length_score = max(0, 10 - (len_deviation * 20))
        
        # Paragraph structure score
        target_para = self.metrics['target_sentences_per_para']
        actual_para = content_metrics['avg_sentences_per_para']
        para_deviation = abs(actual_para - target_para) / target_para if target_para > 0 else 1.0
        para_structure_score = max(0, 10 - (para_deviation * 15))
        
        # Format alignment score (bullets/lists)
        format_aligned = True
        if self.metrics['uses_bullets'] != content_metrics['has_bullets']:
            format_aligned = False
        if self.metrics['uses_numbered_lists'] != content_metrics['has_numbered_lists']:
            format_aligned = False
        format_score = 10 if format_aligned else 5
        
        # Signature phrase usage
        content_lower = content.lower()
        phrases_found = [p for p in self.metrics['signature_phrases'] if p in content_lower]
        phrase_usage_ratio = len(phrases_found) / max(len(self.metrics['signature_phrases']), 1)
        phrase_score = min(10, phrase_usage_ratio * 20)  # Use 50% of phrases = score 10
        
        # Overall structure score
        structure_score = (sentence_length_score + para_structure_score + format_score) / 3
        
        return {
            'structure_score': round(structure_score, 1),
            'phrase_score': round(phrase_score, 1),
            'sentence_length_score': round(sentence_length_score, 1),
            'para_structure_score': round(para_structure_score, 1),
            'format_score': format_score,
            'details': {
                'target_sentence_length': round(target_len, 1),
                'actual_sentence_length': round(actual_len, 1),
                'target_sentences_per_para': round(target_para, 1),
                'actual_sentences_per_para': round(actual_para, 1),
                'signature_phrases_available': len(self.metrics['signature_phrases']),
                'signature_phrases_used': len(phrases_found),
                'phrases_found': phrases_found[:10]
            }
        }

# ===== BRAND LEARNING MEMORY (ENHANCED) =====
class BrandLearningMemory:
    """
    Persistent memory system that learns from human feedback AND automatic metrics.
    Stores approved content patterns, rejected patterns, and successful variations.
    """
    
    def __init__(self, business_id: str):
        self.business_id = business_id
        clean_id = business_id.replace("-", "_").replace(".", "_")
        self.table_name = f"brand_learning_{clean_id}"
        self._init_table()
    
    def _init_table(self):
        """Create learning tables if not exists"""
        import psycopg2
        from urllib.parse import urlparse
        
        parsed = urlparse(POSTGRES_URI)
        
        try:
            conn = psycopg2.connect(
                host=parsed.hostname,
                port=parsed.port or 5432,
                database=parsed.path.lstrip('/'),
                user=parsed.username,
                password=parsed.password
            )
            cursor = conn.cursor()
            
            # Main learning table
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id SERIAL PRIMARY KEY,
                    content_type VARCHAR(50),
                    learning_type VARCHAR(50),
                    pattern_data JSONB,
                    auto_score FLOAT,
                    human_approved BOOLEAN,
                    human_score FLOAT,
                    human_feedback TEXT,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
            
            # Index for fast retrieval
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_type 
                ON {self.table_name}(content_type, learning_type, human_approved)
            """)
            
            conn.commit()
            cursor.close()
            conn.close()
            logger.info(f"Learning memory initialized: {self.table_name}")
            
        except Exception as e:
            logger.error(f"Failed to init learning memory: {e}")
    
    def save_learning(self, content_type: str, learning_type: str, 
                      pattern_data: dict, auto_score: float,
                      human_approved: Optional[bool] = None,
                      human_score: Optional[float] = None,
                      human_feedback: str = ""):
        """Save a learning pattern (with both auto and optional human scores)"""
        import psycopg2
        import psycopg2.extras
        from urllib.parse import urlparse
        
        parsed = urlparse(POSTGRES_URI)
        
        try:
            conn = psycopg2.connect(
                host=parsed.hostname,
                port=parsed.port or 5432,
                database=parsed.path.lstrip('/'),
                user=parsed.username,
                password=parsed.password
            )
            cursor = conn.cursor()
            
            cursor.execute(f"""
                INSERT INTO {self.table_name} 
                (content_type, learning_type, pattern_data, auto_score,
                 human_approved, human_score, human_feedback)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (content_type, learning_type, json.dumps(pattern_data), 
                  auto_score, human_approved, human_score, human_feedback))
            
            conn.commit()
            cursor.close()
            conn.close()
            logger.info(f"💾 Saved learning: {learning_type} for {content_type}")
            
        except Exception as e:
            logger.error(f"Failed to save learning: {e}")
    
    def get_approved_patterns(self, content_type: str, learning_type: str, limit: int = 10):
        """Retrieve approved patterns (prioritize human-approved, fall back to high auto-scores)"""
        import psycopg2
        import psycopg2.extras
        from urllib.parse import urlparse
        
        parsed = urlparse(POSTGRES_URI)
        
        try:
            conn = psycopg2.connect(
                host=parsed.hostname,
                port=parsed.port or 5432,
                database=parsed.path.lstrip('/'),
                user=parsed.username,
                password=parsed.password
            )
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # Prioritize human-approved, then high auto-scores
            cursor.execute(f"""
                SELECT pattern_data, auto_score, human_approved, human_score, 
                       human_feedback, created_at
                FROM {self.table_name}
                WHERE content_type = %s 
                  AND learning_type = %s 
                  AND (human_approved = TRUE OR (human_approved IS NULL AND auto_score >= 8.0))
                ORDER BY 
                    CASE WHEN human_approved = TRUE THEN 1 ELSE 2 END,
                    COALESCE(human_score, auto_score) DESC,
                    created_at DESC
                LIMIT %s
            """, (content_type, learning_type, limit))
            
            results = cursor.fetchall()
            cursor.close()
            conn.close()
            
            return [dict(row) for row in results]
            
        except Exception as e:
            logger.error(f"Failed to retrieve patterns: {e}")
            return []
    
    def get_rejected_patterns(self, content_type: str, limit: int = 5):
        """Get rejected patterns (human-rejected or low auto-scores)"""
        import psycopg2
        import psycopg2.extras
        from urllib.parse import urlparse
        
        parsed = urlparse(POSTGRES_URI)
        
        try:
            conn = psycopg2.connect(
                host=parsed.hostname,
                port=parsed.port or 5432,
                database=parsed.path.lstrip('/'),
                user=parsed.username,
                password=parsed.password
            )
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            cursor.execute(f"""
                SELECT pattern_data, auto_score, human_feedback, created_at
                FROM {self.table_name}
                WHERE content_type = %s 
                  AND (human_approved = FALSE OR (human_approved IS NULL AND auto_score < 7.0))
                ORDER BY created_at DESC
                LIMIT %s
            """, (content_type, limit))
            
            results = cursor.fetchall()
            cursor.close()
            conn.close()
            
            return [dict(row) for row in results]
            
        except Exception as e:
            logger.error(f"Failed to retrieve rejected patterns: {e}")
            return []
    
    def get_learning_summary(self, content_type: str) -> str:
        """Generate a summary of learned patterns for agent context"""
        approved = self.get_approved_patterns(content_type, "creative_variation", limit=5)
        rejected = self.get_rejected_patterns(content_type, limit=3)
        
        summary = f"\n=== LEARNED PATTERNS FOR {content_type.upper()} ===\n\n"
        
        if approved:
            summary += "SUCCESSFUL APPROACHES (Human-Approved or High Auto-Score):\n"
            for i, pattern in enumerate(approved, 1):
                data = pattern.get('pattern_data', {})
                if isinstance(data, str):
                    data = json.loads(data)
                
                score_display = f"{pattern.get('human_score', pattern.get('auto_score', 0)):.1f}/10"
                approval_type = "👤 Human" if pattern.get('human_approved') else "Auto"
                
                summary += f"{i}. {approval_type} Score: {score_display}\n"
                summary += f"   Creative angle: {data.get('creative_angle', 'N/A')}\n"
                summary += f"   What worked: {pattern.get('human_feedback') or data.get('what_worked', 'Good alignment')}\n\n"
        
        if rejected:
            summary += "\nAVOID THESE APPROACHES (Rejected):\n"
            for i, pattern in enumerate(rejected, 1):
                data = pattern.get('pattern_data', {})
                if isinstance(data, str):
                    data = json.loads(data)
                summary += f"{i}. Score: {pattern.get('auto_score', 0):.1f}/10\n"
                summary += f"   What failed: {data.get('issue', 'N/A')}\n"
                summary += f"   Feedback: {pattern.get('human_feedback', 'Low alignment')}\n\n"
        
        if not approved and not rejected:
            summary += "No learning data yet. Will learn from first attempt.\n"
        
        return summary




# ===== RAG SYSTEM =====
class Marketing_Rag_System:
    def __init__(self, data_path: str = "./data/marketing", business_id=None, content_type: str = "default"):
        self._embed_model_cache = None
        self._setup_lock = threading.Lock()
        
        if not os.path.isabs(data_path):
            data_path = os.path.abspath(data_path)
        
        self.data_path = data_path
        self.content_type = content_type
        clean_id = (business_id or 'default').replace("-", "_")
        self.table_name = f"rag_data_{clean_id}_{content_type}"
        self.business_id = business_id
        get_init_lock(business_id)

    def initialize_embedding_model(self):
        if self._embed_model_cache is not None:
            return self._embed_model_cache
        
        with self._setup_lock:
            if self._embed_model_cache is not None:
                return self._embed_model_cache
            
            logger.info("Setting up embedding model...")
            self.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
            self._embed_model_cache = self.embed_model
            logger.info("Embedding model ready!")
            return self.embed_model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def build_marketing_index(self):
        from llama_index.llms.groq import Groq
        from llama_index.core import Settings
        rag_llm = Groq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)
                
        Settings.embed_model = self.initialize_embedding_model()
        Settings.llm = rag_llm 
        
        if not self.business_id:
            logger.warning("Business ID not given")

        lock = get_init_lock(self.business_id)
        with lock:
            if self.content_type == "blog":
                parser = SemanticSplitterNodeParser.from_defaults(buffer_size=1, embed_model=Settings.embed_model,breakpoint_percentile_threshold=95)
            elif self.content_type in ["social", "ad", "email"]:
                parser = SimpleNodeParser.from_defaults(chunk_size=256, chunk_overlap=50)
            else:
                parser = SimpleNodeParser.from_defaults(chunk_size=512, chunk_overlap=100)
            
            try:
                from urllib.parse import urlparse
                raw_uri = os.getenv("POSTGRES_ASYNC_URI", "").strip()
                raw_uri = raw_uri.replace("&channel_binding=require", "")
                raw_uri = raw_uri.replace("channel_binding=require&", "")
                raw_uri = raw_uri.replace("?channel_binding=require", "")
                parsed = urlparse(raw_uri)
                
                logger.info("="*80)
                logger.info(f"Building index for: {self.business_id} / {self.content_type}")
                logger.info(f"Data path: {self.data_path}")
                logger.info(f"Table name: {self.table_name}")
                
                if not os.path.exists(self.data_path):
                    logger.error(f"Data path does not exist: {self.data_path}")
                    raise ValueError(f"Data path not found: {self.data_path}")
                
                files = os.listdir(self.data_path)
                logger.info(f"Files found: {len(files)}")
                for f in files[:5]:
                    logger.info(f"   - {f}")
                
                if not files:
                    logger.warning(f"No files in {self.data_path}")
                    return None
                
                logger.info("="*80)
                
                self.vector_store = PGVectorStore.from_params(
                    host=parsed.hostname,
                    port=parsed.port or 5432,
                    database=parsed.path.lstrip('/'),
                    user=parsed.username,
                    password=parsed.password,
                    table_name=self.table_name,
                    embed_dim=384,
                    hybrid_search=True,
                    hnsw_kwargs={
                        "hnsw_m": 16,
                        "hnsw_ef_construction": 64,
                        "hnsw_ef_search": 40,
                        "hnsw_dist_method": "vector_cosine_ops"
                    }
                )

                Settings.node_parser = parser
                storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
                
                try:
                    index = VectorStoreIndex.from_vector_store(
                        vector_store=self.vector_store,
                        embed_model=Settings.embed_model
                    )
                    logger.info(f"Loaded existing index: {self.table_name}")
                    
                    try:
                        retriever = index.as_retriever(similarity_top_k=3)
                        test_result = retriever.retrieve("test query")
                        logger.info(f"Index verification: Found {len(test_result)} nodes")
                        
                        if len(test_result) == 0:
                            logger.warning("Index exists but is empty! Rebuilding...")
                            raise Exception("Empty index")
                            
                    except Exception as verify_error:
                        logger.warning(f"Index verification failed: {verify_error}")
                        raise
                        
                except Exception as load_error:
                    logger.info(f"Building NEW index for {self.table_name}")
                    
                    documents = SimpleDirectoryReader(self.data_path).load_data()
                    logger.info(f"Loaded {len(documents)} documents")
                    
                    if not documents:
                        logger.error(f"No documents loaded from {self.data_path}")
                        return None
                    
                    logger.info("🏗️  Building vector index...")
                    index = VectorStoreIndex.from_documents(
                        documents,
                        storage_context=storage_context,
                        show_progress=True
                    )
                    logger.info(f"Index built successfully: {self.table_name}")
                
                logger.info("="*80)
                return index
                
            except Exception as e:
                logger.error(f"Index build failed: {e}")
                raise

# ===== KNOWLEDGE BASE =====
VALID_CONTENT_TYPES = {"blog", "social", "ad"}

class BrandVoiceKnowledgeBase:
    def __init__(self, postgres_uri: Optional[str] = None, llm=None):
        self.postgres_uri = os.getenv("POSTGRES_ASYNC_URI") 
        self.llm = llm
        self._kb_cache = {}
        self._load_errors = {}

    def _get_kb(self, content_type=str, business_id=str):
        cache_key = (content_type, business_id)

        if cache_key in self._kb_cache:
            logger.info(f"♻️  Using cached KB for {business_id}/{content_type}")
            return self._kb_cache[cache_key]
        
        lock = get_init_lock(f"{business_id}_{content_type}")

        with lock:
            if cache_key in self._kb_cache:
                return self._kb_cache[cache_key]
            
            try:
                data_path = os.path.abspath(f"brand_{content_type}s/{business_id}")
                
                logger.info(f"🔨 Creating new KB for {business_id}/{content_type}")
                
                rag_system = Marketing_Rag_System(
                    data_path=data_path,
                    content_type=content_type,
                    business_id=business_id
                )
                kb = rag_system.build_marketing_index()
                self._kb_cache[cache_key] = kb
                return kb
            except Exception as e:
                logger.error(f"Failed to load {content_type} KB: {e}", exc_info=True)
                self._load_errors[cache_key] = str(e)
                raise
    
    def _get_llm(self):
        if self.llm is None:
            self.llm = get_llm()
        return self.llm
    
    def get_style_guide(self, content_type: str, business_id):
        if content_type not in VALID_CONTENT_TYPES:
            raise ValueError(f"Invalid content_type. Must be one of: {VALID_CONTENT_TYPES}")
        
        self.business_id = business_id
        
        try:
            kb = self._get_kb(content_type, business_id)
            
            if kb is None:
                raise ValueError(f"KB for {content_type} returned None")
            
            top_k_map = {"blog": 5, "social": 3, "ad": 3}
            
            return kb.as_query_engine(
                similarity_top_k=top_k_map[content_type],
                response_mode="compact"
            )
        except Exception as e:
            logger.error(f"get_style_guide failed: {e}", exc_info=True)
            raise

# ========== PART 2/3: TOOLS ==========
# Append this to Part 1

# ===== TOOLS =====
class KBQueryTool(BaseTool):
    name: str = "kb_query"
    description: str = """Query brand knowledge base for style examples.
    
    IMPORTANT: content_type must be EXACTLY one of: 'blog', 'social', or 'ad'
    - Use 'blog' for blog articles, long-form content
    - Use 'social' for social media posts, tweets, LinkedIn posts
    - Use 'ad' for advertisements, ad copy
    
    Example: {"content_type": "blog", "query": "tone examples"}
    """
    kb: Any = Field(default=None, exclude=True)
    business_id: str = Field(default="default", exclude=True)
    
    def __init__(self, kb: BrandVoiceKnowledgeBase, business_id: str):
        super().__init__()
        object.__setattr__(self, 'kb', kb)
        object.__setattr__(self, 'business_id', business_id)
    
    def _run(self, content_type: str, query: str) -> str:
        logger.info("="*80)
        logger.info(f"🔍 KB QUERY - {content_type}: {query}")
        logger.info("="*80)
        
        if not content_type or not query:
            return "Error: Missing content_type or query"
        
        if content_type not in VALID_CONTENT_TYPES:
            return f"Error: content_type must be one of {VALID_CONTENT_TYPES}"
        
        try:
            style_guide = self.kb.get_style_guide(content_type, self.business_id)
            response = style_guide.query(query)
            response_str = str(response)
            
            logger.info(f"KB Response: {len(response_str)} chars")
            
            if not response_str or response_str.strip() == "":
                logger.warning("Empty KB response")
                return self._get_fallback_response(content_type)
            
            return response_str
            
        except Exception as e:
            logger.error(f"KB query failed: {e}", exc_info=True)
            return self._get_fallback_response(content_type)
    
    def _get_fallback_response(self, content_type: str) -> str:
        fallbacks = {
            "blog": "BRAND FALLBACK: Conversational yet professional tone, clear structure, practical insights",
            "social": "BRAND FALLBACK: Engaging, brief, personality-driven",
            "ad": "BRAND FALLBACK: Clear value proposition, compelling call-to-action"
        }
        return fallbacks.get(content_type, fallbacks["blog"])


class TavilySearchTool(BaseTool):
    name: str = "tavily_search"
    description: str = "Search the web using Tavily API for current information and trends"
    
    def _run(self, query: str) -> str:
        if not TAVILY_API_KEY:
            return "Web search unavailable - TAVILY_API_KEY not configured"
        
        try:
            client = TavilyClient(api_key=TAVILY_API_KEY)
            result = client.search(query, max_results=5)
            return str(result)
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return f"Search failed: {str(e)}"


class LearningMemoryTool(BaseTool):
    """Tool to access and update brand learning memory"""
    name: str = "learning_memory"
    description: str = """Access learned brand patterns from past human feedback and auto-scores.
    
    Actions:
    - get_approved: Get successful creative approaches (human or auto-approved)
    - get_rejected: Get patterns that were rejected (to avoid them)
    - get_summary: Get full learning summary for context
    """
    
    memory: Any = Field(default=None, exclude=True)
    
    def __init__(self, memory: BrandLearningMemory):
        super().__init__()
        object.__setattr__(self, 'memory', memory)
    
    def _run(self, action: str, content_type: str = "blog") -> str:
        try:
            if action == "get_approved":
                patterns = self.memory.get_approved_patterns(content_type, "creative_variation")
                if not patterns:
                    return "No approved patterns yet. This is your first attempt!"
                
                result = "SUCCESSFUL CREATIVE APPROACHES:\n"
                for p in patterns:
                    data = p.get('pattern_data', {})
                    if isinstance(data, str):
                        data = json.loads(data)
                    
                    approval_type = "👤 Human" if p.get('human_approved') else "Auto"
                    score = p.get('human_score') or p.get('auto_score', 0)
                    result += f"- {approval_type} | {data.get('creative_angle', 'N/A')} (Score: {score:.1f}/10)\n"
                return result
            
            elif action == "get_rejected":
                patterns = self.memory.get_rejected_patterns(content_type)
                if not patterns:
                    return "No rejected patterns yet."
                
                result = "AVOID THESE APPROACHES:\n"
                for p in patterns:
                    data = p.get('pattern_data', {})
                    if isinstance(data, str):
                        data = json.loads(data)
                    result += f"- {data.get('issue', 'N/A')}: {p.get('human_feedback', 'Low score')}\n"
                return result
            
            elif action == "get_summary":
                return self.memory.get_learning_summary(content_type)
            
            else:
                return f"Unknown action: {action}"
                
        except Exception as e:
            logger.error(f"Learning memory access failed: {e}")
            return f"Memory access failed: {str(e)}"


class SaveLearningTool(BaseTool):
    """Save learning from automatic scoring (optionally with human feedback)"""
    name: str = "save_learning"
    description: str = """Save a learning pattern after scoring.
    
    Required: content_type, creative_angle, auto_score
    Optional: human_approved, human_score, human_feedback, what_worked, what_failed
    """
    
    memory: Any = Field(default=None, exclude=True)
    
    def __init__(self, memory: BrandLearningMemory):
        super().__init__()
        object.__setattr__(self, 'memory', memory)
    
    def _run(self, content_type: str, creative_angle: str, 
             auto_score: float, human_approved: bool = None,
             human_score: float = None, human_feedback: str = "",
             what_worked: str = "", what_failed: str = "") -> str:
        try:
            pattern_data = {
                "creative_angle": creative_angle,
                "what_worked": what_worked,
                "what_failed": what_failed,
                "issue": what_failed  # For consistency with rejected patterns
            }
            
            self.memory.save_learning(
                content_type=content_type,
                learning_type="creative_variation",
                pattern_data=pattern_data,
                auto_score=auto_score,
                human_approved=human_approved,
                human_score=human_score,
                human_feedback=human_feedback
            )
            
            approval_status = "human-approved" if human_approved else ("auto-approved" if auto_score >= 8.0 else "needs improvement")
            return f"Learning saved: {creative_angle} ({approval_status}, score: {auto_score:.1f})"
            
        except Exception as e:
            logger.error(f"Failed to save learning: {e}")
            return f"Failed to save: {str(e)}"


class BrandMetricsTool(BaseTool):
    """Access concrete brand metrics for scoring"""
    name: str = "brand_metrics"
    description: str = """Get concrete brand metrics extracted from documents.
    
    Returns: target sentence length, paragraph structure, signature phrases, etc.
    Use this to compare your content against measurable brand standards.
    """
    
    analyzer: Any = Field(default=None, exclude=True)
    
    def __init__(self, analyzer: BrandMetricsAnalyzer):
        super().__init__()
        object.__setattr__(self, 'analyzer', analyzer)
    
    def _run(self, action: str = "get_metrics") -> str:
        try:
            if action == "get_metrics":
                metrics = self.analyzer.metrics
                
                result = "BRAND METRICS (GROUND TRUTH):\n\n"
                result += f"Target Sentence Length: {metrics['target_sentence_length']:.1f} words\n"
                result += f"Target Sentences/Paragraph: {metrics['target_sentences_per_para']:.1f}\n"
                result += f"Uses Bullet Points: {metrics['uses_bullets']}\n"
                result += f"Uses Numbered Lists: {metrics['uses_numbered_lists']}\n"
                result += f"\nSignature Phrases ({len(metrics['signature_phrases'])}):\n"
                for i, phrase in enumerate(metrics['signature_phrases'][:15], 1):
                    result += f"{i}. {phrase}\n"
                
                return result
            
            else:
                return f"Unknown action: {action}"
                
        except Exception as e:
            logger.error(f"Metrics access failed: {e}")
            return f"Metrics access failed: {str(e)}"


# ===== GLOBAL TOOL INSTANCES =====
tavily_search = TavilySearchTool()
# ========== PART 3/3: MAIN EXECUTION AND CLI ==========
# Append this to Part 2

# ===== ADDITIONAL TOOL: AUTOMATIC CONTENT SCORING =====
class ContentScoringTool(BaseTool):
    """Automatically score content against brand metrics - NO LLM GUESSING"""
    name: str = "score_content"
    description: str = """Automatically score content structure against brand metrics.
    
    Pass the full generated content as a string.
    Returns objective scores based on actual measurements.
    
    This is NOT subjective - it counts words, sentences, paragraphs mathematically.
    """
    
    analyzer: Any = Field(default=None, exclude=True)
    
    def __init__(self, analyzer: BrandMetricsAnalyzer):
        super().__init__()
        object.__setattr__(self, 'analyzer', analyzer)
    
    def _run(self, content: str) -> str:
        try:
            scores = self.analyzer.score_content(content)
            
            result = "AUTOMATIC STRUCTURAL SCORES (Objective Measurements):\n\n"
            result += f"Overall Structure Score: {scores['structure_score']}/10\n"
            result += f"Signature Phrase Score: {scores['phrase_score']}/10\n\n"
            
            result += "DETAILED BREAKDOWN:\n"
            result += f"- Sentence Length Score: {scores['sentence_length_score']}/10\n"
            result += f"  Target: {scores['details']['target_sentence_length']} words\n"
            result += f"  Actual: {scores['details']['actual_sentence_length']} words\n\n"
            
            result += f"- Paragraph Structure Score: {scores['para_structure_score']}/10\n"
            result += f"  Target: {scores['details']['target_sentences_per_para']} sentences/para\n"
            result += f"  Actual: {scores['details']['actual_sentences_per_para']} sentences/para\n\n"
            
            result += f"- Format Alignment Score: {scores['format_score']}/10\n\n"
            
            result += f"Signature Phrases:\n"
            result += f"- Available: {scores['details']['signature_phrases_available']}\n"
            result += f"- Used: {scores['details']['signature_phrases_used']}\n"
            result += f"- Found: {', '.join(scores['details']['phrases_found']) if scores['details']['phrases_found'] else 'None'}\n"
            
            return result
            
        except Exception as e:
            logger.error(f"Content scoring failed: {e}")
            return f"Scoring failed: {str(e)}"

# ===== HUMAN FEEDBACK FUNCTIONS (Add to deployer.py) =====

def save_human_feedback(business_id: str, content_type: str, creative_angle: str,
                       auto_score: float, human_approved: bool, human_score: float,
                       human_feedback: str, generation_id: str = None):
    """
    Save human feedback for a generation.
    Can be called from CLI or web interface.
    
    Args:
        business_id: Business identifier
        content_type: blog/social/ad
        creative_angle: The creative approach used
        auto_score: Automatic score from system
        human_approved: True if human approves
        human_score: Human's score (0-10)
        human_feedback: Human's text feedback
        generation_id: Optional ID to link to specific generation
    """
    memory = BrandLearningMemory(business_id=business_id)
    
    pattern_data = {
        "creative_angle": creative_angle,
        "what_worked": human_feedback if human_approved else "",
        "what_failed": human_feedback if not human_approved else "",
        "issue": human_feedback if not human_approved else "",
        "generation_id": generation_id
    }
    
    memory.save_learning(
        content_type=content_type,
        learning_type="creative_variation",
        pattern_data=pattern_data,
        auto_score=auto_score,
        human_approved=human_approved,
        human_score=human_score,
        human_feedback=human_feedback
    )
    
    status = "Approved" if human_approved else "Rejected"
    logger.info(f"Human feedback saved: {creative_angle} - {status} ({human_score}/10)")
    
    return {
        "success": True,
        "message": f"Feedback saved: {status}",
        "score": human_score
    }


def get_feedback_stats(business_id: str, content_type: str = "blog") -> dict:
    """
    Get statistics on human feedback for analysis.
    
    Returns:
        dict with approval rates, average scores, common issues
    """
    memory = BrandLearningMemory(business_id=business_id)
    
    approved = memory.get_approved_patterns(content_type, "creative_variation", limit=100)
    rejected = memory.get_rejected_patterns(content_type, limit=100)
    
    total = len(approved) + len(rejected)
    
    if total == 0:
        return {
            "total_feedback": 0,
            "approval_rate": 0,
            "avg_human_score": 0,
            "avg_auto_score": 0,
            "message": "No feedback yet"
        }
    
    approval_rate = (len(approved) / total) * 100
    
    human_scores = [p.get('human_score') for p in approved + rejected if p.get('human_score')]
    avg_human_score = sum(human_scores) / len(human_scores) if human_scores else 0
    
    auto_scores = [p.get('auto_score') for p in approved + rejected if p.get('auto_score')]
    avg_auto_score = sum(auto_scores) / len(auto_scores) if auto_scores else 0
    
    return {
        "total_feedback": total,
        "approved": len(approved),
        "rejected": len(rejected),
        "approval_rate": round(approval_rate, 1),
        "avg_human_score": round(avg_human_score, 1),
        "avg_auto_score": round(avg_auto_score, 1)
    }


def collect_feedback_interactive_cli(content: str, auto_score: float, 
                                     creative_angle: str, business_id: str,
                                     content_type: str = "blog"):
    """
    CLI-based interactive feedback collection.
    Use this during development/testing.
    """
    print("\n" + "="*80)
    print("GENERATED CONTENT")
    print("="*80)
    print(content)
    print("="*80)
    print(f"\nAutomatic Score: {auto_score}/10")
    print(f"Creative Angle: {creative_angle}")
    print(f"Business: {business_id}")
    
    print("\n" + "-"*80)
    print("👤 HUMAN FEEDBACK")
    print("-"*80)
    
    human_approved = input("Approve this content? (y/n): ").lower() == 'y'
    human_score = float(input("Your score (0-10): "))
    
    print("\n💬 Feedback (what worked or what failed):")
    print("   (Press Enter twice when done)")
    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)
    human_feedback = "\n".join(lines)
    
    result = save_human_feedback(
        business_id=business_id,
        content_type=content_type,
        creative_angle=creative_angle,
        auto_score=auto_score,
        human_approved=human_approved,
        human_score=human_score,
        human_feedback=human_feedback
    )
    
    print("\n" + "="*80)
    print(f"{result['message']}")
    print("="*80)
    print("💡 This feedback will be used to improve future generations.")
    print("="*80 + "\n")
    
    return result

# ===== UPDATED CREW FACTORY WITH AUTOMATIC SCORING =====
class ContentCrewFactory:
    def __init__(self, kb: BrandVoiceKnowledgeBase, tavily_search: TavilySearchTool,
                 learning_memory: BrandLearningMemory, metrics_analyzer: BrandMetricsAnalyzer,
                 business_id: str):
        self.kb = kb
        self.tavily_search = tavily_search
        self.learning_memory = learning_memory
        self.metrics_analyzer = metrics_analyzer
        self.business_id = business_id
    
    def create_crew(self) -> Crew:
        kb_tool = KBQueryTool(kb=self.kb, business_id=self.business_id)
        learning_tool = LearningMemoryTool(memory=self.learning_memory)
        save_learning_tool = SaveLearningTool(memory=self.learning_memory)
        metrics_tool = BrandMetricsTool(analyzer=self.metrics_analyzer)
        scoring_tool = ContentScoringTool(analyzer=self.metrics_analyzer)  # NEW!
        
        # Use higher temperature for creative agents
        creative_llm = get_llm(temperature=0.8)
        analytical_llm = get_llm(temperature=0.5)
        
        # ===== AGENT 1: RESEARCHER =====
        researcher_agent = Agent(
            role="Research Analyst",
            goal="Find current, factual information and diverse perspectives on {topic}",
            backstory="""You gather comprehensive research including:
- Current trends and data
- Multiple perspectives and angles
- Unexpected insights and fresh takes
- Statistics and expert opinions

You provide factual information WITHOUT brand voice - that's the writer's job.""",
            tools=[self.tavily_search],
            llm=analytical_llm,
            verbose=True,
        )
        
        research_task = Task(
            description="""Research {topic} for {format} content.

Find:
1. Latest trends and developments (2026)
2. Different angles and perspectives
3. Surprising insights or contrarian views
4. Data, statistics, expert quotes
5. What competitors/others are saying

Provide 10-15 diverse factual findings.""",
            expected_output="Comprehensive research with multiple angles and perspectives",
            agent=researcher_agent,
        )
        
        # ===== AGENT 2: CREATIVE STRATEGIST =====
        creative_strategist = Agent(
            role="Creative Content Strategist",
            goal="Find unique, engaging angles that match brand voice but stand out",
            backstory="""You're a creative strategist who makes content memorable.

CRITICAL BALANCE:
Stay true to brand voice (tone, values, vocabulary)
Find fresh perspectives competitors haven't used
Make it scroll-stopping while staying authentic

You review past learnings and propose 2-3 creative angles that are:
- Fresh and haven't been overused
- Aligned with brand personality
- Engaging and memorable
- Practically valuable""",
            tools=[kb_tool, learning_tool],
            llm=creative_llm,
            verbose=True,
        )
        
        creative_strategy_task = Task(
            description="""Develop creative content strategy for {topic} in {format}.

STEP 1 - Learn from past:
Use learning_memory tool:
- {{"action": "get_summary", "content_type": "blog"}}

STEP 2 - Understand brand:
Query KB 2 times:
- {{"content_type": "blog", "query": "brand personality and values"}}
- {{"content_type": "blog", "query": "tone and voice characteristics"}}

STEP 3 - Propose 2-3 creative angles:

ANGLE 1: [Name]
- Hook: [unique perspective]
- Differentiation: [why it's fresh]
- Brand alignment: [how it matches voice]
- Example opening: "[sample]"

ANGLE 2: [Name]
...

RECOMMENDED ANGLE: [Which one and why]""",
            agent=creative_strategist,
            context=[research_task],
            expected_output="2-3 creative angles with recommended approach"
        )
        
        # ===== AGENT 3: BRAND VOICE ANALYST =====
        brand_analyst = Agent(
            role="Brand Voice Pattern Analyst",
            goal="Extract concrete metrics and qualitative patterns from brand documents",
            backstory="""You analyze brand voice using:

1. METRICS TOOL: Exact measurements (sentence length, paragraph structure, phrases)
2. KB QUERIES: Qualitative patterns (tone, style, approach)

You provide MEASURABLE CONSTRAINTS for the writer.""",
            tools=[kb_tool, metrics_tool],
            llm=analytical_llm,
            verbose=True,
        )
        
        brand_analysis_task = Task(
            description="""Extract brand voice patterns for {format}.

STEP 1 - Get Metrics:
Use brand_metrics tool: {{"action": "get_metrics"}}

STEP 2 - Query KB (4 times):
1. {{"content_type": "blog", "query": "tone characteristics"}}
2. {{"content_type": "blog", "query": "opening and closing patterns"}}
3. {{"content_type": "blog", "query": "vocabulary and perspective"}}
4. {{"content_type": "blog", "query": "content structure and format rules"}}

OUTPUT:

STRUCTURAL METRICS:
- Target sentence length: [X] words (±2 allowed)
- Target sentences/paragraph: [X] (±1 allowed)
- Format: [blog/email/social structure]
- Signature phrases: [list top 10]

TONE & STYLE:
- Primary tone: [descriptors]
- Opening pattern: [how to start]
- Closing pattern: [how to end]
- Perspective: [you/we/they]

CRITICAL FORMAT RULES:
- Content type: {format}
- Structural requirements: [specific to type]
- What to avoid: [format mistakes]""",
            agent=brand_analyst,
            context=[research_task],
            expected_output="Measurable brand blueprint with exact metrics and patterns"
        )
        
        # ===== AGENT 4: WRITER =====
        writer_agent = Agent(
            role="Brand Voice Content Writer",
            goal="Create content matching exact brand metrics while being creative",
            backstory="""You balance:
1. CREATIVE ANGLE: Fresh, engaging perspective
2. STRUCTURAL METRICS: Exact sentence/paragraph targets
3. BRAND VOICE: Tone, vocabulary, phrases

You write with precision, counting sentences and using metrics as guardrails.""",
            tools=[kb_tool, metrics_tool],
            llm=creative_llm,
            verbose=True,
            memory=True,
        )
        
        writer_task = Task(
            description="""Write {topic} content in {format} format.

CONSTRAINTS FROM BLUEPRINT:
- Sentence length target: [from analyst]
- Sentences per paragraph: [from analyst]
- Signature phrases to include: 5-8 total
- Format: Match {format} structure exactly

STEP 1 - Use recommended creative angle

STEP 2 - Write with metrics in mind:
- Count words per sentence as you write
- Count sentences per paragraph
- Include signature phrases naturally (spread them out)
- Match the exact format (blog ≠ email ≠ social)

STEP 3 - Verify format:
- BLOG: No email signature, conversational flow
- EMAIL: Include greeting/closing
- SOCIAL: Brief, platform-appropriate

OUTPUT:
[Content here]

---
SELF-CHECK METADATA:
Creative Angle: [name]
Estimated Avg Sentence Length: [X] words
Estimated Sentences/Para: [X]
Signature Phrases Used: [list]
Format Type: [Blog/Email/Social]""",
            agent=writer_agent,
            context=[research_task, creative_strategy_task, brand_analysis_task],
            expected_output="On-brand content with metadata"
        )
        
        # ===== AGENT 5: HYBRID REVIEWER (WITH AUTOMATIC SCORING) =====
        reviewer_agent = Agent(
            role="Hybrid Quality Enforcer",
            goal="Score content using AUTOMATIC metrics + LLM judgment, save learnings",
            backstory="""You use a HYBRID system:

**AUTOMATIC (50%):** Use score_content tool - it measures structure objectively
**LLM JUDGMENT (50%):** You evaluate tone, creativity, engagement

CRITICAL: You MUST use the score_content tool. Don't guess at measurements.

Scoring:
- 9.0-10: Excellent
- 8.0-8.9: Good
- 7.0-7.9: Needs revision
- <7.0: Major issues

Always save learnings.""",
            tools=[scoring_tool, learning_tool, save_learning_tool, kb_tool],
            memory=True,
            verbose=True,
            llm=analytical_llm,
        )
        
        reviewer_task = Task(
            description="""Review content with HYBRID scoring.

CRITICAL: You MUST use the score_content tool. DO NOT try to count manually.

STEP 1 - AUTOMATIC SCORING (50% weight):

Use score_content tool with the full generated content:
{{"content": "[paste the entire generated content here]"}}

This returns:
- Structure score (sentence length, paragraphs, format)
- Phrase usage score
- Detailed measurements

AUTOMATIC_SCORE = (structure_score + phrase_score) / 2

STEP 2 - LLM QUALITATIVE SCORING (50% weight):

You evaluate:

A) Tone Alignment (0-10):
   - Matches brand tone?
   - Appropriate vocabulary?
   - Correct perspective?
   
B) Creative Freshness (0-10):
   - Unique angle?
   - Different from templates?
   - Engaging hook?
   
C) Practical Value (0-10):
   - Actionable insights?
   - Clear takeaways?
   - Useful to audience?

LLM_SCORE = (A + B + C) / 3

STEP 3 - FINAL SCORE:

FINAL_SCORE = (AUTOMATIC_SCORE × 0.5) + (LLM_SCORE × 0.5)

STEP 4 - DECISION:
- >= 9.0: APPROVED
- 8.0-8.9: APPROVED (minor suggestions)
- 7.0-7.9: REVISION NEEDED (specific fixes)
- < 7.0: MAJOR REVISION (detailed issues)

STEP 5 - IDENTIFY ISSUES:

Based on automatic scores:
- Sentence length off? → "Sentences too long/short"
- Paragraph structure off? → "Paragraphs need adjustment"
- Missing phrases? → "Include more signature phrases"
- Format wrong? → "Format mismatch: blog vs email"

Based on LLM scores:
- Tone issues? → "Tone too formal/casual"
- Not creative? → "Generic angle, needs fresh perspective"
- Low value? → "Add practical takeaways"

STEP 6 - SAVE LEARNING:

Use save_learning tool:
{{
  "content_type": "blog",
  "creative_angle": "[from writer metadata]",
  "auto_score": [FINAL_SCORE],
  "what_worked": "[if >= 8.0]",
  "what_failed": "[if < 8.0 - be specific]"
}}

STEP 7 - OUTPUT (STRICT JSON FORMAT):

You MUST return ONLY valid JSON with this exact structure:

{
  "decision": "APPROVED",
  "final_score": 8.5,
  "automatic_score": 8.2,
  "llm_score": 8.8,
  "creative_angle": "Nourishing Immunity",
  "generated_content": "[full content here]",
  "detailed_scores": {
    "structure": 8.2,
    "phrase_usage": 7.5,
    "tone_alignment": 9.0,
    "creative_freshness": 8.5,
    "practical_value": 9.0
  },
  "measurements": {
    "target_sentence_length": 14,
    "actual_sentence_length": 18,
    "target_sentences_per_para": 3,
    "actual_sentences_per_para": 4,
    "signature_phrases_used": 5,
    "signature_phrases_available": 15
  },
  "issues": {
    "critical": ["Sentences too long", "Format mismatch"],
    "suggestions": ["Add more signature phrases"]
  },
  "learning_saved": true
}

DO NOT add any text before or after the JSON. ONLY return JSON.""",
    agent=reviewer_agent,
    context=[writer_task, brand_analysis_task, creative_strategy_task],
    expected_output="Valid JSON object with scores, content, and metadata"
)
        
        return Crew(
            agents=[researcher_agent, creative_strategist, brand_analyst, writer_agent, reviewer_agent],
            tasks=[research_task, creative_strategy_task, brand_analysis_task, writer_task, reviewer_task],
            process=Process.sequential,
            verbose=True,
        )


# ===== FACTORY FUNCTION =====
def get_crew(business_id: str = "default", topic: str = "AI Agents", 
             format_type: str = "Blog Article", voice: str = "formal") -> tuple:
    """
    Create a crew with hybrid reflection learning.
    
    Returns: (crew, kb, learning_memory, metrics_analyzer)
    """
    load_dotenv()
    
    # Determine content type
    if "blog" in format_type.lower():
        content_type = "blog"
    elif "social" in format_type.lower():
        content_type = "social"
    elif "ad" in format_type.lower() or "advertisement" in format_type.lower():
        content_type = "ad"
    else:
        content_type = "blog"
    
    kb = BrandVoiceKnowledgeBase()
    learning_memory = BrandLearningMemory(business_id=business_id)
    metrics_analyzer = BrandMetricsAnalyzer(business_id=business_id, content_type=content_type)
    
    factory = ContentCrewFactory(
        kb=kb,
        tavily_search=tavily_search,
        learning_memory=learning_memory,
        metrics_analyzer=metrics_analyzer,
        business_id=business_id
    )
    
    crew = factory.create_crew()
    return crew, kb, learning_memory, metrics_analyzer



# ===== CLI ENTRY POINT =====
# At the very end of deployer.py, replace your existing if __name__ == "__main__": block

# ===== CLI ENTRY POINT (REPLACE EXISTING) =====
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate brand-aligned marketing content")
    parser.add_argument("--topic", default="AI Agents", help="Content topic")
    parser.add_argument("--format", default="Blog Article", help="Content format")
    parser.add_argument("--voice", default="formal", help="Voice style")
    parser.add_argument("--business_id", default="BAND_Foods", help="Business identifier")
    parser.add_argument("--collect-feedback", action="store_true", 
                       help="Collect human feedback after generation")
    args = parser.parse_args()
    
    print(f"\nStarting Hybrid Reflection Content Generation")
    print(f"Topic: {args.topic}")
    print(f"Format: {args.format}")
    print(f"Business: {args.business_id}")
    print("="*60 + "\n")
    
    crew, kb, learning_memory, metrics_analyzer = get_crew(
        business_id=args.business_id,
        topic=args.topic,
        format_type=args.format,
        voice=args.voice
    )
    
    # Show learning summary
    content_type = "blog" if "blog" in args.format.lower() else "social" if "social" in args.format.lower() else "ad"
    print("\nLEARNING MEMORY SUMMARY")
    print("="*60)
    print(learning_memory.get_learning_summary(content_type))
    print("="*60 + "\n")
    
    # Show brand metrics
    print("\nBRAND METRICS (GROUND TRUTH)")
    print("="*60)
    print(f"Target Sentence Length: {metrics_analyzer.metrics['target_sentence_length']:.1f} words")
    print(f"Target Sentences/Paragraph: {metrics_analyzer.metrics['target_sentences_per_para']:.1f}")
    print(f"Signature Phrases Available: {len(metrics_analyzer.metrics['signature_phrases'])}")
    print(f"Sample Documents Analyzed: {metrics_analyzer.metrics['sample_count']}")
    print("="*60 + "\n")
    
    result = crew.kickoff(inputs={
        "topic": args.topic,
        "format": args.format,
        "voice": args.voice
    })
    
    print("\n" + "="*60)
    print("GENERATION COMPLETE")
    print("="*60)
    print(result)
    print("\n" + "="*60)
    
    print("\n💡 TO ADD HUMAN FEEDBACK:")
    print("="*60)
    print("from deployer import save_human_feedback")
    print()
    print("save_human_feedback(")
    print(f"    business_id='{args.business_id}',")
    print(f"    content_type='{content_type}',")
    print("    creative_angle='[angle from output]',")
    print("    auto_score=8.5,  # from automatic scoring")
    print("    human_approved=True,  # or False")
    print("    human_score=9.5,  # your score")
    print("    human_feedback='What worked or what failed'")
    print(")")
    print("="*60 + "\n")