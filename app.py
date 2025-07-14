# app.py - Optimized OpsVoice Backend
# Production-ready with enhanced performance, security, and maintainability

import os
import json
import time
import asyncio
import secrets
import hashlib
import logging
import re
import glob
import shutil
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from functools import lru_cache, wraps
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from io import BytesIO

from flask import Flask, request, jsonify, send_file, send_from_directory, make_response
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_caching import Cache
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

import requests
from langchain_community.document_loaders import UnstructuredWordDocumentLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==== Configuration ====
@dataclass
class Config:
    # Paths
    DATA_PATH: str = os.getenv("DATA_PATH", "/data" if os.path.exists("/data") else "./data")
    SOP_FOLDER: str = None
    CHROMA_DIR: str = None
    AUDIO_CACHE_DIR: str = None
    STATUS_FILE: str = None
    
    # API Keys
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    ELEVENLABS_API_KEY: str = os.getenv("ELEVENLABS_API_KEY")
    
    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", secrets.token_hex(32))
    API_KEY: str = os.getenv("API_KEY")  # Optional API key for additional security
    
    # Limits
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    MAX_QUERY_LENGTH: int = 500
    MAX_CACHE_SIZE: int = 1000
    AUDIO_CACHE_TTL: int = 3600 * 24  # 24 hours
    QUERY_CACHE_TTL: int = 3600  # 1 hour
    
    # Performance
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    RETRIEVAL_K: int = 5
    EMBEDDING_BATCH_SIZE: int = 100
    
    # Rate Limiting - Different limits for demo vs. regular companies
    DEMO_RATE_LIMIT_PER_MINUTE: int = 60  # Looser for demo
    DEMO_RATE_LIMIT_PER_HOUR: int = 1000   # Looser for demo
    RATE_LIMIT_PER_MINUTE: int = 20        # Tighter for real companies
    RATE_LIMIT_PER_HOUR: int = 300         # Tighter for real companies
    
    # Document Limits
    DEMO_MAX_DOCUMENTS: int = 10           # Demo can have more docs
    COMPANY_MAX_DOCUMENTS: int = 5         # Regular companies limited
    
    # Redis (optional)
    REDIS_URL: str = os.getenv("REDIS_URL")
    
    # Feature Flags
    USE_REDIS_SESSIONS: bool = bool(os.getenv("REDIS_URL"))
    USE_ASYNC_TTS: bool = False  # Set to False since we don't have aiohttp by default
    ENABLE_METRICS: bool = True
    
    def __post_init__(self):
        # Initialize dependent paths
        self.SOP_FOLDER = os.path.join(self.DATA_PATH, "sop-files")
        self.CHROMA_DIR = os.path.join(self.DATA_PATH, "chroma_db")
        self.AUDIO_CACHE_DIR = os.path.join(self.DATA_PATH, "audio_cache")
        self.STATUS_FILE = os.path.join(self.SOP_FOLDER, "status.json")
        
        # Create directories
        for path in [self.SOP_FOLDER, self.CHROMA_DIR, self.AUDIO_CACHE_DIR]:
            os.makedirs(path, exist_ok=True)

config = Config()

# ==== Session Management ====
class SessionManager:
    def __init__(self, use_redis: bool = False, redis_url: str = None):
        self.use_redis = use_redis and redis_url
        if self.use_redis:
            try:
                import redis
                self.redis_client = redis.from_url(redis_url, decode_responses=True)
                logger.info("Redis session storage initialized")
            except ImportError:
                logger.warning("Redis not installed, falling back to memory storage")
                self.use_redis = False
                self.memory_store = {}
        else:
            self.memory_store = {}
    
    def get_session(self, session_id: str) -> Dict:
        """Get session data"""
        if self.use_redis:
            try:
                data = self.redis_client.get(f"session:{session_id}")
                return json.loads(data) if data else {}
            except:
                return {}
        return self.memory_store.get(session_id, {})
    
    def set_session(self, session_id: str, data: Dict, ttl: int = 3600):
        """Set session data with TTL"""
        if self.use_redis:
            try:
                self.redis_client.setex(
                    f"session:{session_id}",
                    ttl,
                    json.dumps(data)
                )
            except:
                pass
        else:
            self.memory_store[session_id] = {
                **data,
                '_expires': time.time() + ttl
            }
    
    def get_conversation_memory(self, session_id: str) -> ConversationBufferMemory:
        """Get or create conversation memory for session"""
        session_data = self.get_session(session_id)
        
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Restore chat history if exists
        if 'chat_history' in session_data:
            # Reconstruct memory from stored messages
            for msg in session_data['chat_history']:
                if msg['type'] == 'human':
                    memory.chat_memory.add_user_message(msg['content'])
                else:
                    memory.chat_memory.add_ai_message(msg['content'])
        
        return memory
    
    def save_conversation_memory(self, session_id: str, memory: ConversationBufferMemory):
        """Save conversation memory to session"""
        messages = []
        for msg in memory.chat_memory.messages:
            messages.append({
                'type': 'human' if msg.__class__.__name__ == 'HumanMessage' else 'ai',
                'content': msg.content
            })
        
        session_data = self.get_session(session_id)
        session_data['chat_history'] = messages[-20:]  # Keep last 20 messages
        self.set_session(session_id, session_data)
    
    def cleanup_expired(self):
        """Clean up expired sessions (memory store only)"""
        if not self.use_redis:
            current_time = time.time()
            expired = [
                sid for sid, data in self.memory_store.items()
                if data.get('_expires', float('inf')) < current_time
            ]
            for sid in expired:
                del self.memory_store[sid]

# ==== Performance Monitoring ====
class PerformanceMonitor:
    def __init__(self):
        self.metrics = {
            'total_queries': 0,
            'cache_hits': 0,
            'avg_response_time': 0,
            'response_times': [],
            'model_usage': {'gpt-3.5-turbo': 0, 'gpt-4': 0},
            'sources': {}
        }
    
    def track_query(self, response_time: float, model: str, source: str, cache_hit: bool = False):
        """Track query performance metrics"""
        self.metrics['total_queries'] += 1
        self.metrics['response_times'].append(response_time)
        
        # Keep only last 100 response times
        if len(self.metrics['response_times']) > 100:
            self.metrics['response_times'] = self.metrics['response_times'][-100:]
        
        self.metrics['avg_response_time'] = sum(self.metrics['response_times']) / len(self.metrics['response_times'])
        
        if cache_hit:
            self.metrics['cache_hits'] += 1
        
        if model in self.metrics['model_usage']:
            self.metrics['model_usage'][model] += 1
        
        self.metrics['sources'][source] = self.metrics['sources'].get(source, 0) + 1
    
    def get_metrics(self) -> Dict:
        """Get current metrics"""
        cache_rate = (self.metrics['cache_hits'] / max(1, self.metrics['total_queries'])) * 100
        return {
            **self.metrics,
            'cache_hit_rate': round(cache_rate, 2),
            'avg_response_time': round(self.metrics['avg_response_time'], 3)
        }

# ==== Enhanced Cache Manager ====
class CacheManager:
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.ttl = ttl
        self.hits = 0
        self.misses = 0
    
    def _make_key(self, query: str, company_id: str) -> str:
        """Generate cache key"""
        combined = f"{company_id}:{query.lower().strip()}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def get(self, query: str, company_id: str) -> Optional[Dict]:
        """Get cached response"""
        key = self._make_key(query, company_id)
        
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry['timestamp'] < self.ttl:
                # Move to end (LRU)
                self.cache.move_to_end(key)
                self.hits += 1
                return entry['data']
            else:
                # Expired
                del self.cache[key]
        
        self.misses += 1
        return None
    
    def set(self, query: str, company_id: str, data: Dict):
        """Set cached response"""
        key = self._make_key(query, company_id)
        
        # Add to cache
        self.cache[key] = {
            'data': data,
            'timestamp': time.time()
        }
        
        # Evict oldest if over size limit
        while len(self.cache) > self.max_size:
            self.cache.popitem(last=False)
    
    def clear_company(self, company_id: str):
        """Clear all cache entries for a company"""
        to_remove = [
            key for key in self.cache
            if key.startswith(hashlib.sha256(f"{company_id}:".encode()).hexdigest()[:16])
        ]
        for key in to_remove:
            del self.cache[key]

# ==== Vector Store Manager ====
class VectorStoreManager:
    def __init__(self, persist_dir: str):
        self.persist_dir = persist_dir
        self.embedding = OpenAIEmbeddings()
        self.vectorstore = None
        self.executor = ThreadPoolExecutor(max_workers=2)
        self._load_vectorstore()
    
    def _load_vectorstore(self):
        """Load or create vector store"""
        try:
            self.vectorstore = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embedding
            )
            logger.info("Vector store loaded successfully")
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            self.vectorstore = None
    
    def add_documents(self, documents: List[Document], company_id: str):
        """Add documents to vector store with company metadata"""
        if not self.vectorstore:
            self._load_vectorstore()
        
        # Ensure all documents have company_id in metadata
        for doc in documents:
            doc.metadata['company_id_slug'] = company_id
        
        # Add in batches for better performance
        batch_size = config.EMBEDDING_BATCH_SIZE
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            self.vectorstore.add_documents(batch)
        
        self.vectorstore.persist()
    
    def search(self, query: str, company_id: str, k: int = 5) -> List[Document]:
        """Search documents for a specific company"""
        if not self.vectorstore:
            return []
        
        try:
            # Search with company filter
            results = self.vectorstore.similarity_search(
                query,
                k=k,
                filter={"company_id_slug": company_id}
            )
            return results
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

# ==== Audio Service ====
class AudioService:
    def __init__(self, api_key: str, cache_dir: str):
        self.api_key = api_key
        self.cache_dir = cache_dir
        self.voice_id = "tnSpp4vdxKPjI9w0GnoV"  # Default voice
    
    def generate_audio_sync(self, text: str, cache_key: str) -> bytes:
        """Generate audio synchronously"""
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.mp3")
        
        # Check cache first
        if os.path.exists(cache_path):
            # Check if not too old
            if time.time() - os.path.getmtime(cache_path) < config.AUDIO_CACHE_TTL:
                with open(cache_path, 'rb') as f:
                    return f.read()
        
        # Generate new audio
        try:
            response = requests.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}/stream",
                headers={
                    "xi-api-key": self.api_key,
                    "Content-Type": "application/json"
                },
                json={
                    "text": text[:500],  # Limit text length
                    "voice_settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.75,
                        "style": 0.0,
                        "use_speaker_boost": True
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                audio_data = response.content
                
                # Cache the audio
                with open(cache_path, 'wb') as f:
                    f.write(audio_data)
                
                return audio_data
            else:
                raise Exception(f"TTS API error: {response.status_code}")
        except Exception as e:
            logger.error(f"Audio generation error: {e}")
            raise
    
    def cleanup_old_cache(self):
        """Remove old cache files"""
        cutoff_time = time.time() - config.AUDIO_CACHE_TTL
        for filename in os.listdir(self.cache_dir):
            filepath = os.path.join(self.cache_dir, filename)
            try:
                if os.path.getmtime(filepath) < cutoff_time:
                    os.remove(filepath)
            except:
                pass

# ==== Initialize Services ====
session_manager = SessionManager(
    use_redis=config.USE_REDIS_SESSIONS,
    redis_url=config.REDIS_URL
)
performance_monitor = PerformanceMonitor()
cache_manager = CacheManager(
    max_size=config.MAX_CACHE_SIZE,
    ttl=config.QUERY_CACHE_TTL
)
vector_store_manager = VectorStoreManager(config.CHROMA_DIR)
audio_service = AudioService(config.ELEVENLABS_API_KEY, config.AUDIO_CACHE_DIR)

# ==== Flask App ====
app = Flask(__name__)
app.config['SECRET_KEY'] = config.SECRET_KEY
app.config['MAX_CONTENT_LENGTH'] = config.MAX_FILE_SIZE

# CORS Configuration
CORS(app, resources={
    r"/*": {
        "origins": [
            "https://opsvoice-widget.vercel.app",
            "http://localhost:3000",
            "http://127.0.0.1:3000"
        ],
        "supports_credentials": True,
        "allow_headers": ["Content-Type", "Authorization", "X-API-Key"],
        "methods": ["GET", "POST", "OPTIONS"]
    }
})

# Rate Limiting with different limits for demo
def get_rate_limit_key():
    """Get rate limit key based on company"""
    company_id = request.json.get('company_id_slug', '') if request.json else ''
    if company_id == 'demo-business-123':
        return f"demo:{get_remote_address()}"
    return f"{get_remote_address()}:{company_id}"

limiter = Limiter(
    app=app,
    key_func=get_rate_limit_key,
    storage_uri=config.REDIS_URL if config.REDIS_URL else "memory://",
)

# Flask-Caching
cache = Cache(app, config={
    'CACHE_TYPE': 'simple',
    'CACHE_DEFAULT_TIMEOUT': 300
})

# ==== Security Decorators ====
def require_api_key(f):
    """Require API key if configured"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if config.API_KEY:
            provided_key = request.headers.get('X-API-Key') or request.args.get('api_key')
            if not provided_key or provided_key != config.API_KEY:
                return jsonify({'error': 'Invalid or missing API key'}), 401
        return f(*args, **kwargs)
    return decorated_function

def validate_company_id(f):
    """Validate company ID format"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        company_id = (
            kwargs.get('company_id_slug') or
            request.json.get('company_id_slug', '') if request.json else '' or
            request.form.get('company_id_slug', '') or
            request.args.get('company_id_slug', '')
        )
        
        if not company_id or not re.match(r'^[a-zA-Z0-9_-]+$', company_id):
            return jsonify({'error': 'Invalid company identifier'}), 400
        
        # Prevent path traversal
        if '..' in company_id or '/' in company_id or '\\' in company_id:
            return jsonify({'error': 'Invalid company identifier'}), 400
        
        return f(*args, **kwargs)
    return decorated_function

# ==== Utility Functions ====
def clean_text(text: str) -> str:
    """Clean text for processing and TTS"""
    if not text:
        return ""
    
    # Remove problematic characters
    text = text.replace('\u2022', '-').replace('\t', ' ')
    text = ' '.join(text.split())  # Normalize whitespace
    
    # Remove potential XSS/injection attempts
    text = re.sub(r'[<>\"\'`]', '', text)
    
    return text.strip()

def get_query_complexity(query: str) -> str:
    """Determine query complexity for model selection"""
    words = query.lower().split()
    
    # Simple queries
    if len(words) <= 8 or any(word in query.lower() for word in ['what', 'when', 'where', 'who', 'how many']):
        return 'simple'
    
    # Complex queries
    if len(words) > 15 or any(word in query.lower() for word in ['analyze', 'compare', 'explain', 'walk me through']):
        return 'complex'
    
    return 'medium'

def get_optimal_llm(complexity: str) -> ChatOpenAI:
    """Select optimal LLM based on query complexity"""
    if complexity == 'simple':
        return ChatOpenAI(temperature=0, model='gpt-3.5-turbo', request_timeout=30)
    else:
        return ChatOpenAI(temperature=0, model='gpt-4', request_timeout=60)

def generate_followup_questions(query: str, answer: str) -> List[str]:
    """Generate contextual follow-up questions"""
    followups = []
    
    # Based on answer content
    if any(word in answer.lower() for word in ['procedure', 'process', 'steps']):
        followups.append("Would you like more details about this process?")
    
    if any(word in answer.lower() for word in ['policy', 'rule', 'guideline']):
        followups.append("Are there any exceptions to this policy?")
    
    # Based on query type
    if 'how' in query.lower():
        followups.append("Do you need help with a specific part?")
    elif 'what' in query.lower():
        followups.append("Would you like examples?")
    
    # Default fallbacks
    if len(followups) < 2:
        followups.extend([
            "Can I clarify anything else?",
            "Do you have a related question?"
        ])
    
    return followups[:3]

def get_company_document_count(company_id: str) -> int:
    """Get count of documents for a company"""
    if not os.path.exists(config.STATUS_FILE):
        return 0
    
    try:
        with open(config.STATUS_FILE, 'r') as f:
            data = json.load(f)
        
        count = sum(1 for doc in data.values() 
                   if doc.get('company_id_slug') == company_id 
                   and doc.get('status') == 'completed')
        return count
    except:
        return 0

# ==== Document Processing ====
def process_document_async(filepath: str, metadata: Dict):
    """Process document asynchronously"""
    try:
        filename = os.path.basename(filepath)
        ext = filename.rsplit('.', 1)[-1].lower()
        
        # Load document
        if ext == 'pdf':
            loader = PyPDFLoader(filepath)
        elif ext == 'docx':
            loader = UnstructuredWordDocumentLoader(filepath)
        elif ext == 'txt':
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            docs = [Document(page_content=content, metadata={"source": filepath})]
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        
        if ext != 'txt':
            docs = loader.load()
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]
        )
        chunks = text_splitter.split_documents(docs)
        
        # Add metadata to chunks
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                **metadata,
                'chunk_index': i,
                'total_chunks': len(chunks)
            })
        
        # Add to vector store
        vector_store_manager.add_documents(chunks, metadata['company_id_slug'])
        
        # Update status
        update_document_status(filename, {
            'status': 'completed',
            'chunks': len(chunks),
            **metadata
        })
        
        logger.info(f"Successfully processed {filename}: {len(chunks)} chunks")
        
    except Exception as e:
        logger.error(f"Error processing {filename}: {e}")
        update_document_status(filename, {
            'status': 'error',
            'error': str(e),
            **metadata
        })

def update_document_status(filename: str, status: Dict):
    """Update document processing status"""
    try:
        status_file = config.STATUS_FILE
        
        # Load existing status
        if os.path.exists(status_file):
            with open(status_file, 'r') as f:
                data = json.load(f)
        else:
            data = {}
        
        # Update status
        data[filename] = {
            **status,
            'updated_at': datetime.now().isoformat()
        }
        
        # Save status
        with open(status_file, 'w') as f:
            json.dump(data, f, indent=2)
            
    except Exception as e:
        logger.error(f"Error updating status for {filename}: {e}")

def generate_fallback_answer(query: str, company_id: str) -> str:
    """Generate helpful fallback answer when no documents found"""
    query_lower = query.lower()
    
    # Customer service queries
    if any(word in query_lower for word in ['angry', 'upset', 'difficult', 'complaint']):
        return """I don't see specific customer service procedures in your documents, but here are general best practices:

1. **Listen actively** - Let the customer express their concerns fully
2. **Stay calm and professional** - Don't take it personally
3. **Acknowledge their feelings** - Show empathy
4. **Apologize sincerely** - Even if it's not your fault
5. **Find a solution** - Focus on what you can do
6. **Follow up** - Ensure the issue is resolved

For company-specific procedures, please check with your manager or upload your customer service guidelines."""

    # Process/procedure queries
    elif any(word in query_lower for word in ['process', 'procedure', 'how to', 'steps']):
        return """I couldn't find that specific procedure in your uploaded documents. 

To get accurate information:
1. Check if this procedure has been uploaded to the system
2. Ask your supervisor for the current process
3. Look for related procedures that might help

Would you like to know what documents are currently available?"""

    # Policy queries
    elif any(word in query_lower for word in ['policy', 'rule', 'allowed', 'permitted']):
        return """I don't see that policy in your current documents.

For accurate policy information:
- Check with your HR department
- Review your employee handbook
- Ask your manager for clarification

Company policies should be documented and uploaded to ensure everyone has access to the same information."""

    # Default fallback
    else:
        return f"""I couldn't find specific information about that in your documents.

This might be because:
- The relevant document hasn't been uploaded yet
- The information is under a different topic
- This requires checking with your supervisor

Try asking about other topics, or ensure the relevant documents are uploaded to the system."""

# ==== Routes ====
@app.route('/')
def home():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'version': '2.0.0',
        'features': [
            'multi-tenant-rag',
            'conversational-memory',
            'performance-optimized',
            'secure-api',
            'audio-tts',
            'smart-caching'
        ],
        'performance': performance_monitor.get_metrics() if config.ENABLE_METRICS else None
    })

@app.route('/healthz')
def health():
    """Detailed health check"""
    sop_files = glob.glob(os.path.join(config.SOP_FOLDER, "*.pdf")) + \
                glob.glob(os.path.join(config.SOP_FOLDER, "*.docx")) + \
                glob.glob(os.path.join(config.SOP_FOLDER, "*.txt"))
    
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'services': {
            'vectorstore': vector_store_manager.vectorstore is not None,
            'redis': session_manager.use_redis,
            'cache_hits': cache_manager.hits,
            'cache_misses': cache_manager.misses,
            'sop_files_count': len(sop_files)
        },
        'data_path': config.DATA_PATH,
        'persistent_storage': os.path.exists("/data"),
        'metrics': performance_monitor.get_metrics() if config.ENABLE_METRICS else None
    })

@app.route('/upload-sop', methods=['POST'])
@validate_company_id
def upload_sop():
    """Upload and process SOP document"""
    try:
        # Get company ID
        company_id = request.form.get('company_id_slug', '').strip()
        
        # Apply rate limits based on company
        if company_id == 'demo-business-123':
            # Demo has looser limits - checked by decorator
            pass
        else:
            # Check document count for regular companies
            current_count = get_company_document_count(company_id)
            if current_count >= config.COMPANY_MAX_DOCUMENTS:
                return jsonify({
                    'error': f'Document limit reached. Maximum {config.COMPANY_MAX_DOCUMENTS} documents allowed.',
                    'current_count': current_count
                }), 400
        
        # Validate file
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if not file or not file.filename:
            return jsonify({'error': 'No file selected'}), 400
        
        # Secure filename
        filename = secure_filename(file.filename)
        if not filename:
            return jsonify({'error': 'Invalid filename'}), 400
        
        # Check extension
        ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
        if ext not in ['pdf', 'docx', 'txt']:
            return jsonify({'error': 'Invalid file type. Only PDF, DOCX, and TXT allowed'}), 400
        
        # Get metadata
        title = request.form.get('doc_title', filename)[:200]
        
        # Generate unique filename
        timestamp = int(time.time())
        unique_filename = f"{company_id}_{timestamp}_{filename}"
        filepath = os.path.join(config.SOP_FOLDER, unique_filename)
        
        # Save file
        file.save(filepath)
        
        # Verify file was saved
        if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
            return jsonify({'error': 'File save failed'}), 500
        
        # Prepare metadata (FIXED - removed duplicate line)
        metadata = {
            'company_id_slug': company_id,
            'title': title,
            'filename': unique_filename,
            'original_filename': filename,
            'uploaded_at': datetime.now().isoformat(),
            'file_size': os.path.getsize(filepath)
        }
        
        # Update status
        update_document_status(unique_filename, {
            'status': 'processing',
            **metadata
        })
        
        # Process document asynchronously
        vector_store_manager.executor.submit(process_document_async, filepath, metadata)
        
        # Clear cache for this company
        cache_manager.clear_company(company_id)
        
        return jsonify({
            'message': 'Document uploaded successfully',
            'doc_id': unique_filename,
            'status': 'processing',
            'sop_file_url': f"{request.host_url.rstrip('/')}/static/sop-files/{unique_filename}",
            **metadata
        }), 201
        
    except RequestEntityTooLarge:
        return jsonify({'error': 'File too large. Maximum size is 10MB'}), 413
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({'error': 'Upload failed'}), 500
        
        # Prepare metadata
        metadata = {
            'company_id_slug': company_id,
            'title': title,
            'filename': unique_filename,
            'original_filename': filename,
            'uploaded_at': datetime.now().isoformat(),
            'file_size': os.path.getsize(filepath)
        }
        
        # Update status
        update_document_status(unique_filename, {
            'status': 'processing',
            **metadata
        })
        
        # Process document asynchronously
        vector_store_manager.executor.submit(process_document_async, filepath, metadata)
        
        # Clear cache for this company
        cache_manager.clear_company(company_id)
        
        return jsonify({
            'message': 'Document uploaded successfully',
            'doc_id': unique_filename,
            'status': 'processing',
            'sop_file_url': f"{request.host_url.rstrip('/')}/static/sop-files/{unique_filename}",
            **metadata
        }), 201
        
    except RequestEntityTooLarge:
        return jsonify({'error': 'File too large. Maximum size is 10MB'}), 413
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({'error': 'Upload failed'}), 500

# Apply different rate limits for demo vs regular companies
@app.route('/query', methods=['POST'])
@validate_company_id
def query_sop():
   """Process user query with RAG"""
   start_time = time.time()
   
   try:
       data = request.get_json()
       if not data:
           return jsonify({'error': 'Invalid request format'}), 400
       
       # Extract and validate inputs
       query = clean_text(data.get('query', ''))
       company_id = data.get('company_id_slug', '').strip()
       session_id = data.get('session_id', f"{company_id}_{int(time.time())}")
       
       # Apply rate limits
       if company_id == 'demo-business-123':
           # Demo gets 60/min, 1000/hour
           @limiter.limit(f"{config.DEMO_RATE_LIMIT_PER_MINUTE} per minute; {config.DEMO_RATE_LIMIT_PER_HOUR} per hour")
           def demo_query():
               pass
           demo_query()
       else:
           # Regular companies get 20/min, 300/hour
           @limiter.limit(f"{config.RATE_LIMIT_PER_MINUTE} per minute; {config.RATE_LIMIT_PER_HOUR} per hour")
           def regular_query():
               pass
           regular_query()
       
       if not query:
           return jsonify({'error': 'Query is required'}), 400
       
       if len(query) > config.MAX_QUERY_LENGTH:
           return jsonify({'error': f'Query too long. Maximum {config.MAX_QUERY_LENGTH} characters'}), 400
       
       # Check cache first
       cached_response = cache_manager.get(query, company_id)
       if cached_response:
           performance_monitor.track_query(
               time.time() - start_time,
               cached_response.get('model_used', 'gpt-3.5-turbo'),
               'cache',
               cache_hit=True
           )
           return jsonify({
               **cached_response,
               'cache_hit': True,
               'response_time': time.time() - start_time
           })
       
       # Get or create conversation memory
       memory = session_manager.get_conversation_memory(session_id)
       
       # Determine query complexity and select model
       complexity = get_query_complexity(query)
       llm = get_optimal_llm(complexity)
       
       # Search for relevant documents
       search_results = vector_store_manager.search(query, company_id, k=config.RETRIEVAL_K)
       
       if not search_results and company_id == 'demo-business-123':
           # For demo, also search without filter to find any demo content
           search_results = vector_store_manager.search(query, '', k=3)
       
       # Create retriever
       if vector_store_manager.vectorstore:
           retriever = vector_store_manager.vectorstore.as_retriever(
               search_kwargs={
                   'k': len(search_results) if search_results else 1,
                   'filter': {'company_id_slug': company_id}
               }
           )
           
           # Create QA chain
           qa_chain = ConversationalRetrievalChain.from_llm(
               llm=llm,
               retriever=retriever,
               memory=memory,
               return_source_documents=True,
               verbose=False
           )
           
           # Get answer
           result = qa_chain.invoke({'question': query})
           answer = clean_text(result.get('answer', ''))
           
           # Check if answer is helpful
           if not answer or len(answer.split()) < 10 or 'I don\'t know' in answer:
               # Generate helpful fallback
               answer = generate_fallback_answer(query, company_id)
               source = 'fallback'
           else:
               source = 'documents'
       else:
           # No vectorstore available
           answer = generate_fallback_answer(query, company_id)
           source = 'fallback'
       
       # Save conversation memory
       session_manager.save_conversation_memory(session_id, memory)
       
       # Prepare response
       response_data = {
           'answer': answer,
           'session_id': session_id,
           'source': source,
           'model_used': llm.model_name,
           'complexity': complexity,
           'followups': generate_followup_questions(query, answer),
           'documents_found': len(search_results),
           'response_time': round(time.time() - start_time, 3)
       }
       
       # Cache the response
       cache_manager.set(query, company_id, response_data)
       
       # Track performance
       performance_monitor.track_query(
           time.time() - start_time,
           llm.model_name,
           source
       )
       
       return jsonify(response_data)
       
   except Exception as e:
       logger.error(f"Query error: {e}", exc_info=True)
       performance_monitor.track_query(
           time.time() - start_time,
           'unknown',
           'error'
       )
       return jsonify({
           'error': 'Failed to process query',
           'session_id': session_id if 'session_id' in locals() else None,
           'response_time': round(time.time() - start_time, 3)
       }), 500

@app.route('/voice-reply', methods=['POST'])
def voice_reply():
   """Generate audio response"""
   try:
       data = request.get_json()
       if not data:
           return jsonify({'error': 'Invalid request format'}), 400
       
       text = clean_text(data.get('query', ''))
       company_id = data.get('company_id_slug', 'default')
       
       if not text:
           return jsonify({'error': 'Text is required'}), 400
       
       # Apply rate limits based on company
       if company_id == 'demo-business-123':
           @limiter.limit(f"40 per minute")  # More generous for demo
           def demo_tts():
               pass
           demo_tts()
       else:
           @limiter.limit(f"20 per minute")  # Tighter for regular companies
           def regular_tts():
               pass
           regular_tts()
       
       # Limit text length for TTS
       text = text[:500]
       
       # Generate cache key
       cache_key = hashlib.sha256(f"{company_id}:{text}".encode()).hexdigest()[:16]
       
       try:
           # Generate audio (uses cache internally)
           audio_data = audio_service.generate_audio_sync(text, cache_key)
           
           # Return audio
           return send_file(
               BytesIO(audio_data),
               mimetype='audio/mpeg',
               as_attachment=False,
               download_name='response.mp3'
           )
       except Exception as e:
           logger.error(f"Audio generation error: {e}")
           return jsonify({'error': 'Audio generation failed'}), 502
           
   except Exception as e:
       logger.error(f"Voice reply error: {e}")
       return jsonify({'error': 'Audio generation failed'}), 500

@app.route('/company-docs/<company_id_slug>')
@validate_company_id
@cache.cached(timeout=60)
def get_company_documents(company_id_slug):
   """Get list of documents for a company"""
   try:
       if not os.path.exists(config.STATUS_FILE):
           return jsonify([])
       
       with open(config.STATUS_FILE, 'r') as f:
           all_docs = json.load(f)
       
       # Filter documents for this company
       company_docs = []
       for filename, metadata in all_docs.items():
           if metadata.get('company_id_slug') == company_id_slug:
               company_docs.append({
                   'id': filename,
                   'title': metadata.get('title', filename),
                   'filename': metadata.get('original_filename', filename),
                   'status': metadata.get('status', 'unknown'),
                   'uploaded_at': metadata.get('uploaded_at'),
                   'file_size': metadata.get('file_size'),
                   'chunks': metadata.get('chunks', 0),
                   'sop_file_url': f"{request.host_url}static/sop-files/{filename}"
               })
       
       # Sort by upload date
       company_docs.sort(key=lambda x: x.get('uploaded_at', ''), reverse=True)
       
       return jsonify(company_docs)
       
   except Exception as e:
       logger.error(f"Error fetching company docs: {e}")
       return jsonify({'error': 'Failed to fetch documents'}), 500

@app.route('/static/sop-files/<path:filename>')
def serve_sop_file(filename):
   """Serve SOP files"""
   try:
       safe_filename = secure_filename(filename)
       return send_from_directory(config.SOP_FOLDER, safe_filename)
   except Exception as e:
       logger.error(f"Error serving file: {e}")
       return jsonify({'error': 'File not found'}), 404

@app.route('/continue', methods=['POST'])
@validate_company_id
def continue_conversation():
   """Continue from previous response"""
   try:
       data = request.get_json()
       if not data:
           return jsonify({'error': 'Invalid request format'}), 400
       
       session_id = data.get('session_id')
       company_id = data.get('company_id_slug')
       
       if not session_id:
           return jsonify({'error': 'Session ID required'}), 400
       
       # Get conversation memory
       memory = session_manager.get_conversation_memory(session_id)
       
       # Check if there's a previous conversation
       if not memory.chat_memory.messages:
           return jsonify({
               'answer': "I don't see a previous conversation to continue from. Please ask a new question.",
               'session_id': session_id,
               'source': 'system'
           })
       
       # Generate continuation query
       continuation_query = "Please continue with more details from where you left off."
       
       # Process as a regular query
       request.json = {
           'query': continuation_query,
           'company_id_slug': company_id,
           'session_id': session_id
       }
       
       return query_sop()
       
   except Exception as e:
       logger.error(f"Continue error: {e}")
       return jsonify({'error': 'Failed to continue conversation'}), 500

@app.route('/clear-session', methods=['POST'])
def clear_session():
   """Clear conversation session"""
   try:
       data = request.get_json()
       session_id = data.get('session_id') if data else None
       
       if session_id:
           session_manager.set_session(session_id, {})
           return jsonify({'message': 'Session cleared'})
       
       return jsonify({'error': 'Session ID required'}), 400
       
   except Exception as e:
       logger.error(f"Clear session error: {e}")
       return jsonify({'error': 'Failed to clear session'}), 500

@app.route('/metrics')
@require_api_key
def get_metrics():
   """Get performance metrics (protected)"""
   if not config.ENABLE_METRICS:
       return jsonify({'error': 'Metrics disabled'}), 404
   
   return jsonify({
       'performance': performance_monitor.get_metrics(),
       'cache': {
           'hits': cache_manager.hits,
           'misses': cache_manager.misses,
           'hit_rate': round((cache_manager.hits / max(1, cache_manager.hits + cache_manager.misses)) * 100, 2)
       },
       'sessions': {
           'active': len(session_manager.memory_store) if not session_manager.use_redis else 'N/A'
       }
   })

@app.route('/sop-status')
def sop_status():
   """Get document processing status"""
   if os.path.exists(config.STATUS_FILE):
       return send_file(config.STATUS_FILE)
   return jsonify({})

@app.route('/reload-db', methods=['POST'])
@require_api_key
def reload_db():
   """Reload the vector database"""
   vector_store_manager._load_vectorstore()
   return jsonify({'message': 'Vectorstore reloaded successfully'})

@app.route('/clear-cache', methods=['POST'])
@require_api_key
def clear_cache():
   """Clear all caches"""
   try:
       # Clear query cache
       cache_size = len(cache_manager.cache)
       cache_manager.cache.clear()
       cache_manager.hits = 0
       cache_manager.misses = 0
       
       # Clear audio cache
       audio_files_cleared = 0
       if os.path.exists(config.AUDIO_CACHE_DIR):
           for filename in os.listdir(config.AUDIO_CACHE_DIR):
               if filename.endswith('.mp3'):
                   try:
                       os.remove(os.path.join(config.AUDIO_CACHE_DIR, filename))
                       audio_files_cleared += 1
                   except:
                       pass
       
       return jsonify({
           'message': 'Cache cleared successfully',
           'query_cache_cleared': cache_size,
           'audio_files_cleared': audio_files_cleared
       })
   except Exception as e:
       return jsonify({'error': str(e)}), 500

@app.route('/clear-sessions', methods=['POST'])
@require_api_key
def clear_all_sessions():
   """Clear all conversation sessions"""
   session_count = len(session_manager.memory_store) if not session_manager.use_redis else 0
   session_manager.memory_store.clear()
   return jsonify({
       'message': f'Cleared {session_count} conversation sessions'
   })

# ==== Background Tasks ====
def cleanup_old_data():
   """Periodic cleanup of old data"""
   try:
       # Clean old audio cache
       audio_service.cleanup_old_cache()
       
       # Clean expired sessions (if using memory store)
       session_manager.cleanup_expired()
       
       logger.info("Cleanup completed successfully")
   except Exception as e:
       logger.error(f"Cleanup error: {e}")

# ==== Error Handlers ====
@app.errorhandler(413)
def request_entity_too_large(e):
   return jsonify({'error': 'File too large. Maximum size is 10MB'}), 413

@app.errorhandler(429)
def ratelimit_handler(e):
   return jsonify({
       'error': 'Rate limit exceeded. Please try again later.',
       'retry_after': e.description
   }), 429

@app.errorhandler(500)
def internal_error(e):
   logger.error(f"Internal error: {e}")
   return jsonify({'error': 'Internal server error'}), 500

# ==== Startup ====
if __name__ == '__main__':
   # Ensure we don't lose existing data
   logger.info("Starting OpsVoice API...")
   logger.info(f"Data path: {config.DATA_PATH}")
   logger.info(f"Existing vectorstore: {os.path.exists(config.CHROMA_DIR)}")
   logger.info(f"Existing SOP files: {len(glob.glob(os.path.join(config.SOP_FOLDER, '*')))}")
   
   # Run cleanup on startup
   cleanup_old_data()
   
   # Schedule periodic cleanup (every hour)
   from threading import Timer
   def run_cleanup():
       cleanup_old_data()
       Timer(3600, run_cleanup).start()
   
   Timer(3600, run_cleanup).start()
   
   # Start Flask app
   port = int(os.environ.get('PORT', 10000))
   debug = os.environ.get('FLASK_ENV') == 'development'
   
   logger.info(f"Starting OpsVoice API on port {port}")
   logger.info(f"Redis sessions: {config.USE_REDIS_SESSIONS}")
   logger.info(f"Demo company: demo-business-123 (enhanced limits)")
   
   app.run(
       host='0.0.0.0',
       port=port,
       debug=debug,
       threaded=True
   )