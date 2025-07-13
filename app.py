import os
import json
import time
import asyncio
import secrets
import hashlib
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from functools import lru_cache, wraps
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import redis
from dataclasses import dataclass, asdict

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_caching import Cache
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

import aiohttp
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
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 30
    RATE_LIMIT_PER_HOUR: int = 500
    
    # Redis (optional)
    REDIS_URL: str = os.getenv("REDIS_URL")
    
    # Feature Flags
    USE_REDIS_SESSIONS: bool = bool(os.getenv("REDIS_URL"))
    USE_ASYNC_TTS: bool = True
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
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
        else:
            self.memory_store = {}
    
    def get_session(self, session_id: str) -> Dict:
        """Get session data"""
        if self.use_redis:
            data = self.redis_client.get(f"session:{session_id}")
            return json.loads(data) if data else {}
        return self.memory_store.get(session_id, {})
    
    def set_session(self, session_id: str, data: Dict, ttl: int = 3600):
        """Set session data with TTL"""
        if self.use_redis:
            self.redis_client.setex(
                f"session:{session_id}",
                ttl,
                json.dumps(data)
            )
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
        session_data['chat_history'] = messages
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
        self.lock = asyncio.Lock() if asyncio else None
    
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
    
    def delete_company_documents(self, company_id: str):
        """Delete all documents for a company"""
        # Note: ChromaDB doesn't support deletion by metadata filter directly
        # This would need to be implemented based on your specific needs
        pass

# ==== Audio Service ====
class AudioService:
    def __init__(self, api_key: str, cache_dir: str):
        self.api_key = api_key
        self.cache_dir = cache_dir
        self.voice_id = "tnSpp4vdxKPjI9w0GnoV"  # Default voice
    
    async def generate_audio_async(self, text: str, cache_key: str) -> bytes:
        """Generate audio asynchronously"""
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.mp3")
        
        # Check cache first
        if os.path.exists(cache_path):
            # Check if not too old
            if time.time() - os.path.getmtime(cache_path) < config.AUDIO_CACHE_TTL:
                with open(cache_path, 'rb') as f:
                    return f.read()
        
        # Generate new audio
        async with aiohttp.ClientSession() as session:
            async with session.post(
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
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    audio_data = await response.read()
                    
                    # Cache the audio
                    with open(cache_path, 'wb') as f:
                        f.write(audio_data)
                    
                    return audio_data
                else:
                    raise Exception(f"TTS API error: {response.status}")
    
    def cleanup_old_cache(self):
        """Remove old cache files"""
        cutoff_time = time.time() - config.AUDIO_CACHE_TTL
        for filename in os.listdir(self.cache_dir):
            filepath = os.path.join(self.cache_dir, filename)
            if os.path.getmtime(filepath) < cutoff_time:
                try:
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

def limiter_key_func():
    # If POST/JSON, check company_id_slug
    try:
        if request.is_json:
            company_id = (request.get_json() or {}).get('company_id_slug', '')
        elif request.form:
            company_id = request.form.get('company_id_slug', '')
        else:
            company_id = request.args.get('company_id_slug', '')
    except Exception:
        company_id = ''
    # Unlimited for demo
    if company_id == 'demo-business-123':
        return f"unlimited-demo"
    # Rate limit all others
    return f"{get_remote_address()}:{company_id}"

limiter = Limiter(
    app=app,
    key_func=limiter_key_func,    # <--- This line!
    storage_uri=config.REDIS_URL if config.REDIS_URL else "memory://",
    default_limits=[
        f"{config.RATE_LIMIT_PER_MINUTE} per minute",
        f"{config.RATE_LIMIT_PER_HOUR} per hour"
    ]
)

# Flask-Caching for additional caching
cache = Cache(app, config={
    'CACHE_TYPE': 'redis' if config.REDIS_URL else 'simple',
    'CACHE_REDIS_URL': config.REDIS_URL,
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
            request.json.get('company_id_slug') if request.json else None or
            request.form.get('company_id_slug') or
            request.args.get('company_id_slug')
        )
        
        if not company_id or not company_id.replace('-', '').replace('_', '').isalnum():
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
    import re
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
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'services': {
            'vectorstore': vector_store_manager.vectorstore is not None,
            'redis': session_manager.use_redis,
            'cache_hits': cache_manager.hits,
            'cache_misses': cache_manager.misses
        },
        'metrics': performance_monitor.get_metrics() if config.ENABLE_METRICS else None
    })

@app.route('/upload-sop', methods=['POST'])
@limiter.limit("10 per hour")
@validate_company_id
def upload_sop():
    """Upload and process SOP document"""
    try:
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
        company_id = request.form.get('company_id_slug', '').strip()
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
            **metadata
        }), 201
        
    except RequestEntityTooLarge:
        return jsonify({'error': 'File too large. Maximum size is 10MB'}), 413
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({'error': 'Upload failed'}), 500

@app.route('/query', methods=['POST'])
@limiter.limit("30 per minute")
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
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        if len(query) > config.MAX_QUERY_LENGTH:
            return jsonify({'error': f'Query too long. Maximum {config.MAX_QUERY_LENGTH} characters'}