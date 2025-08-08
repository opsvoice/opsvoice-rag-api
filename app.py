from flask import Flask, request, jsonify, send_from_directory, send_file, make_response
import os, glob, json, re, time, io, shutil, requests, hashlib, traceback, secrets
from dotenv import load_dotenv
from threading import Thread, Timer
from functools import lru_cache, wraps
from collections import OrderedDict
from langchain_community.document_loaders import UnstructuredWordDocumentLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from datetime import datetime, timedelta
import logging

load_dotenv()

# ==== LOGGING SETUP ====
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==== CONFIGURATION ====
if os.path.exists("/data"):
    DATA_PATH = "/data"
else:
    DATA_PATH = os.path.join(os.getcwd(), "data")

SOP_FOLDER = os.path.join(DATA_PATH, "sop-files")
CHROMA_DIR = os.path.join(DATA_PATH, "chroma_db")
AUDIO_CACHE_DIR = os.path.join(DATA_PATH, "audio_cache")
STATUS_FILE = os.path.join(SOP_FOLDER, "status.json")
METRICS_FILE = os.path.join(DATA_PATH, "metrics.json")

os.makedirs(SOP_FOLDER, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)
os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)

# ==== CONSTANTS ====
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_CACHE_SIZE = 1000
QUERY_CACHE_TTL = 3600  # 1 hour
AUDIO_CACHE_TTL = 86400  # 24 hours
SESSION_TTL = 7200  # 2 hours

# Rate limits - VERY RELAXED for demo, stricter for production
DEMO_RATE_LIMITS = {
    'queries_per_minute': 200,
    'queries_per_hour': 5000,
    'uploads_per_hour': 100,
    'max_documents': 50
}

PRODUCTION_RATE_LIMITS = {
    'queries_per_minute': 30,
    'queries_per_hour': 500,
    'uploads_per_hour': 20,
    'max_documents': 10
}

# ==== GLOBAL STATE ====
query_cache = OrderedDict()
conversation_sessions = {}
rate_limit_tracker = {}

performance_metrics = {
    "total_queries": 0,
    "cache_hits": 0,
    "avg_response_time": 0,
    "response_times": [],
    "model_usage": {"gpt-3.5-turbo": 0, "gpt-4": 0},
    "response_sources": {"sop": 0, "fallback": 0, "cache": 0, "error": 0, "business": 0},
    "companies": {},
    "daily_stats": {},
    "error_count": 0
}

embedding = OpenAIEmbeddings()
vectorstore = None

# ==== FLASK APP SETUP ====
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', secrets.token_hex(32))
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Ensure app is properly initialized before any route decorators
print(f"Flask app initialized: {app}")
print(f"App name: {app.name}")
print(f"Debug mode: {app.debug}")

# Enhanced CORS Configuration
ALLOWED_ORIGINS = [
    "https://opsvoice-widget.vercel.app",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://*.vercel.app",  # Allow all Vercel deployments
]

CORS(app, resources={
    r"/*": {
        "origins": ALLOWED_ORIGINS,
        "supports_credentials": True,
        "allow_headers": [
            "Content-Type", "Authorization", "X-Requested-With", 
            "Accept", "Origin", "X-API-Key", "Cache-Control"
        ],
        "methods": ["GET", "POST", "OPTIONS", "PUT", "DELETE"],
        "expose_headers": ["Content-Range", "X-Content-Range"]
    }
})

# CRITICAL CORS FIX - Handle all preflight requests
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = make_response()
        origin = request.headers.get('Origin')
        
        # Check if origin is allowed
        if origin:
            if any(allowed in origin for allowed in ALLOWED_ORIGINS) or origin in ALLOWED_ORIGINS:
                response.headers['Access-Control-Allow-Origin'] = origin
            else:
                response.headers['Access-Control-Allow-Origin'] = '*'  # Fallback for development
        else:
            response.headers['Access-Control-Allow-Origin'] = '*'
            
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS, PUT, DELETE'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With, Accept, Origin, X-API-Key, Cache-Control'
        response.headers['Access-Control-Allow-Credentials'] = 'true'
        response.headers['Access-Control-Max-Age'] = '3600'
        response.status_code = 200
        return response

@app.after_request
def after_request(response):
    origin = request.headers.get('Origin')
    
    if origin:
        if any(allowed in origin for allowed in ALLOWED_ORIGINS) or origin in ALLOWED_ORIGINS:
            response.headers['Access-Control-Allow-Origin'] = origin
        else:
            response.headers['Access-Control-Allow-Origin'] = '*'
    else:
        response.headers['Access-Control-Allow-Origin'] = '*'
    
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS, PUT, DELETE'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With, Accept, Origin, X-API-Key, Cache-Control'
    
    # Security headers
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    
    return response

# ==== RATE LIMITING SYSTEM ====
def get_rate_limits(company_id: str) -> dict:
    """Get rate limits based on company type"""
    if company_id == 'demo-business-123':
        return DEMO_RATE_LIMITS
    return PRODUCTION_RATE_LIMITS

def check_rate_limit(company_id: str, endpoint: str) -> tuple:
    """Check if request is within rate limits"""
    current_minute = int(time.time() // 60)
    current_hour = int(time.time() // 3600)
    current_day = int(time.time() // 86400)
    
    limits = get_rate_limits(company_id)
    
    # Initialize tracking
    if company_id not in rate_limit_tracker:
        rate_limit_tracker[company_id] = {}
    
    tracker = rate_limit_tracker[company_id]
    
    # Clean old entries
    tracker = {k: v for k, v in tracker.items() 
               if int(k.split('_')[-1]) > current_minute - 60}
    rate_limit_tracker[company_id] = tracker
    
    # Check limits based on endpoint
    if endpoint == 'query':
        minute_key = f"query_minute_{current_minute}"
        hour_key = f"query_hour_{current_hour}"
        
        minute_count = tracker.get(minute_key, 0)
        hour_count = tracker.get(hour_key, 0)
        
        if minute_count >= limits['queries_per_minute']:
            return False, f"Rate limit exceeded: {limits['queries_per_minute']} queries per minute"
        
        if hour_count >= limits['queries_per_hour']:
            return False, f"Rate limit exceeded: {limits['queries_per_hour']} queries per hour"
        
        # Increment counters
        tracker[minute_key] = minute_count + 1
        tracker[hour_key] = hour_count + 1
        
    elif endpoint == 'upload':
        hour_key = f"upload_hour_{current_hour}"
        hour_count = tracker.get(hour_key, 0)
        
        if hour_count >= limits['uploads_per_hour']:
            return False, f"Upload limit exceeded: {limits['uploads_per_hour']} uploads per hour"
        
        tracker[hour_key] = hour_count + 1
    
    return True, "OK"

# ==== SECURITY & VALIDATION ====
def validate_company_id(company_id: str) -> bool:
    """Validate company ID format and security"""
    if not company_id or len(company_id) < 3 or len(company_id) > 50:
        return False
    
    # Allow demo-business-123 specifically (CRITICAL FIX)
    if company_id == "demo-business-123":
        return True
    
    # Allow kenco-970 specifically (if you're testing that too)
    if company_id == "kenco-970":
        return True
    
    # Allow alphanumeric, hyphens, underscores only
    if not re.match(r'^[a-zA-Z0-9_-]+$', company_id):
        return False
    
    # Prevent path traversal and injection
    dangerous_patterns = ['..', '/', '\\', '<', '>', '"', "'", '&', '|', ';', '$']
    if any(pattern in company_id for pattern in dangerous_patterns):
        return False
    
    return True

def debug_company_validation(company_id: str) -> dict:
    """Debug company ID validation"""
    return {
        "company_id": company_id,
        "length": len(company_id) if company_id else 0,
        "is_demo": company_id == "demo-business-123",
        "is_kenco": company_id == "kenco-970",
        "regex_match": bool(re.match(r'^[a-zA-Z0-9_-]+$', company_id)) if company_id else False,
        "dangerous_patterns": [p for p in ['..', '/', '\\', '<', '>', '"', "'", '&', '|', ';', '$'] if p in (company_id or '')],
        "final_validation": validate_company_id(company_id) if company_id else False
    }

def validate_session_id(session_id: str) -> str:
    """Validate and sanitize session ID"""
    if not session_id:
        return ""
    
    # Remove dangerous characters
    clean_session = re.sub(r'[^a-zA-Z0-9_-]', '', session_id)
    
    # Limit length
    return clean_session[:64]

def sanitize_text_input(text: str, max_length: int = 500) -> str:
    """Sanitize text input for security"""
    if not text:
        return ""
    
    # Remove dangerous characters
    text = re.sub(r'[<>"\'\x00\r\n\t]', '', text)
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    # Limit length
    return text[:max_length].strip()

def validate_file_upload(file) -> tuple:
    """Comprehensive file validation"""
    if not file or not file.filename:
        return False, "No file uploaded"
    
    # Secure filename
    filename = secure_filename(file.filename)
    if not filename:
        return False, "Invalid filename"
    
    # Check extension
    if '.' not in filename:
        return False, "File must have extension"
    
    ext = filename.rsplit('.', 1)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Only {', '.join(ALLOWED_EXTENSIONS).upper()} files allowed"
    
    # Check file size
    file.seek(0, 2)
    size = file.tell()
    file.seek(0)
    
    if size > MAX_FILE_SIZE:
        return False, f"File too large (max {MAX_FILE_SIZE // 1024 // 1024}MB)"
    
    if size < 100:
        return False, "File appears to be empty or corrupted"
    
    return True, filename

# ==== ENVIRONMENT VALIDATION ====
def validate_environment():
    """Validate required environment variables"""
    required_vars = {
        'OPENAI_API_KEY': 'OpenAI API key for GPT models',
        'ELEVENLABS_API_KEY': 'ElevenLabs API key for TTS'
    }
    
    missing_vars = []
    for var, description in required_vars.items():
        if not os.getenv(var):
            missing_vars.append(f"{var} ({description})")
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        return False
    
    logger.info("All required environment variables are configured")
    return True

# ==== DEMO DOCUMENT LOADING ====
def load_demo_documents():
    """Load demo documents for demo-business-123"""
    demo_company_id = "demo-business-123"
    demo_sops_dir = os.path.join(os.path.dirname(__file__), "demo_sops")
    
    if not os.path.exists(demo_sops_dir):
        logger.warning(f"Demo SOPs directory not found: {demo_sops_dir}")
        return False
    
    try:
        # Check if demo documents are already loaded
        existing_docs = get_company_documents_internal(demo_company_id)
        if len(existing_docs) >= 5:
            logger.info(f"Demo documents already loaded for {demo_company_id}: {len(existing_docs)} documents")
            return True
        
        # Load demo documents
        demo_files = [
            "customer_service_procedures.pdf",
            "daily_operations_procedures.pdf", 
            "emergency_procedures_manual.pdf",
            "employee_procedures_manual.pdf",
            "onboarding_training_manual.pdf"
        ]
        
        loaded_count = 0
        for filename in demo_files:
            source_path = os.path.join(demo_sops_dir, filename)
            if not os.path.exists(source_path):
                logger.warning(f"Demo file not found: {source_path}")
                continue
            
            # Check if already processed
            timestamp = int(time.time())
            safe_filename = f"{demo_company_id}_{timestamp}_{filename}"
            dest_path = os.path.join(SOP_FOLDER, safe_filename)
            
            # Copy file if not exists
            if not os.path.exists(dest_path):
                shutil.copy2(source_path, dest_path)
                logger.info(f"Copied demo document: {filename}")
            
            # Prepare metadata
            metadata = {
                "title": filename.replace('.pdf', '').replace('_', ' ').title(),
                "company_id_slug": demo_company_id,
                "filename": safe_filename,
                "original_filename": filename,
                "uploaded_at": time.time(),
                "file_size": os.path.getsize(dest_path),
                "file_extension": "pdf",
                "is_demo_document": True
            }
            
            # Update status and process
            update_status(safe_filename, {"status": "processing", **metadata})
            Thread(target=embed_sop_worker, args=(dest_path, metadata), daemon=True).start()
            loaded_count += 1
        
        logger.info(f"Loaded {loaded_count} demo documents for {demo_company_id}")
        return loaded_count > 0
        
    except Exception as e:
        logger.error(f"Error loading demo documents: {e}")
        return False

# ==== AUDIO TRANSCRIPTION ====
def transcribe_audio(audio_data):
    """Transcribe audio using OpenAI Whisper"""
    try:
        # Use OpenAI Whisper API for transcription
        headers = {
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
        }
        
        files = {
            'file': ('audio.wav', audio_data, 'audio/wav'),
            'model': (None, 'whisper-1'),
            'language': (None, 'en')
        }
        
        response = requests.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers=headers,
            files=files,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get('text', '').strip()
        else:
            logger.error(f"Transcription failed: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        logger.error(f"Audio transcription error: {e}")
        return None

# ==== UTILITY FUNCTIONS ====
def clean_text(txt: str) -> str:
    """Clean and normalize text"""
    if not txt:
        return ""
    
    # Remove problematic characters
    txt = txt.replace('\u2022', '-').replace('\t', ' ')
    txt = re.sub(r'\s+', ' ', txt)
    txt = re.sub(r'[*#]+', '', txt)
    txt = re.sub(r'\[.*?\]', '', txt)
    
    return txt.strip()

def get_query_complexity(query: str) -> str:
    """Determine query complexity for optimal model selection"""
    words = query.lower().split()
    word_count = len(words)
    
    # Simple query indicators
    simple_indicators = [
        word_count <= 8,
        any(word in query.lower() for word in ['what', 'when', 'where', 'who', 'how many']),
        query.endswith('?') and word_count <= 6,
        any(word in query.lower() for word in ['yes', 'no', 'help', 'hi', 'hello'])
    ]
    
    # Complex query indicators
    complex_indicators = [
        word_count > 15,
        any(word in query.lower() for word in [
            'analyze', 'compare', 'explain why', 'walk me through', 'break down',
            'evaluate', 'assess', 'comprehensive', 'detailed', 'because', 'therefore'
        ]),
        query.count('?') > 1,
        any(word in query.lower() for word in ['however', 'although', 'nevertheless'])
    ]
    
    if sum(complex_indicators) > 0:
        return "complex"
    elif sum(simple_indicators) >= 2:
        return "simple"
    else:
        return "medium"

def get_optimal_llm(complexity: str) -> ChatOpenAI:
    """Select optimal LLM based on query complexity"""
    performance_metrics["model_usage"][
        "gpt-3.5-turbo" if complexity == "simple" else "gpt-4"
    ] += 1
    
    if complexity == "simple":
        return ChatOpenAI(
            temperature=0, 
            model="gpt-3.5-turbo", 
            request_timeout=30,
            max_retries=2
        )
    else:
        return ChatOpenAI(
            temperature=0, 
            model="gpt-4", 
            request_timeout=60,
            max_retries=2
        )

def smart_truncate(text: str, max_words: int = 150) -> str:
    """Smart truncation that preserves meaning"""
    if not text:
        return text
    
    words = text.split()
    if len(words) <= max_words:
        return text
    
    # Find last complete sentence within limit
    truncated = " ".join(words[:max_words])
    
    # Look for sentence endings
    sentence_endings = ['.', '!', '?', ':']
    last_ending = -1
    
    for ending in sentence_endings:
        pos = truncated.rfind(ending)
        if pos > last_ending:
            last_ending = pos
    
    # If found good sentence break (not too short)
    if last_ending > len(truncated) * 0.6:
        return truncated[:last_ending + 1] + " Would you like me to continue with more details?"
    else:
        # Fallback to word truncation
        return " ".join(words[:130]) + "... Should I continue with the rest of the information?"

# ==== CACHING SYSTEM ====
def get_cache_key(query: str, company_id: str) -> str:
    """Generate cache key for query"""
    combined = f"{company_id}:{query.lower().strip()}"
    return hashlib.sha256(combined.encode()).hexdigest()

def get_cached_response(query: str, company_id: str) -> dict:
    """Get cached response if available"""
    cache_key = get_cache_key(query, company_id)
    cached = query_cache.get(cache_key)
    
    if cached and time.time() - cached['timestamp'] < QUERY_CACHE_TTL:
        # Move to end (LRU)
        query_cache.move_to_end(cache_key)
        performance_metrics["cache_hits"] += 1
        performance_metrics["response_sources"]["cache"] += 1
        return cached['response']
    
    # Remove expired entry
    if cached:
        del query_cache[cache_key]
    
    return None

def cache_response(query: str, company_id: str, response: dict):
    """Cache response with LRU eviction"""
    cache_key = get_cache_key(query, company_id)
    
    query_cache[cache_key] = {
        'response': response,
        'timestamp': time.time()
    }
    
    # LRU eviction
    while len(query_cache) > MAX_CACHE_SIZE:
        query_cache.popitem(last=False)

# ==== SESSION MANAGEMENT ====
def get_session_memory(session_id: str) -> ConversationBufferMemory:
    """Get or create conversation memory for session"""
    if session_id not in conversation_sessions:
        conversation_sessions[session_id] = {
            'memory': ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            ),
            'created_at': time.time(),
            'last_accessed': time.time()
        }
    
    # Update access time
    conversation_sessions[session_id]['last_accessed'] = time.time()
    
    return conversation_sessions[session_id]['memory']

def cleanup_expired_sessions():
    """Clean up expired conversation sessions"""
    current_time = time.time()
    expired_sessions = [
        session_id for session_id, data in conversation_sessions.items()
        if current_time - data['last_accessed'] > SESSION_TTL
    ]
    
    for session_id in expired_sessions:
        del conversation_sessions[session_id]
    
    logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")

# ==== PERFORMANCE MONITORING ====
def update_metrics(response_time: float, source: str, company_id: str = None):
    """Update performance metrics"""
    performance_metrics["total_queries"] += 1
    performance_metrics["response_sources"][source] = performance_metrics["response_sources"].get(source, 0) + 1
    
    # Update response times (keep last 1000)
    performance_metrics["response_times"].append(response_time)
    if len(performance_metrics["response_times"]) > 1000:
        performance_metrics["response_times"] = performance_metrics["response_times"][-1000:]
    
    # Update average response time
    performance_metrics["avg_response_time"] = round(
        sum(performance_metrics["response_times"]) / len(performance_metrics["response_times"]), 3
    )
    
    # Track per-company metrics
    if company_id:
        if company_id not in performance_metrics["companies"]:
            performance_metrics["companies"][company_id] = {
                "queries": 0, "avg_response_time": 0, "sources": {}
            }
        
        company_metrics = performance_metrics["companies"][company_id]
        company_metrics["queries"] += 1
        company_metrics["sources"][source] = company_metrics["sources"].get(source, 0) + 1
        
        # Update company average response time
        current_avg = company_metrics["avg_response_time"]
        company_metrics["avg_response_time"] = round(
            ((current_avg * (company_metrics["queries"] - 1)) + response_time) / company_metrics["queries"], 3
        )
    
    # Save metrics periodically
    if performance_metrics["total_queries"] % 25 == 0:
        save_metrics()

def save_metrics():
    """Save metrics to file"""
    try:
        with open(METRICS_FILE, 'w') as f:
            json.dump(performance_metrics, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving metrics: {e}")

def load_metrics():
    """Load metrics from file"""
    try:
        if os.path.exists(METRICS_FILE):
            with open(METRICS_FILE, 'r') as f:
                saved_metrics = json.load(f)
                performance_metrics.update(saved_metrics)
                logger.info(f"Loaded existing metrics: {performance_metrics['total_queries']} total queries")
    except Exception as e:
        logger.error(f"Error loading metrics: {e}")

# ==== DOCUMENT PROCESSING ====
def update_status(filename: str, status: dict):
    """Update document processing status"""
    try:
        data = json.load(open(STATUS_FILE)) if os.path.exists(STATUS_FILE) else {}
        data[filename] = {**status, "updated_at": time.time()}
        with open(STATUS_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Error updating status for {filename}: {e}")

def embed_sop_worker(fpath: str, metadata: dict = None):
    """Background worker for document embedding"""
    fname = os.path.basename(fpath)
    try:
        ext = fname.rsplit(".", 1)[-1].lower()
        
        # Load document based on type
        if ext == "docx":
            docs = UnstructuredWordDocumentLoader(fpath).load()
        elif ext == "pdf":
            docs = PyPDFLoader(fpath).load()
        elif ext == "txt":
            with open(fpath, 'r', encoding='utf-8') as f:
                content = f.read()
            from langchain.schema import Document
            docs = [Document(page_content=content, metadata={"source": fpath})]
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        
        # Enhanced chunking strategy
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]
        ).split_documents(docs)
        
        # Add comprehensive metadata
        company_id_slug = metadata.get("company_id_slug") if metadata else None
        for i, chunk in enumerate(chunks):
            # Analyze chunk content for better categorization
            content_lower = chunk.page_content.lower()
            
            # Determine content type
            if any(word in content_lower for word in ["angry", "upset", "difficult", "complaint"]):
                chunk_type = "customer_service"
            elif any(word in content_lower for word in ["cash", "money", "payment", "refund"]):
                chunk_type = "financial"
            elif any(word in content_lower for word in ["first day", "onboard", "training"]):
                chunk_type = "onboarding"
            elif any(word in content_lower for word in ["emergency", "safety", "evacuation"]):
                chunk_type = "emergency"
            else:
                chunk_type = "general"
            
            # Extract keywords (as string to avoid ChromaDB issues)
            words = re.findall(r'\b[a-zA-Z]{4,}\b', content_lower)
            stopwords = {'that', 'this', 'with', 'they', 'have', 'will', 'from', 'been', 'were', 'what', 'when'}
            keywords = [word for word in set(words) if word not in stopwords][:8]
            keywords_string = " ".join(keywords)
            
            chunk.metadata.update({
                "company_id_slug": company_id_slug,
                "filename": fname,
                "chunk_id": f"{fname}_{i}",
                "chunk_index": i,
                "total_chunks": len(chunks),
                "chunk_type": chunk_type,
                "keywords": keywords_string,
                "source": fpath,
                "uploaded_at": metadata.get("uploaded_at", time.time()),
                "title": metadata.get("title", fname),
                "file_size": metadata.get("file_size", 0)
            })
        
        # Add to vector store
        db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding)
        db.add_documents(chunks)
        db.persist()
        
        logger.info(f"Successfully embedded {len(chunks)} chunks from {fname} for company {company_id_slug}")
        update_status(fname, {
            "status": "embedded",
            "chunk_count": len(chunks),
            **(metadata or {})
        })
        
    except Exception as e:
        logger.error(f"Error embedding {fname}: {traceback.format_exc()}")
        update_status(fname, {
            "status": f"error: {str(e)}",
            **(metadata or {})
        })

def load_vectorstore():
    """Load the vector database"""
    global vectorstore
    try:
        vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding)
        logger.info("Vector store loaded successfully")
    except Exception as e:
        logger.error(f"Error loading vector store: {e}")
        vectorstore = None

def ensure_vectorstore():
    """Ensure vectorstore is available and healthy"""
    global vectorstore
    try:
        if not vectorstore:
            load_vectorstore()
        
        # Health check
        if vectorstore and hasattr(vectorstore, '_collection'):
            test_results = vectorstore.similarity_search("test", k=1)
            logger.debug(f"Vectorstore healthy, {len(test_results)} test results")
        
        return vectorstore is not None
    except Exception as e:
        logger.error(f"Vectorstore health check failed: {e}")
        load_vectorstore()
        return vectorstore is not None

def get_company_documents_internal(company_id_slug: str) -> list:
    """Get documents for a specific company"""
    if not os.path.exists(STATUS_FILE):
        return []
    
    try:
        with open(STATUS_FILE, 'r') as f:
            data = json.load(f)
        
        company_docs = []
        for filename, metadata in data.items():
            if metadata.get("company_id_slug") == company_id_slug:
                safe_filename = secure_filename(filename)
                if safe_filename == filename:  # Ensure filename is safe
                    doc_info = {
                        "filename": safe_filename,
                        "title": metadata.get("title", safe_filename),
                        "status": metadata.get("status", "unknown"),
                        "company_id_slug": company_id_slug,
                        "uploaded_at": metadata.get("uploaded_at"),
                        "sop_file_url": f"{request.host_url}static/sop-files/{safe_filename}",
                        "file_size": metadata.get("file_size"),
                        "chunk_count": metadata.get("chunk_count")
                    }
                    company_docs.append(doc_info)
        
        return company_docs
    except Exception as e:
        logger.error(f"Error fetching docs for {company_id_slug}: {e}")
        return []

def get_document_count(company_id: str) -> int:
    """Get count of successfully embedded documents for a company"""
    if not os.path.exists(STATUS_FILE):
        return 0
    
    try:
        with open(STATUS_FILE, 'r') as f:
            data = json.load(f)
        
        count = sum(1 for doc in data.values() 
                   if doc.get('company_id_slug') == company_id 
                   and doc.get('status') == 'embedded')
        return count
    except:
        return 0

# ==== ENHANCED QUERY PROCESSING ====
def is_unhelpful_answer(text: str) -> bool:
    """Detect if answer is unhelpful or generic"""
    if not text or not text.strip():
        return True
    
    low = text.lower()
    
    # Definitive unhelpful phrases
    unhelpful_triggers = [
        "don't know", "no information", "i'm not sure", "sorry",
        "unavailable", "not covered", "cannot find", "no specific information",
        "not mentioned", "doesn't provide", "no details", "not included",
        "context provided does not include", "text does not provide",
        "i don't have access", "not available in the", "no relevant information"
    ]
    
    has_trigger = any(trigger in low for trigger in unhelpful_triggers)
    is_too_short = len(low.split()) < 10
    
    return has_trigger or is_too_short

def generate_contextual_followups(query: str, answer: str, previous_queries: list = None) -> list:
    """Generate smart contextual follow-up questions"""
    q = query.lower()
    a = answer.lower()
    base_followups = []
    
    # Answer-based followups
    if any(word in a for word in ["step", "procedure", "process"]):
        base_followups.append("Would you like the complete step-by-step procedure?")
    
    if any(word in a for word in ["policy", "rule", "requirement"]):
        base_followups.append("Are there any exceptions to this policy?")
    
    if any(word in a for word in ["form", "document", "paperwork"]):
        base_followups.append("Do you need help finding the actual forms?")
    
    # Query-based followups
    if any(word in q for word in ["employee", "staff", "worker"]):
        base_followups.append("Do you need information about employee procedures?")
    elif any(word in q for word in ["time", "schedule", "hours"]):
        base_followups.append("Would you like details about scheduling policies?")
    elif any(word in q for word in ["customer", "client"]):
        base_followups.append("Do you need customer service procedures?")
    
    # Context-aware followups based on previous queries
    if previous_queries:
        context_words = []
        for prev_q in previous_queries[-2:]:
            context_words.extend(prev_q.lower().split())
        
        if "training" in context_words and "procedure" in q:
            base_followups.append("Would you like the training checklist for this procedure?")
    
    # Default fallbacks
    if not base_followups:
        base_followups.extend([
            "Do you want to know more details?",
            "Would you like steps for a related task?",
            "Need help finding a specific document?"
        ])
    
    return base_followups[:3]

def is_vague_query(query: str) -> bool:
    """Enhanced vague query detection"""
    if not query or len(query.strip()) < 3:
        return True
    
    # Very short queries without context
    if len(query.split()) < 2:
        return True
    
    # Single word questions (except specific ones)
    words = query.lower().split()
    if len(words) == 1 and words[0] not in ['help', 'documents', 'procedures', 'policies']:
        return True
    
    # Greetings without questions
    greetings = ['hello', 'hi', 'hey', 'good morning', 'good afternoon']
    if any(greeting in query.lower() for greeting in greetings) and '?' not in query:
        return True
    
    return False

def expand_query_with_synonyms(query: str) -> str:
    """Expand query with synonyms for better matching"""
    synonyms = {
        "angry": "angry upset difficult frustrated mad",
        "customer": "customer client guest patron visitor",
        "handle": "handle deal manage respond address",
        "procedure": "procedure process protocol steps workflow",
        "policy": "policy rule guideline standard regulation",
        "refund": "refund return money back exchange",
        "cash": "cash money payment transaction financial",
        "first day": "first day onboarding orientation training new employee",
        "emergency": "emergency urgent crisis safety evacuation",
        "closing": "closing close end finish shutdown",
        "opening": "opening open start begin startup"
    }
    
    expanded = query
    for word, expansion in synonyms.items():
        if word in query.lower():
            expanded += f" {expansion}"
    
    return expanded

def generate_business_fallback(query: str, company_id: str) -> str:
    """Generate helpful business fallback when no documents found"""
    query_lower = query.lower()
    company_name = company_id.replace('-', ' ').title()
    
    # Customer service queries
    if any(word in query_lower for word in ['angry', 'upset', 'difficult', 'complaint', 'customer service']):
        return f"""I don't see specific customer service procedures in your documents for {company_name}, but here are proven best practices for handling difficult customers:

**Immediate Response:**
1. **Listen actively** - Let them express their full concern without interruption
2. **Stay calm and professional** - Don't take complaints personally
3. **Acknowledge their feelings** - "I understand this is frustrating"
4. **Apologize sincerely** - Even if it wasn't your fault directly

**Resolution Steps:**
5. **Ask clarifying questions** - Get all the details you need
6. **Offer solutions** - Focus on what you CAN do, not what you can't
7. **Follow up** - Ensure they're satisfied with the resolution
8. **Document the interaction** - For future reference and improvement

For your company's specific customer service procedures, please check with your manager or upload your customer service guidelines to this system."""

    # Financial/money queries
    elif any(word in query_lower for word in ['cash', 'money', 'refund', 'payment', 'financial']):
        return f"""I don't see specific financial procedures in your documents for {company_name}.

**General Financial Best Practices:**
- Always get manager approval for refunds over your authorization limit
- Document all financial transactions immediately
- Follow your company's cash handling procedures
- Never process payments without proper verification
- Keep receipts and records for all transactions

**Next Steps:**
1. Check with your supervisor for authorization limits
2. Review your employee handbook for financial policies  
3. Contact the accounting department for specific procedures
4. Upload your financial procedures to this system for easy access

Try asking about other topics from your uploaded company documents."""

    # Process/procedure queries
    elif any(word in query_lower for word in ['process', 'procedure', 'how to', 'steps', 'workflow']):
        return f"""I couldn't find that specific procedure in your uploaded documents for {company_name}.

**To get the right procedure:**
1. **Check if it's been uploaded** - Ask your manager if this procedure exists in writing
2. **Ask your supervisor** - They can walk you through the current process
3. **Look for related procedures** - The information might be in a different document
4. **Document it** - If no written procedure exists, consider creating one

**Available Resources:**
- Your employee handbook
- Department-specific procedures  
- Manager or team lead guidance
- Company intranet or documentation system

Would you like to know what documents are currently available in the system?"""

    # Policy queries
    elif any(word in query_lower for word in ['policy', 'rule', 'allowed', 'permitted', 'regulation']):
        return f"""I don't see that specific policy in your current documents for {company_name}.

**For accurate policy information:**
- **Check your employee handbook** - Most policies are documented there
- **Ask HR department** - They have the most current policy information
- **Consult your manager** - They can clarify policy applications
- **Review company intranet** - Policies may be posted online

**Important:** Company policies should always be in writing and accessible to all employees. If you can't find a written policy, ask your HR department to clarify and document it.

Try asking about other policies that might be in your uploaded documents."""

    # Emergency procedures
    elif any(word in query_lower for word in ['emergency', 'urgent', 'crisis', 'evacuation', 'safety']):
        return f"""I don't see specific emergency procedures in your documents for {company_name}.

**General Emergency Response:**
1. **Immediate safety first** - Remove yourself and others from danger
2. **Call emergency services** - 911 for life-threatening situations
3. **Follow evacuation procedures** - Use nearest safe exit
4. **Report to designated assembly point** - Wait for all-clear
5. **Notify management** - Inform supervisors as soon as it's safe

**Important:** Every workplace should have written emergency procedures. Please ask your manager for:
- Emergency evacuation plans
- Fire safety procedures  
- Medical emergency protocols
- Emergency contact information

This is critical safety information that should be documented and uploaded to this system."""

    # Training/onboarding queries
    elif any(word in query_lower for word in ['training', 'onboard', 'first day', 'new employee']):
        return f"""I don't see specific training procedures in your documents for {company_name}.

**Standard Onboarding Elements:**
1. **Welcome and introductions** - Meet the team
2. **Paperwork completion** - HR documents, tax forms
3. **System setup** - Computer access, email, passwords
4. **Training schedule** - Job-specific and safety training
5. **Mentor assignment** - Buddy system for first weeks
6. **Goal setting** - Clear expectations and objectives

**Next Steps:**
- Check with HR for the official onboarding checklist
- Ask your manager about department-specific training
- Review your employee handbook for training requirements

Upload your training materials to this system so all employees can access them easily."""

    # Default comprehensive fallback
    else:
        return f"""I couldn't find specific information about that in your documents for {company_name}.

**This might be because:**
- The relevant document hasn't been uploaded to the system yet
- The information is covered under a different topic or section
- This requires checking with your supervisor or manager
- It's a new situation that needs to be documented

**Recommended Actions:**
1. **Check with your manager** - They may have the information you need
2. **Review your employee handbook** - Many procedures are documented there
3. **Ask a experienced colleague** - They might know the informal process
4. **Search your company intranet** - Additional resources may be available online

**Available in System:**
Try asking about procedures, policies, or customer service topics that might be in your uploaded documents, or ask "What documents do you have?" to see what's available.

If this is a common question, consider uploading the relevant procedure to this system so everyone can access it easily."""

# ==== FLASK ROUTES ====
@app.route('/')
def home():
    """Enhanced home endpoint with comprehensive system info"""
    return jsonify({
        'status': 'healthy',
        'service': 'OpsVoice RAG API',
        'version': '3.1.0-production-ready',
        'timestamp': time.time(),
        'features': [
            'multi_tenant_rag',
            'conversational_memory', 
            'smart_model_selection',
            'intelligent_caching',
            'rate_limiting',
            'performance_monitoring',
            'secure_file_upload',
            'audio_tts_support',
            'business_intelligence_fallback',
            'cors_compliant',
            'session_management'
        ],
        'endpoints': {
            'health_check': '/healthz',
            'query': '/query',
            'upload': '/upload-sop',
            'voice_tts': '/voice-reply',
            'vapi_function': '/vapi-function',
            'voice_query': '/voice-query',
            'documents': '/company-docs/{company_id}',
            'metrics': '/metrics',
            'continue': '/continue'
        },
        'rate_limits': {
            'demo_company': DEMO_RATE_LIMITS,
            'production_company': PRODUCTION_RATE_LIMITS
        }
    })

@app.route('/healthz', methods=['GET', 'OPTIONS'])
def healthz():
    """Comprehensive health check endpoint"""
    if request.method == "OPTIONS":
        return "", 204
    
    # Check system components
    vectorstore_status = "loaded" if vectorstore else "not_loaded"
    
    # Count files and sessions
    total_files = len(glob.glob(os.path.join(SOP_FOLDER, "*.*")))
    active_sessions = len(conversation_sessions)
    
    # Database health check
    db_healthy = ensure_vectorstore()
    
    health_data = {
        "status": "healthy" if db_healthy else "degraded",
        "timestamp": time.time(),
        "version": "3.1.0",
        "system": {
            "vectorstore": vectorstore_status,
            "vectorstore_healthy": db_healthy,
            "data_path": DATA_PATH,
            "persistent_storage": os.path.exists("/data"),
            "cache_size": len(query_cache),
            "active_sessions": active_sessions,
            "total_files": total_files
        },
        "performance": {
            "avg_response_time": performance_metrics.get("avg_response_time", 0),
            "total_queries": performance_metrics.get("total_queries", 0),
            "cache_hit_rate": round(
                (performance_metrics.get("cache_hits", 0) / max(1, performance_metrics.get("total_queries", 1))) * 100, 2
            )
        },
        "rate_limiting": {
            "demo_limits": DEMO_RATE_LIMITS,
            "production_limits": PRODUCTION_RATE_LIMITS
        }
    }
    
    status_code = 200 if db_healthy else 503
    return jsonify(health_data), status_code

@app.route('/list-sops')
def list_sops():
    """List all uploaded SOP files with metadata"""
    try:
        docs = (glob.glob(os.path.join(SOP_FOLDER, "*.docx")) + 
                glob.glob(os.path.join(SOP_FOLDER, "*.pdf")) + 
                glob.glob(os.path.join(SOP_FOLDER, "*.txt")))
        
        file_list = []
        for doc_path in docs:
            filename = os.path.basename(doc_path)
            try:
                file_info = {
                    'filename': filename,
                    'size': os.path.getsize(doc_path),
                    'modified': os.path.getmtime(doc_path),
                    'extension': filename.rsplit('.', 1)[-1].lower() if '.' in filename else 'unknown'
                }
                file_list.append(file_info)
            except:
                continue
        
        # Sort by modification time (newest first)
        file_list.sort(key=lambda x: x['modified'], reverse=True)
        
        return jsonify({
            'files': file_list,
            'count': len(file_list),
            'total_size': sum(f['size'] for f in file_list),
            'by_extension': {
                ext: len([f for f in file_list if f['extension'] == ext])
                for ext in ['pdf', 'docx', 'txt']
            }
        })
    except Exception as e:
        logger.error(f"Error listing SOPs: {e}")
        return jsonify({'error': 'Failed to list documents'}), 500

@app.route('/static/sop-files/<path:filename>')
def serve_sop(filename):
    """Serve SOP files with security validation"""
    try:
        # Security validation
        safe_filename = secure_filename(filename)
        if safe_filename != filename:
            return jsonify({"error": "Invalid filename"}), 400
        
        file_path = os.path.join(SOP_FOLDER, safe_filename)
        if not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 404
        
        # Additional security: ensure file is within SOP folder
        if not os.path.abspath(file_path).startswith(os.path.abspath(SOP_FOLDER)):
            return jsonify({"error": "Access denied"}), 403
        
        return send_from_directory(SOP_FOLDER, safe_filename, as_attachment=False)
    except Exception as e:
        logger.error(f"File serve error: {e}")
        return jsonify({"error": "File serve failed"}), 500

@app.route('/upload-sop', methods=['POST', 'OPTIONS'])
def upload_sop():
    """Enhanced secure document upload with comprehensive validation"""
    if request.method == "OPTIONS":
        return "", 204
    
    start_time = time.time()
    
    try:
        # Get and validate company ID
        company_id = request.form.get('company_id_slug', '').strip()
        if not validate_company_id(company_id):
            return jsonify({"error": "Invalid company identifier"}), 400
        
        # Rate limiting check
        rate_ok, rate_msg = check_rate_limit(company_id, 'upload')
        if not rate_ok:
            return jsonify({"error": rate_msg}), 429
        
        # Document count limits
        current_count = get_document_count(company_id)
        limits = get_rate_limits(company_id)
        
        if current_count >= limits['max_documents']:
            return jsonify({
                "error": f"Document limit reached. Maximum {limits['max_documents']} documents allowed.",
                "current_count": current_count,
                "limit": limits['max_documents']
            }), 400
        
        # File validation
        file = request.files.get("file")
        is_valid, result = validate_file_upload(file)
        if not is_valid:
            return jsonify({"error": result}), 400
        
        filename = result
        
        # Get and sanitize metadata
        title = sanitize_text_input(request.form.get("doc_title", filename), 200)
        
        # Generate unique, secure filename
        timestamp = int(time.time())
        safe_filename = f"{company_id}_{timestamp}_{filename}"
        save_path = os.path.join(SOP_FOLDER, safe_filename)
        
        # Save file securely
        file.save(save_path)
        
        # Verify file was saved correctly
        if not os.path.exists(save_path) or os.path.getsize(save_path) == 0:
            if os.path.exists(save_path):
                os.remove(save_path)
            return jsonify({"error": "File save verification failed"}), 500
        
        # Prepare comprehensive metadata
        metadata = {
            "title": title,
            "company_id_slug": company_id,
            "filename": safe_filename,
            "original_filename": filename,
            "uploaded_at": time.time(),
            "file_size": os.path.getsize(save_path),
            "file_extension": filename.rsplit('.', 1)[-1].lower(),
            "upload_ip": request.environ.get('HTTP_X_FORWARDED_FOR', request.environ.get('REMOTE_ADDR')),
            "user_agent": request.headers.get('User-Agent', '')[:200]
        }
        
        # Update status and start background processing
        update_status(safe_filename, {"status": "processing", **metadata})
        Thread(target=embed_sop_worker, args=(save_path, metadata), daemon=True).start()
        
        # Clear cache for this company
        cache_keys_to_remove = [key for key in query_cache.keys() if company_id in key]
        for key in cache_keys_to_remove:
            del query_cache[key]
        
        logger.info(f"Document uploaded: {safe_filename} for company {company_id}")
        
        return jsonify({
            "message": "Document uploaded successfully and is being processed",
            "doc_id": safe_filename,
            "doc_title": title,
            "company_id_slug": company_id,
            "status": "processing",
            "sop_file_url": f"{request.host_url.rstrip('/')}/static/sop-files/{safe_filename}",
            "upload_id": secrets.token_hex(8),
            "estimated_processing_time": "1-3 minutes",
            **metadata
        }), 201
        
    except Exception as e:
        logger.error(f"Upload error: {traceback.format_exc()}")
        if 'save_path' in locals() and os.path.exists(save_path):
            try:
                os.remove(save_path)
            except:
                pass
        return jsonify({"error": "Upload failed", "details": "Please try again"}), 500

@app.route('/query', methods=['POST', 'OPTIONS'])
def query_sop():
    """Enhanced query processing with comprehensive features"""
    if request.method == "OPTIONS":
        return "", 204
    
    start_time = time.time()
    
    # Ensure vectorstore is ready
    if not ensure_vectorstore():
        update_metrics(time.time() - start_time, "error")
        return jsonify({"error": "Service temporarily unavailable"}), 503
    
    try:
        # Input validation and sanitization
        if not request.is_json:
            return jsonify({"error": "Invalid request format - JSON required"}), 400
        
        payload = request.get_json() or {}
        
        # Extract and validate inputs
        raw_query = payload.get("query", "")
        qtext = sanitize_text_input(raw_query, 500)
        company_id = payload.get("company_id_slug", "").strip()
        session_id = validate_session_id(payload.get("session_id", f"{company_id}_{int(time.time())}"))
        
        # Debug logging for company validation
        validation_debug = debug_company_validation(company_id)
        logger.info(f"Company validation debug: {validation_debug}")
        
        if not qtext:
            return jsonify({"error": "Query is required"}), 400
        
        if not validate_company_id(company_id):
            logger.error(f"Company ID validation failed for: {company_id} - Debug: {validation_debug}")
            return jsonify({
                "error": "Invalid company identifier", 
                "debug": validation_debug if os.getenv('FLASK_ENV') == 'development' else None
            }), 400
        
        # Rate limiting
        rate_ok, rate_msg = check_rate_limit(company_id, 'query')
        if not rate_ok:
            return jsonify({"error": rate_msg}), 429
        
        # Check cache first
        cached_response = get_cached_response(qtext, company_id)
        if cached_response:
            cached_response.update({
                "cache_hit": True,
                "response_time": time.time() - start_time,
                "session_id": session_id
            })
            update_metrics(time.time() - start_time, "cache", company_id)
            return jsonify(cached_response)
        
        # Handle document listing queries
        doc_keywords = [
            'what documents', 'what files', 'what sops', 'uploaded documents', 
            'what do you have', 'what can you help', 'available documents',
            'list documents', 'show documents'
        ]
        if any(keyword in qtext.lower() for keyword in doc_keywords):
            docs = get_company_documents_internal(company_id)
            if docs:
                doc_titles = []
                for doc in docs:
                    title = doc.get('title', doc.get('filename', 'Unknown Document'))
                    # Clean up title
                    if title.endswith(('.docx', '.pdf', '.txt')):
                        title = title.rsplit('.', 1)[0]
                    doc_titles.append(title)
                
                response = {
                    "answer": f"I have access to {len(doc_titles)} documents for your company: {', '.join(doc_titles)}. I can answer questions about any of these procedures, policies, and processes.",
                    "source": "document_list",
                    "document_count": len(docs),
                    "followups": [
                        "Would you like details about any specific procedure?",
                        "Do you need help with a particular process?",
                        "What specific information are you looking for?"
                    ],
                    "session_id": session_id,
                    "response_time": time.time() - start_time
                }
                cache_response(qtext, company_id, response)
                update_metrics(time.time() - start_time, "document_list", company_id)
                return jsonify(response)
            else:
                response = {
                    "answer": "No documents have been uploaded for your company yet. Please upload your SOPs, procedures, and policies to get started.",
                    "source": "no_documents",
                    "followups": [
                        "Upload your company documents",
                        "Contact your manager about documentation",
                        "Check if documents are being processed"
                    ],
                    "session_id": session_id
                }
                update_metrics(time.time() - start_time, "no_documents", company_id)
                return jsonify(response)
        
        # Check for vague queries
        if is_vague_query(qtext):
            response = {
                "answer": "Could you be more specific? For example, ask about a particular procedure, policy, or process you need help with.",
                "source": "clarification_needed",
                "followups": generate_contextual_followups(qtext, ""),
                "session_id": session_id,
                "response_time": time.time() - start_time
            }
            update_metrics(time.time() - start_time, "clarification", company_id)
            return jsonify(response)
        
        # Filter out off-topic queries
        off_topic_keywords = [
            "weather", "news", "stock price", "sports", "celebrity", "movie", 
            "restaurant", "recipe", "personal", "family", "dating"
        ]
        if any(keyword in qtext.lower() for keyword in off_topic_keywords):
            response = {
                "answer": "I'm focused on helping with your business procedures and operations. Please ask about your company's SOPs, policies, customer service, or general business questions.",
                "source": "off_topic",
                "followups": [
                    "Ask about company procedures",
                    "Inquire about policies",
                    "Get help with customer service"
                ],
                "session_id": session_id
            }
            update_metrics(time.time() - start_time, "off_topic", company_id)
            return jsonify(response)
        
        # Determine optimal model and processing approach
        complexity = get_query_complexity(qtext)
        optimal_llm = get_optimal_llm(complexity)
        
        logger.info(f"Processing {complexity} query with {optimal_llm.model_name} for {company_id}: {qtext[:50]}...")
        
        # Enhanced query expansion
        expanded_query = expand_query_with_synonyms(qtext)
        
        # Set up retriever with company filtering
        retriever = vectorstore.as_retriever(
            search_kwargs={
                "k": 7,  # Retrieve more documents for better context
                "filter": {"company_id_slug": company_id}
            }
        )
        
        # Get or create session memory
        memory = get_session_memory(session_id)
        
        # Create conversational QA chain
        qa = ConversationalRetrievalChain.from_llm(
            optimal_llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            verbose=False
        )
        
        # Execute query
        result = qa.invoke({"question": expanded_query})
        answer = clean_text(result.get("answer", ""))
        source_docs = result.get("source_documents", [])
        
        # Evaluate answer quality
        if is_unhelpful_answer(answer):
            # Generate intelligent business fallback
            fallback_answer = generate_business_fallback(qtext, company_id)
            
            response = {
                "answer": fallback_answer,
                "fallback_used": True,
                "original_answer": answer,
                "source": "business_intelligence",
                "followups": [
                    "Would you like me to help create documentation for this?",
                    "Do you want to know about related procedures?",
                    "Need help with anything else?"
                ],
                "model_used": optimal_llm.model_name,
                "session_id": session_id,
                "response_time": time.time() - start_time
            }
            
            cache_response(qtext, company_id, response)
            update_metrics(time.time() - start_time, "business", company_id)
            return jsonify(response)
        
        # Smart truncation for voice compatibility
        if len(answer.split()) > 80:
            answer = smart_truncate(answer, 80)
        
        # Prepare successful response
        response = {
            "answer": answer,
            "fallback_used": False,
            "source": "sop",
            "source_documents": len(source_docs),
            "followups": generate_contextual_followups(qtext, answer),
            "model_used": optimal_llm.model_name,
            "complexity": complexity,
            "session_id": session_id,
            "response_time": time.time() - start_time,
            "cache_hit": False
        }
        
        # Cache successful responses
        cache_response(qtext, company_id, response)
        update_metrics(time.time() - start_time, "sop", company_id)
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Query error: {traceback.format_exc()}")
        performance_metrics["error_count"] += 1
        
        response = {
            "answer": "I'm having trouble processing your question right now. Please try rephrasing it or ask about a different topic.",
            "error": "Query processing failed",
            "source": "error",
            "followups": [
                "Try asking about a specific procedure",
                "Rephrase your question",
                "Ask about available documents"
            ],
            "session_id": session_id if 'session_id' in locals() else None,
            "response_time": time.time() - start_time
        }
        
        update_metrics(time.time() - start_time, "error", company_id if 'company_id' in locals() else None)
        return jsonify(response), 500

# ==== VAPI FUNCTION ENDPOINT ====
@app.route('/vapi-function', methods=['POST', 'OPTIONS'])
def vapi_function():
    """
    VAPI Function Call Endpoint for Generic Business Queries
    Accepts POST with JSON: { "functionName": ..., "args": { "query": ..., "company_id_slug": ... } }
    Returns: { "result": ... }
    Always returns 200 for VAPI compatibility.
    """
    if request.method == "OPTIONS":
        return "", 204
    
    start_time = time.time()
    try:
        if not request.is_json:
            logger.warning("VAPI function call: Non-JSON request received")
            return jsonify({"result": "Invalid request: JSON required."}), 200

        payload = request.get_json() or {}
        function_name = payload.get("functionName", "")
        args = payload.get("args", {}) or {}

        logger.info(f"VAPI function call received: functionName={function_name}, args={args}")

        # Extract query and company_id_slug
        raw_query = args.get("query", "")
        company_id = args.get("company_id_slug", "demo-business-123")
        qtext = sanitize_text_input(raw_query, 500) if raw_query else ""
        company_id = company_id.strip() if company_id else "demo-business-123"

        if not qtext:
            logger.info("VAPI function call: No query provided")
            return jsonify({"result": "Please provide a query in args.query."}), 200

        # Validate company ID
        if not validate_company_id(company_id):
            return jsonify({"result": "Invalid company identifier."}), 200

        # Check if vectorstore is ready
        if not ensure_vectorstore():
            fallback_response = (
                "I'm currently unable to access the business procedures database. "
                "However, here are general business guidelines:\n\n"
                "1. Customer Service: Always be professional and empathetic\n"
                "2. Safety First: Follow all safety protocols and procedures\n"
                "3. Documentation: Keep accurate records of all transactions\n"
                "4. Communication: Clear communication is essential\n"
                "5. Compliance: Follow all company policies and regulations\n\n"
                "For specific procedures, please consult your supervisor or employee handbook."
            )
            logger.info("VAPI function call: Vectorstore not ready, returning fallback response")
            return jsonify({"result": fallback_response}), 200

        # Expand query with business terms
        def expand_business_query(query):
            query_lower = query.lower()
            expansions = []
            business_terms = {
                "customer": ["customer", "client", "guest", "patron", "service"],
                "employee": ["employee", "staff", "worker", "personnel", "team"],
                "procedure": ["procedure", "process", "protocol", "steps", "workflow"],
                "policy": ["policy", "rule", "guideline", "standard", "regulation"],
                "safety": ["safety", "hazard", "risk", "danger", "precaution"],
                "emergency": ["emergency", "urgent", "crisis", "evacuation", "incident"]
            }
            for key, terms in business_terms.items():
                if key in query_lower:
                    expansions.extend(terms)
            expanded = query + " " + " ".join(expansions)
            return expanded[:1000]

        expanded_query = expand_business_query(qtext)

        try:
            # Set up retriever with company filtering
            retriever = vectorstore.as_retriever(
                search_kwargs={
                    "k": 5,
                    "filter": {"company_id_slug": company_id}
                }
            )
            
            # Create memory and LLM
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
            llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
            
            # Create QA chain
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm,
                retriever=retriever,
                memory=memory,
                return_source_documents=True,
                verbose=False
            )
            
            result = qa_chain.invoke({"question": expanded_query})
            answer = clean_text(result.get("answer", ""))
            source_docs = result.get("source_documents", [])

            # Check if answer is unhelpful and provide fallback
            if is_unhelpful_answer(answer):
                answer = generate_business_fallback(qtext, company_id)

            # Smart truncation for voice compatibility
            if len(answer.split()) > 80:
                answer = smart_truncate(answer, 80)

            logger.info(f"VAPI function call processed in {round(time.time() - start_time, 3)}s")
            return jsonify({"result": answer}), 200

        except Exception as query_error:
            logger.error(f"VAPI function call: Query processing error: {query_error}")
            return jsonify({
                "result": (
                    "I encountered an error processing your question. "
                    "For immediate business guidance:\n\n"
                    "• Follow company policies and procedures\n"
                    "• Maintain professional communication\n"
                    "• Document important interactions\n"
                    "• Report issues to your supervisor\n\n"
                    "Please try again or contact your supervisor."
                )
            }), 200

    except Exception as e:
        logger.error(f"VAPI function call: Unexpected error: {traceback.format_exc()}")
        return jsonify({
            "result": "I'm experiencing technical difficulties. Please try again or contact support."
        }), 200

# ==== VOICE QUERY ENDPOINT ====
@app.route('/voice-query', methods=['POST', 'OPTIONS'])
def voice_query():
    """
    Voice Query Endpoint
    Accepts audio or text queries and returns both text and audio response
    """
    if request.method == "OPTIONS":
        return "", 204
    
    start_time = time.time()
    
    try:
        # Ensure vectorstore is ready
        if not ensure_vectorstore():
            return jsonify({"error": "Service temporarily unavailable"}), 503
        
        # Handle different content types
        if request.content_type and 'multipart/form-data' in request.content_type:
            # Audio file upload
            audio_file = request.files.get('audio')
            if not audio_file:
                return jsonify({"error": "No audio file provided"}), 400
            
            # Validate audio file
            if not audio_file.filename.lower().endswith(('.wav', '.mp3', '.m4a')):
                return jsonify({"error": "Invalid audio format. Use WAV, MP3, or M4A"}), 400
            
            # Read audio data
            audio_data = audio_file.read()
            if len(audio_data) > 10 * 1024 * 1024:  # 10MB limit
                return jsonify({"error": "Audio file too large (max 10MB)"}), 413
            
            # Transcribe audio
            transcript = transcribe_audio(audio_data)
            if not transcript:
                return jsonify({"error": "Failed to transcribe audio"}), 400
            
            query_text = transcript
            
        elif request.is_json:
            # Text query
            payload = request.get_json() or {}
            query_text = sanitize_text_input(payload.get("query", ""), 500)
            if not query_text:
                return jsonify({"error": "Query text required"}), 400
        else:
            return jsonify({"error": "Invalid content type. Use multipart/form-data for audio or JSON for text"}), 400
        
        # Extract and validate company ID
        company_id = request.form.get('company_id_slug') or request.get_json().get('company_id_slug', 'demo-business-123')
        company_id = company_id.strip() if company_id else "demo-business-123"
        
        if not validate_company_id(company_id):
            return jsonify({"error": "Invalid company identifier"}), 400
        
        # Rate limiting
        rate_ok, rate_msg = check_rate_limit(company_id, 'query')
        if not rate_ok:
            return jsonify({"error": rate_msg}), 429
        
        # Process query using existing logic
        session_id = f"{company_id}_{int(time.time())}"
        
        # Set up retriever with company filtering
        retriever = vectorstore.as_retriever(
            search_kwargs={
                "k": 7,
                "filter": {"company_id_slug": company_id}
            }
        )
        
        # Get or create session memory
        memory = get_session_memory(session_id)
        
        # Determine optimal model
        complexity = get_query_complexity(query_text)
        optimal_llm = get_optimal_llm(complexity)
        
        # Create conversational QA chain
        qa = ConversationalRetrievalChain.from_llm(
            optimal_llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            verbose=False
        )
        
        # Execute query
        result = qa.invoke({"question": query_text})
        answer = clean_text(result.get("answer", ""))
        source_docs = result.get("source_documents", [])
        
        # Evaluate answer quality
        if is_unhelpful_answer(answer):
            answer = generate_business_fallback(query_text, company_id)
            source = "business_intelligence"
        else:
            source = "sop"
        
        # Smart truncation for voice compatibility
        if len(answer.split()) > 80:
            answer = smart_truncate(answer, 80)
        
        # Generate audio response
        audio_url = None
        try:
            # Generate TTS audio
            tts_response = requests.post(
                "https://api.elevenlabs.io/v1/text-to-speech/bIHbv24MWmeRgasZH58o/stream",
                headers={
                    "xi-api-key": os.getenv("ELEVENLABS_API_KEY"),
                    "Content-Type": "application/json"
                },
                json={
                    "text": answer,
                    "voice_settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.75,
                        "style": 0.0,
                        "use_speaker_boost": True
                    },
                    "model_id": "eleven_multilingual_v2"
                },
                timeout=30
            )
            
            if tts_response.status_code == 200:
                # Save audio to cache
                content_hash = hashlib.sha256(answer.encode()).hexdigest()[:16]
                cache_key = f"{company_id}_{content_hash}.mp3"
                cache_path = os.path.join(AUDIO_CACHE_DIR, cache_key)
                
                with open(cache_path, "wb") as f:
                    f.write(tts_response.content)
                
                audio_url = f"{request.host_url.rstrip('/')}/static/audio/{cache_key}"
                logger.info(f"Generated audio response: {cache_key}")
            else:
                logger.error(f"TTS generation failed: {tts_response.status_code}")
                
        except Exception as audio_error:
            logger.error(f"Audio generation error: {audio_error}")
        
        # Prepare response
        response = {
            "query": query_text,
            "answer": answer,
            "audio_url": audio_url,
            "source": source,
            "source_documents": len(source_docs),
            "model_used": optimal_llm.model_name,
            "session_id": session_id,
            "response_time": time.time() - start_time,
            "company_id_slug": company_id
        }
        
        # Update metrics
        update_metrics(time.time() - start_time, source, company_id)
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Voice query error: {traceback.format_exc()}")
        return jsonify({
            "error": "Voice query processing failed",
            "message": "Please try again or use text input"
        }), 500

@app.route('/voice-reply', methods=['POST', 'OPTIONS'])
def voice_reply():
    """Enhanced text-to-speech with caching and optimization"""
    if request.method == "OPTIONS":
        return "", 204
    
    try:
        # Input validation
        if not request.is_json:
            return jsonify({"error": "Invalid request format - JSON required"}), 400
        
        data = request.get_json() or {}
        text = clean_text(data.get("query", ""))
        company_id = data.get("company_id_slug", "default")
        
        if not text:
            return jsonify({"error": "Text is required"}), 400
        
        # Validate company ID if provided
        if company_id != "default" and not validate_company_id(company_id):
            return jsonify({"error": "Invalid company identifier"}), 400
        
        # Rate limiting for TTS
        if company_id != "default":
            rate_ok, rate_msg = check_rate_limit(company_id, 'tts')
            if not rate_ok:
                return jsonify({"error": rate_msg}), 429
        
        # Optimize text for TTS
        tts_text = text[:500] if len(text) > 500 else text
        
        # Generate cache key
        content_hash = hashlib.sha256(tts_text.encode()).hexdigest()[:16]
        cache_key = f"{company_id}_{content_hash}.mp3"
        cache_path = os.path.join(AUDIO_CACHE_DIR, cache_key)
        
        # Check cache first
        if os.path.exists(cache_path):
            # Verify cache file isn't too old
            if time.time() - os.path.getmtime(cache_path) < AUDIO_CACHE_TTL:
                logger.debug(f"Serving cached audio: {cache_key}")
                return send_file(cache_path, mimetype="audio/mp3", as_attachment=False)
            else:
                # Remove expired cache
                try:
                    os.remove(cache_path)
                except:
                    pass
        
        # Generate new audio
        logger.info(f"Generating TTS audio for: {tts_text[:50]}...")
        
        tts_response = requests.post(
            "https://api.elevenlabs.io/v1/text-to-speech/tnSpp4vdxKPjI9w0GnoV/stream",
            headers={
                "xi-api-key": os.getenv("ELEVENLABS_API_KEY"),
                "Content-Type": "application/json"
            },
            json={
                "text": tts_text,
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.75,
                    "style": 0.0,
                    "use_speaker_boost": True
                },
                "model_id": "eleven_multilingual_v2"
            },
            timeout=30
        )
        
        if tts_response.status_code != 200:
            logger.error(f"ElevenLabs TTS error: {tts_response.status_code} - {tts_response.text}")
            return jsonify({"error": "TTS service temporarily unavailable"}), 502
        
        audio_data = tts_response.content
        
        # Cache the audio
        try:
            with open(cache_path, "wb") as f:
                f.write(audio_data)
            logger.debug(f"Cached TTS audio: {cache_key}")
        except Exception as e:
            logger.warning(f"Failed to cache audio: {e}")
        
        # Return audio
        return send_file(
            io.BytesIO(audio_data),
            mimetype="audio/mp3",
            as_attachment=False,
            download_name="response.mp3"
        )
        
    except requests.exceptions.Timeout:
        logger.error("TTS request timeout")
        return jsonify({"error": "TTS service timeout"}), 504
    except requests.exceptions.RequestException as e:
        logger.error(f"TTS request error: {e}")
        return jsonify({"error": "TTS service error"}), 502
    except Exception as e:
        logger.error(f"TTS error: {traceback.format_exc()}")
        return jsonify({"error": "TTS generation failed"}), 500

# ==== AUDIO FILE SERVING ====
@app.route('/static/audio/<path:filename>')
def serve_audio(filename):
    """Serve cached audio files"""
    try:
        # Security validation
        safe_filename = secure_filename(filename)
        if safe_filename != filename:
            return jsonify({"error": "Invalid filename"}), 400
        
        file_path = os.path.join(AUDIO_CACHE_DIR, safe_filename)
        if not os.path.exists(file_path):
            return jsonify({"error": "Audio file not found"}), 404
        
        # Additional security: ensure file is within audio cache directory
        if not os.path.abspath(file_path).startswith(os.path.abspath(AUDIO_CACHE_DIR)):
            return jsonify({"error": "Access denied"}), 403
        
        return send_from_directory(AUDIO_CACHE_DIR, safe_filename, mimetype="audio/mp3")
    except Exception as e:
        logger.error(f"Audio serve error: {e}")
        return jsonify({"error": "Audio serve failed"}), 500

@app.route('/company-docs/<company_id_slug>')
def company_docs(company_id_slug):
    """Get documents for a specific company with enhanced metadata"""
    # Validate company ID
    if not validate_company_id(company_id_slug):
        return jsonify({"error": "Invalid company identifier"}), 400
    
    try:
        docs = get_company_documents_internal(company_id_slug)
        
        # Sort by upload date (newest first)
        docs.sort(key=lambda x: x.get('uploaded_at', 0), reverse=True)
        
        # Add summary statistics
        total_size = sum(doc.get('file_size', 0) for doc in docs)
        total_chunks = sum(doc.get('chunk_count', 0) for doc in docs)
        
        return jsonify({
            "documents": docs,
            "summary": {
                "total_documents": len(docs),
                "total_size_bytes": total_size,
                "total_chunks": total_chunks,
                "company_id": company_id_slug
            }
        })
        
    except Exception as e:
        logger.error(f"Error fetching company docs: {e}")
        return jsonify({"error": "Failed to fetch documents"}), 500

@app.route('/continue', methods=['POST', 'OPTIONS'])
def continue_conversation():
    """Continue from previous truncated response"""
    if request.method == "OPTIONS":
        return "", 204
    
    try:
        if not request.is_json:
            return jsonify({"error": "Invalid request format - JSON required"}), 400
        
        payload = request.get_json() or {}
        session_id = validate_session_id(payload.get("session_id", ""))
        company_id = payload.get("company_id_slug", "").strip()
        
        # Validation
        if not session_id:
            return jsonify({"error": "Session ID required"}), 400
        
        if not validate_company_id(company_id):
            return jsonify({"error": "Invalid company identifier"}), 400
        
        # Rate limiting
        rate_ok, rate_msg = check_rate_limit(company_id, 'query')
        if not rate_ok:
            return jsonify({"error": rate_msg}), 429
        
        # Check if session exists
        if session_id not in conversation_sessions:
            return jsonify({
                "answer": "I don't see a previous conversation to continue from. Please ask a new question.",
                "source": "no_session",
                "session_id": session_id
            })
        
        # Get conversation memory
        memory = get_session_memory(session_id)
        
        # Check for previous messages
        if not hasattr(memory, 'chat_memory') or not memory.chat_memory.messages:
            return jsonify({
                "answer": "I don't have a previous conversation to continue from. What would you like to know?",
                "source": "no_history",
                "session_id": session_id
            })
        
        # Get last AI response
        last_response = ""
        for msg in reversed(memory.chat_memory.messages):
            if hasattr(msg, 'content') and len(msg.content) > 50:
                last_response = msg.content
                break
        
        # Check if continuation is appropriate
        if last_response:
            continuation_triggers = [
                "Would you like me to continue",
                "Should I continue",
                "For complete details",
                "... For more details"
            ]
            
            if any(trigger in last_response for trigger in continuation_triggers):
                # Process continuation
                continuation_query = "Please continue from where you left off with the rest of the details."
                
                # Create temporary request context for internal call
                temp_payload = {
                    "query": continuation_query,
                    "company_id_slug": company_id,
                    "session_id": session_id
                }
                
                # Store original request data
                original_json = request.get_json()
                
                # Temporarily replace request data
                with app.test_request_context('/query', json=temp_payload, method='POST'):
                    response = query_sop()
                
                return response
            else:
                return jsonify({
                    "answer": "The previous response doesn't appear to be truncated. What specific aspect would you like me to elaborate on?",
                    "source": "clarification",
                    "session_id": session_id,
                    "followups": [
                        "Ask about a specific part",
                        "Request more details on a topic",
                        "Ask a new question"
                    ]
                })
        else:
            return jsonify({
                "answer": "I don't have a previous response to continue from. What would you like to know?",
                "source": "no_previous_response",
                "session_id": session_id
            })
            
    except Exception as e:
        logger.error(f"Continue conversation error: {traceback.format_exc()}")
        return jsonify({"error": "Failed to continue conversation"}), 500

@app.route('/sop-status')
def sop_status():
    """Get document processing status"""
    try:
        if os.path.exists(STATUS_FILE):
            return send_file(STATUS_FILE, mimetype='application/json')
        return jsonify({})
    except Exception as e:
        logger.error(f"Error serving status file: {e}")
        return jsonify({"error": "Status unavailable"}), 500

@app.route('/metrics')
def get_metrics():
    """Get comprehensive performance metrics (public for demo)"""
    try:
        # Calculate additional metrics
        total_queries = performance_metrics.get("total_queries", 0)
        cache_hits = performance_metrics.get("cache_hits", 0)
        cache_hit_rate = round((cache_hits / max(1, total_queries)) * 100, 2)
        
        # Response time statistics
        response_times = performance_metrics.get("response_times", [])
        if response_times:
            avg_time = sum(response_times) / len(response_times)
            max_time = max(response_times)
            min_time = min(response_times)
        else:
            avg_time = max_time = min_time = 0
        
        metrics_data = {
            "performance": {
                "total_queries": total_queries,
                "cache_hits": cache_hits,
                "cache_hit_rate": cache_hit_rate,
                "avg_response_time": round(avg_time, 3),
                "max_response_time": round(max_time, 3),
                "min_response_time": round(min_time, 3),
                "error_count": performance_metrics.get("error_count", 0)
            },
            "usage": {
                "model_usage": performance_metrics.get("model_usage", {}),
                "response_sources": performance_metrics.get("response_sources", {}),
                "companies": {
                    company: {
                        "queries": data["queries"],
                        "avg_response_time": data["avg_response_time"]
                    }
                    for company, data in performance_metrics.get("companies", {}).items()
                }
            },
            "system": {
                "cache_size": len(query_cache),
                "active_sessions": len(conversation_sessions),
                "vectorstore_loaded": vectorstore is not None,
                "total_documents": len(glob.glob(os.path.join(SOP_FOLDER, "*.*")))
            },
            "rate_limits": {
                "demo_limits": DEMO_RATE_LIMITS,
                "production_limits": PRODUCTION_RATE_LIMITS
            }
        }
        
        return jsonify(metrics_data)
        
    except Exception as e:
        logger.error(f"Metrics error: {e}")
        return jsonify({"error": "Metrics unavailable"}), 500

@app.route('/reload-db', methods=['POST'])
def reload_db():
    """Reload the vector database"""
    try:
        load_vectorstore()
        return jsonify({
            "message": "Vectorstore reloaded successfully",
            "status": "loaded" if vectorstore else "failed",
            "timestamp": time.time()
        })
    except Exception as e:
        logger.error(f"Reload error: {e}")
        return jsonify({"error": "Reload failed"}), 500

@app.route('/clear-cache', methods=['POST'])
def clear_cache():
    """Clear query and audio cache"""
    try:
        # Clear query cache
        cache_size = len(query_cache)
        query_cache.clear()
        
        # Clear audio cache
        audio_files_cleared = 0
        if os.path.exists(AUDIO_CACHE_DIR):
            for filename in os.listdir(AUDIO_CACHE_DIR):
                if filename.endswith('.mp3'):
                    try:
                        os.remove(os.path.join(AUDIO_CACHE_DIR, filename))
                        audio_files_cleared += 1
                    except:
                        pass
        
        logger.info(f"Cache cleared: {cache_size} queries, {audio_files_cleared} audio files")
        
        return jsonify({
            "message": "Cache cleared successfully",
            "query_cache_cleared": cache_size,
            "audio_files_cleared": audio_files_cleared,
            "timestamp": time.time()
        })
    except Exception as e:
        logger.error(f"Cache clear error: {e}")
        return jsonify({"error": "Cache clear failed"}), 500

@app.route('/clear-sessions', methods=['POST'])
def clear_sessions():
    """Clear conversation sessions"""
    try:
        session_count = len(conversation_sessions)
        conversation_sessions.clear()
        
        logger.info(f"Cleared {session_count} conversation sessions")
        
        return jsonify({
            "message": f"Cleared {session_count} conversation sessions",
            "timestamp": time.time()
        })
    except Exception as e:
        logger.error(f"Session clear error: {e}")
        return jsonify({"error": "Session clear failed"}), 500

@app.route('/session-info/<session_id>')
def get_session_info(session_id):
    """Get information about a conversation session"""
    try:
        # Sanitize session ID
        clean_session_id = validate_session_id(session_id)
        
        if clean_session_id in conversation_sessions:
            session_data = conversation_sessions[clean_session_id]
            memory = session_data['memory']
            
            messages = []
            if hasattr(memory, 'chat_memory') and hasattr(memory.chat_memory, 'messages'):
                messages = [
                    {
                        "type": type(msg).__name__,
                        "content": msg.content[:200],  # Truncate for API response
                        "timestamp": time.time()  # Placeholder
                    }
                    for msg in memory.chat_memory.messages
                ]
            
            return jsonify({
                "session_id": clean_session_id,
                "message_count": len(messages),
                "messages": messages,
                "created_at": session_data.get('created_at'),
                "last_accessed": session_data.get('last_accessed')
            })
        else:
            return jsonify({"error": "Session not found"}), 404
            
    except Exception as e:
        logger.error(f"Session info error: {e}")
        return jsonify({"error": "Session info unavailable"}), 500

# ==== BACKGROUND TASKS & CLEANUP ====
def cleanup_expired_data():
    """Periodic cleanup of expired data"""
    try:
        # Clean expired sessions
        cleanup_expired_sessions()
        
        # Clean old audio cache
        if os.path.exists(AUDIO_CACHE_DIR):
            cutoff_time = time.time() - AUDIO_CACHE_TTL
            cleaned_count = 0
            
            for filename in os.listdir(AUDIO_CACHE_DIR):
                filepath = os.path.join(AUDIO_CACHE_DIR, filename)
                try:
                    if os.path.getmtime(filepath) < cutoff_time:
                        os.remove(filepath)
                        cleaned_count += 1
                except:
                    pass
            
            if cleaned_count > 0:
                logger.info(f"Cleaned {cleaned_count} old audio cache files")
        
        # Clean old query cache entries
        current_time = time.time()
        expired_keys = [
            key for key, data in query_cache.items()
            if current_time - data['timestamp'] > QUERY_CACHE_TTL
        ]
        
        for key in expired_keys:
            del query_cache[key]
        
        if expired_keys:
            logger.info(f"Cleaned {len(expired_keys)} expired cache entries")
        
        # Save metrics
        save_metrics()
        
        logger.debug("Cleanup completed successfully")
        
    except Exception as e:
        logger.error(f"Cleanup error: {e}")

def periodic_cleanup():
    """Run cleanup and schedule next run"""
    cleanup_expired_data()
    # Schedule next cleanup in 1 hour
    Timer(3600, periodic_cleanup).start()

# ==== ENHANCED ERROR HANDLERS ====
@app.errorhandler(400)
def bad_request(error):
    return jsonify({
        "error": "Bad request",
        "message": "Invalid request format or parameters"
    }), 400

@app.errorhandler(401)
def unauthorized(error):
    logger.warning(f"Unauthorized access attempt: {request.method} {request.path}")
    logger.warning(f"Client IP: {request.environ.get('HTTP_X_FORWARDED_FOR', request.environ.get('REMOTE_ADDR'))}")
    return jsonify({
        "error": "Unauthorized",
        "message": "Authentication required or invalid credentials"
    }), 401

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Not found", 
        "message": "The requested endpoint does not exist"
    }), 404

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        "error": "File too large",
        "message": f"Maximum file size is {MAX_FILE_SIZE // 1024 // 1024}MB"
    }), 413

@app.errorhandler(429)
def ratelimit_handler(error):
    return jsonify({
        "error": "Rate limit exceeded",
        "message": "Too many requests. Please try again later.",
        "retry_after": 60
    }), 429

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    logger.error(f"Request details: {request.method} {request.path}")
    logger.error(f"Request headers: {dict(request.headers)}")
    logger.error(f"Request data: {request.get_data()[:500] if request.get_data() else 'No data'}")
    
    # Log stack trace for debugging
    import traceback
    logger.error(f"Stack trace: {traceback.format_exc()}")
    
    return jsonify({
        "error": "Internal server error",
        "message": "Service temporarily unavailable",
        "request_id": hashlib.md5(f"{time.time()}".encode()).hexdigest()[:8]
    }), 500

@app.errorhandler(503)
def service_unavailable(error):
    return jsonify({
        "error": "Service unavailable",
        "message": "Service is temporarily unavailable"
    }), 503

# ==== REQUEST PROCESSING & MONITORING ====
request_counter = 0

@app.before_request
def before_request():
    """Pre-request processing and monitoring"""
    global request_counter
    request_counter += 1
    
    # Enhanced request logging for debugging
    if request.endpoint not in ['static', 'healthz']:
        client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.environ.get('REMOTE_ADDR'))
        user_agent = request.headers.get('User-Agent', 'Unknown')[:100]
        
        logger.info(f"Request {request_counter}: {request.method} {request.path} from {client_ip}")
        logger.debug(f"User-Agent: {user_agent}")
        logger.debug(f"Headers: {dict(request.headers)}")
        
        # Log specific VAPI requests for debugging
        if 'vapi' in request.path.lower() or 'voice' in request.path.lower():
            logger.info(f"VAPI/Voice request: {request.method} {request.path}")
            logger.debug(f"Request data: {request.get_data()[:200] if request.get_data() else 'No data'}")
    
    # Periodic cleanup every 100 requests
    if request_counter % 100 == 0:
        cleanup_expired_data()

# ==== STARTUP & INITIALIZATION ====
def initialize_application():
    """Initialize application on startup"""
    logger.info("Initializing OpsVoice API...")
    
    # Validate environment
    if not validate_environment():
        logger.error("Exiting due to missing environment variables.")
        return

    # Load vectorstore
    logger.info("Loading vector store...")
    load_vectorstore()

    # Load existing metrics
    load_metrics()

    # Load demo documents for demo-business-123
    logger.info("Loading demo documents for demo-business-123...")
    load_demo_documents()
    
    # Start periodic cleanup
    Timer(3600, periodic_cleanup).start()
    
    # Log system information
    logger.info(f"Data path: {DATA_PATH}")
    logger.info(f"Persistent storage: {os.path.exists('/data')}")
    logger.info(f"Existing vectorstore: {os.path.exists(CHROMA_DIR)}")
    
    existing_files = len(glob.glob(os.path.join(SOP_FOLDER, '*')))
    logger.info(f"Existing SOP files: {existing_files}")
    
    logger.info(f"Demo rate limits: {DEMO_RATE_LIMITS}")
    logger.info(f"Production rate limits: {PRODUCTION_RATE_LIMITS}")
    
    logger.info("OpsVoice API initialization complete")

# ==== INITIALIZATION FOR PRODUCTION ====
def create_app():
    """Factory function to create Flask app (for production deployment)"""
    if not globals().get('app'):
        raise RuntimeError("Flask app not properly initialized")
    
    # Initialize application components
    initialize_application()
    return app

# Initialize application if not in main (for gunicorn)
if __name__ != '__main__':
    # This runs when imported by gunicorn
    logger.info("Initializing app for production deployment...")
    initialize_application()

if __name__ == '__main__':
    # Initialize application
    initialize_application()
    
    # Get configuration
    port = int(os.environ.get('PORT', 10000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    logger.info(f"Starting OpsVoice API v3.1.0 on port {port}")
    logger.info(f"Debug mode: {debug}")
    logger.info(f"CORS origins: {ALLOWED_ORIGINS}")
    
    # Start Flask application
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug,
        threaded=True,
        use_reloader=False  # Disable reloader in production
    )