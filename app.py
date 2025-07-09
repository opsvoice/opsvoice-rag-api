from flask import Flask, request, jsonify, send_from_directory, send_file, make_response
import os, glob, json, re, time, io, shutil, requests, hashlib
from dotenv import load_dotenv
from threading import Thread
from functools import lru_cache
from langchain_community.document_loaders import UnstructuredWordDocumentLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import secrets
from datetime import datetime, timedelta
import traceback
import logging

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---- FIXED: Persistent Data Path ----
# Use /data for Render persistent disk, fallback for local dev
if os.path.exists("/data"):
    DATA_PATH = "/data"  # Render persistent disk
else:
    DATA_PATH = os.path.join(os.getcwd(), "data")  # Local development

# ---- Configuration ----
conversation_sessions = {}
rate_limits = {}
MAX_REQUESTS_PER_MINUTE = 50
MAX_REQUESTS_PER_MINUTE_DEMO = 999  # Unlimited for demo
MAX_CACHE_SIZE = 500
ALLOWED_EXTENSIONS = {'pdf', 'docx'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# ---- Paths & Setup ----
SOP_FOLDER = os.path.join(DATA_PATH, "sop-files")
CHROMA_DIR = os.path.join(DATA_PATH, "chroma_db")
AUDIO_CACHE_DIR = os.path.join(DATA_PATH, "audio_cache")
STATUS_FILE = os.path.join(SOP_FOLDER, "status.json")

# ---- FIXED: Ensure directories exist WITHOUT deleting data ----
os.makedirs(SOP_FOLDER, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)
os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)

# Create demo documents if they don't exist
DEMO_STATUS_FILE = os.path.join(SOP_FOLDER, "demo_status.json")

def ensure_demo_documents():
    """Ensure demo business always has access to demo documents"""
    try:
        # Check if demo documents exist
        demo_files = [f for f in os.listdir(SOP_FOLDER) if f.startswith("demo-business-123_")]
        
        if len(demo_files) < 1:
            # Create a simple demo document
            demo_content = """DEMO COMPANY HANDBOOK

Customer Service Procedures:
1. Always greet customers with a smile
2. Listen to their concerns carefully
3. If a customer is upset or angry, remain calm and professional
4. Offer solutions and alternatives
5. Escalate to manager if needed

Refund Policy:
- Refunds are available within 30 days with receipt
- Manager approval required for refunds over $100
- Cash refunds for cash purchases
- Credit card refunds to original payment method

Employee Onboarding:
First day procedures:
1. Welcome new employee
2. Provide company handbook
3. Set up computer and email
4. Introduce to team members
5. Schedule training sessions

Training Requirements:
- All employees must complete safety training
- Customer service training within first week
- Regular updates on company policies"""

            demo_filename = f"demo-business-123_{int(time.time())}_demo_handbook.txt"
            demo_path = os.path.join(SOP_FOLDER, demo_filename)
            
            with open(demo_path, 'w') as f:
                f.write(demo_content)
            
            # Update status
            status_data = {}
            if os.path.exists(STATUS_FILE):
                with open(STATUS_FILE, 'r') as f:
                    status_data = json.load(f)
            
            status_data[demo_filename] = {
                "title": "Demo Company Handbook",
                "company_id_slug": "demo-business-123",
                "filename": demo_filename,
                "uploaded_at": time.time(),
                "status": "embedded",
                "is_demo": True
            }
            
            with open(STATUS_FILE, 'w') as f:
                json.dump(status_data, f, indent=2)
                
            logger.info(f"Created demo document: {demo_filename}")
            
    except Exception as e:
        logger.error(f"Error ensuring demo documents: {e}")

# Performance tracking
performance_metrics = {
    "total_queries": 0,
    "cache_hits": 0,
    "avg_response_time": 0,
    "model_usage": {"gpt-3.5-turbo": 0, "gpt-4": 0},
    "response_sources": {"sop": 0, "fallback": 0, "cache": 0, "error": 0}
}

# Enhanced caching
query_cache = {}

# Initialize embeddings and vectorstore
embedding = OpenAIEmbeddings()
vectorstore = None

# ---- Flask Setup ----
app = Flask(__name__)

# ---- FIXED: Security-Enhanced CORS ----
# Only allow specific origins in production
ALLOWED_ORIGINS = [
    "https://opsvoice-widget.vercel.app",
    "http://localhost:3000",
    "http://127.0.0.1:3000"
]

# In development, be more permissive
if app.debug or os.getenv("FLASK_ENV") == "development":
    ALLOWED_ORIGINS.append("*")

CORS(app, resources={
    r"/*": {
        "origins": ALLOWED_ORIGINS,
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"],
        "supports_credentials": True,
        "max_age": 3600
    }
})

@app.after_request
def after_request(response):
    """Enhanced CORS headers with security"""
    origin = request.headers.get('Origin')
    
    # Only set origin if it's in allowed list (security fix)
    if origin in ALLOWED_ORIGINS or "*" in ALLOWED_ORIGINS:
        response.headers['Access-Control-Allow-Origin'] = origin
    
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With'
    
    # Security headers
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    
    if request.method == 'OPTIONS':
        response.headers['Access-Control-Max-Age'] = '3600'
        response.status_code = 200
        
    return response

# ---- Security Functions ----
def validate_file_upload(file):
    """Enhanced file validation with security checks"""
    if not file or not file.filename:
        return False, "No file uploaded"
    
    filename = secure_filename(file.filename)
    if not filename:
        return False, "Invalid filename"
    
    # Check for directory traversal attempts
    if '..' in filename or '/' in filename or '\\' in filename:
        return False, "Invalid filename characters"
    
    if '.' not in filename:
        return False, "File must have extension"
    
    ext = filename.rsplit('.', 1)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Only PDF and DOCX files allowed"
    
    # Check file size
    file.seek(0, 2)
    size = file.tell()
    file.seek(0)
    if size > MAX_FILE_SIZE:
        return False, "File too large (max 10MB)"
    
    if size < 100:  # Too small to be valid
        return False, "File appears to be empty or corrupted"
    
    return True, filename

def check_rate_limit(tenant: str) -> bool:
    """Rate limiting with demo bypass"""
    global rate_limits
    
    # Demo business gets unlimited requests
    if tenant == "demo-business-123":
        return True
    
    current_minute = int(time.time() // 60)
    key = f"{tenant}:{current_minute}"
    
    # Clean old entries
    cutoff = current_minute - 5
    rate_limits = {k: v for k, v in rate_limits.items() 
                   if int(k.split(':')[1]) > cutoff}
    
    count = rate_limits.get(key, 0)
    limit = MAX_REQUESTS_PER_MINUTE
    
    if count >= limit:
        return False
    
    rate_limits[key] = count + 1
    return True

def validate_company_id(company_id: str) -> bool:
    """Enhanced company ID validation"""
    if not company_id or len(company_id) < 3 or len(company_id) > 50:
        return False
    
    # Only allow alphanumeric, hyphens, underscores
    if not re.match(r'^[a-zA-Z0-9_-]+$', company_id):
        return False
    
    # Prevent path traversal and dangerous patterns
    dangerous_patterns = ['..', '/', '\\', '<', '>', '"', "'", '&', '|', ';']
    if any(pattern in company_id for pattern in dangerous_patterns):
        return False
    
    return True

def sanitize_text(text: str) -> str:
    """Enhanced text sanitization"""
    if not text:
        return ""
    
    # Remove dangerous characters but preserve normal punctuation
    text = re.sub(r'[<>"\'\x00\r\n\t]', '', text)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Limit length
    return text[:1000].strip()

def safe_json_response(data, status_code=200):
    """Return safe JSON responses with error handling"""
    try:
        response = jsonify(data)
        response.status_code = status_code
        return response
    except Exception as e:
        logger.error(f"JSON response error: {e}")
        return jsonify({"error": "Internal error"}), 500

# ---- Utility Functions ----
def clean_text(txt: str) -> str:
    """Clean text for TTS with enhanced sanitization"""
    if not txt:
        return ""
    
    # Remove problematic characters
    txt = re.sub(r"\s+", " ", txt)
    txt = txt.replace("\u2022", "-").replace("\t", " ")
    txt = re.sub(r"[*#]+", "", txt)
    txt = re.sub(r"\[.*?\]", "", txt)
    txt = re.sub(r"[<>\"'\x00\r\n]", "", txt)  # Security: remove dangerous chars
    txt = txt.strip()
    
    return txt

def smart_truncate(text, max_words=150):
    """Smart truncation preserving sentences"""
    if not text:
        return text
    
    words = text.split()
    if len(words) <= max_words:
        return text
    
    truncated = " ".join(words[:max_words])
    
    # Find last complete sentence
    sentence_endings = ['.', '!', '?', ':']
    last_ending = -1
    
    for ending in sentence_endings:
        pos = truncated.rfind(ending)
        if pos > last_ending:
            last_ending = pos
    
    if last_ending > len(truncated) * 0.6:
        return truncated[:last_ending + 1] + " Would you like me to continue with more details?"
    else:
        return " ".join(words[:130]) + "... Should I continue with the rest of the details?"

def get_session_memory(session_id):
    """Get or create conversation memory with cleanup"""
    global conversation_sessions
    
    # Clean old sessions if too many
    if len(conversation_sessions) > 100:
        # Keep only 50 most recently accessed
        sorted_sessions = sorted(conversation_sessions.items(), 
                               key=lambda x: getattr(x[1], 'last_accessed', 0))
        conversation_sessions = dict(sorted_sessions[-50:])
    
    if session_id not in conversation_sessions:
        memory = ConversationBufferMemory(
            memory_key="chat_history", 
            return_messages=True,
            output_key="answer"
        )
        memory.last_accessed = time.time()
        conversation_sessions[session_id] = memory
    else:
        conversation_sessions[session_id].last_accessed = time.time()
    
    return conversation_sessions[session_id]

def get_query_complexity(query: str) -> str:
    """Determine query complexity for model selection"""
    words = query.lower().split()
    
    simple_indicators = [
        len(words) <= 8,
        any(word in query.lower() for word in ['what', 'when', 'where', 'who', 'how many']),
        query.endswith('?') and len(words) <= 6
    ]
    
    complex_indicators = [
        len(words) > 15,
        any(word in query.lower() for word in ['analyze', 'compare', 'explain why', 'walk me through']),
        query.count('?') > 1
    ]
    
    if sum(complex_indicators) > 0:
        return "complex"
    elif sum(simple_indicators) >= 2:
        return "simple"
    else:
        return "medium"

def get_optimal_llm(complexity: str) -> ChatOpenAI:
    """Select optimal model based on complexity"""
    global performance_metrics
    
    if complexity == "simple":
        performance_metrics["model_usage"]["gpt-3.5-turbo"] += 1
        return ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    else:
        performance_metrics["model_usage"]["gpt-4"] += 1
        return ChatOpenAI(temperature=0, model="gpt-4")

def get_cache_key(query: str, company_id: str) -> str:
    """Generate secure cache key"""
    combined = f"{company_id}:{query.lower().strip()}"
    return hashlib.md5(combined.encode()).hexdigest()

def get_cached_response(query: str, company_id: str) -> dict:
    """Get cached response if available"""
    cache_key = get_cache_key(query, company_id)
    cached = query_cache.get(cache_key)
    
    if cached and time.time() - cached['timestamp'] < 3600:  # 1 hour cache
        performance_metrics["cache_hits"] += 1
        performance_metrics["response_sources"]["cache"] += 1
        return cached['response']
    
    return None

def cache_response(query: str, company_id: str, response: dict):
    """Cache response with size management"""
    cache_key = get_cache_key(query, company_id)
    query_cache[cache_key] = {
        'response': response,
        'timestamp': time.time()
    }
    
    # Manage cache size
    if len(query_cache) > MAX_CACHE_SIZE:
        oldest_keys = sorted(query_cache.keys(), key=lambda k: query_cache[k]['timestamp'])[:100]
        for key in oldest_keys:
            del query_cache[key]

def update_metrics(response_time: float, source: str):
    """Update performance metrics"""
    global performance_metrics
    
    performance_metrics["total_queries"] += 1
    performance_metrics["response_sources"][source] = performance_metrics["response_sources"].get(source, 0) + 1
    
    # Update average response time
    current_avg = performance_metrics["avg_response_time"]
    total_queries = performance_metrics["total_queries"]
    new_avg = ((current_avg * (total_queries - 1)) + response_time) / total_queries
    performance_metrics["avg_response_time"] = round(new_avg, 2)

def is_unhelpful_answer(text):
    """Detect unhelpful answers"""
    if not text or not text.strip(): 
        return True
        
    low = text.lower()
    triggers = ["don't know", "no information", "i'm not sure", "sorry", 
                "unavailable", "not covered", "cannot find", "no specific information",
                "i don't have", "unable to find", "context doesn't", "doesn't contain"]
    
    has_trigger = any(t in low for t in triggers)
    is_too_short = len(low.split()) < 8
    
    return has_trigger or is_too_short

def generate_contextual_followups(query: str, answer: str) -> list:
    """Generate smart follow-up questions"""
    q = query.lower()
    a = answer.lower()
    base_followups = []
    
    # Answer-based followups
    if any(word in a for word in ["step", "procedure", "process"]):
        base_followups.append("Would you like the complete step-by-step procedure?")
    
    if any(word in a for word in ["policy", "rule", "requirement"]):
        base_followups.append("Do you need to know about related policies?")
    
    if any(word in a for word in ["form", "document", "paperwork"]):
        base_followups.append("Do you need help finding the actual forms?")
        
    # Query-based followups
    if any(word in q for word in ["employee", "staff", "worker"]):
        base_followups.append("Do you need information about employee procedures?")
    elif any(word in q for word in ["time", "schedule", "hours"]):
        base_followups.append("Would you like details about scheduling policies?")
    elif any(word in q for word in ["customer", "client"]):
        base_followups.append("Do you need customer service procedures?")
    
    # Default fallbacks
    if not base_followups:
        base_followups.extend([
            "Do you want to know more details?",
            "Would you like steps for a related task?",
            "Need help finding a specific document?"
        ])
    
    return base_followups[:3]

def is_vague(query): 
    """Detect vague queries"""
    if not query or len(query.strip()) < 3:
        return True
        
    if len(query.split()) < 2:
        return True
    
    greetings = ['hello', 'hi', 'hey', 'good morning', 'good afternoon']
    if any(greeting in query.lower() for greeting in greetings) and '?' not in query:
        return True
        
    return False

def expand_query_with_synonyms(query):
    """Expand query with synonyms"""
    synonyms = {
        "angry": "angry upset difficult frustrated mad complaining",
        "customer": "customer client guest patron visitor buyer",
        "handle": "handle deal manage respond address",
        "procedure": "procedure process protocol steps method",
        "policy": "policy rule guideline standard regulation",
        "refund": "refund return money back exchange reimbursement",
        "cash": "cash money payment transaction funds",
        "first day": "first day onboarding orientation training new employee start"
    }
    
    expanded = query
    for word, expansion in synonyms.items():
        if word in query.lower():
            expanded += f" {expansion}"
    
    return expanded

# ---- Embedding Worker ----
def embed_sop_worker(fpath, metadata=None):
    """Background worker for document embedding"""
    fname = os.path.basename(fpath)
    try:
        ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""
        
        # Load documents based on file type
        if ext == "docx":
            docs = UnstructuredWordDocumentLoader(fpath).load()
        elif ext == "pdf":
            docs = PyPDFLoader(fpath).load()
        elif ext == "txt":  # Support for demo documents
            with open(fpath, 'r', encoding='utf-8') as f:
                content = f.read()
            from langchain.schema import Document
            docs = [Document(page_content=content, metadata={"source": fpath})]
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        
        # Enhanced chunking
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]
        ).split_documents(docs)
        
        # Add metadata
        for i, chunk in enumerate(chunks):
            content_lower = chunk.page_content.lower()
            
            # Detect content type
            if any(word in content_lower for word in ["angry", "upset", "difficult", "complaint"]):
                chunk_type = "customer_service"
            elif any(word in content_lower for word in ["cash", "money", "payment", "refund"]):
                chunk_type = "financial"
            elif any(word in content_lower for word in ["first day", "onboard", "training"]):
                chunk_type = "onboarding"
            else:
                chunk_type = "general"
            
            # Extract keywords
            words = re.findall(r'\b[a-zA-Z]{4,}\b', chunk.page_content.lower())
            stopwords = {'that', 'this', 'with', 'they', 'have', 'will', 'from', 'been', 'were'}
            keywords = [word for word in set(words) if word not in stopwords][:5]
            keywords_string = " ".join(keywords)

            chunk.metadata.update({
                **(metadata or {}),
                "chunk_id": f"{fname}_{i}",
                "chunk_type": chunk_type,
                "keywords": keywords_string
            })
       
        # Load or create vector database
        db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding)
        db.add_documents(chunks)
        db.persist()
        
        logger.info(f"Successfully embedded {len(chunks)} chunks from {fname}")
        update_status(fname, {"status": "embedded", "chunk_count": len(chunks), **(metadata or {})})
        
    except Exception as e:
        logger.error(f"Error embedding {fname}: {e}")
        update_status(fname, {"status": f"error: {str(e)}", **(metadata or {})})

def update_status(filename, status):
    """Update document processing status with error handling"""
    try:
        data = json.load(open(STATUS_FILE)) if os.path.exists(STATUS_FILE) else {}
        data[filename] = {**status, "updated_at": time.time()}
        with open(STATUS_FILE, "w") as f: 
            json.dump(data, f, indent=2)
        logger.info(f"Updated status for {filename}: {status.get('status', 'unknown')}")
    except Exception as e:
        logger.error(f"Error updating status for {filename}: {e}")

def load_vectorstore():
    """Load the vector database with error handling"""
    global vectorstore
    try:
        vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding)
        logger.info("Vector store loaded successfully")
        
        # Test the vectorstore
        test_results = vectorstore.similarity_search("test", k=1)
        logger.info(f"Vectorstore test: found {len(test_results)} results")
        
    except Exception as e:
        logger.error(f"Error loading vector store: {e}")
        vectorstore = None

def ensure_vectorstore():
    """Ensure vectorstore is available"""
    global vectorstore
    try:
        if not vectorstore:
            load_vectorstore()
        return vectorstore is not None
    except Exception as e:
        logger.error(f"Vectorstore health check failed: {e}")
        load_vectorstore()
        return vectorstore is not None

# ---- Routes ----
@app.route("/")
def home(): 
    return safe_json_response({
        "status": "ok", 
        "message": "🚀 OpsVoice RAG API is live!",
        "version": "1.5.0-production-fixed",
        "features": ["session_memory", "smart_truncation", "security_enhanced", "data_persistence", "demo_mode"],
        "data_path": DATA_PATH,
        "persistent_storage": os.path.exists("/data")
    })

@app.route("/healthz")
def healthz(): 
    return safe_json_response({
        "status": "healthy",
        "timestamp": time.time(),
        "vectorstore": "loaded" if vectorstore else "not_loaded",
        "cache_size": len(query_cache),
        "active_sessions": len(conversation_sessions),
        "avg_response_time": performance_metrics.get("avg_response_time", 0),
        "data_path": DATA_PATH,
        "persistent_storage": os.path.exists("/data"),
        "sop_files_count": len(glob.glob(os.path.join(SOP_FOLDER, "*.*")))
    })

@app.route("/metrics")
def get_metrics():
    """Get performance metrics"""
    return safe_json_response(performance_metrics)

@app.route("/upload-sop", methods=["POST", "OPTIONS"])
def upload_sop():
    """Upload and process documents with enhanced security"""
    if request.method == "OPTIONS":
        return "", 204
        
    try:
        file = request.files.get("file")
        is_valid, result = validate_file_upload(file)
        if not is_valid:
            return safe_json_response({"error": result}, 400)
        
        filename = result
        
        tenant = request.form.get("company_id_slug", "").strip()
        if not validate_company_id(tenant):
            return safe_json_response({"error": "Invalid company identifier"}, 400)
        
        if not check_rate_limit(tenant):
            return safe_json_response({"error": "Too many requests - please wait"}, 429)
        
        title = request.form.get("doc_title", filename)[:100]
        title = re.sub(r'[<>"\']', '', title)
        
        # Create secure filename with timestamp
        timestamp = int(time.time())
        safe_filename = f"{tenant}_{timestamp}_{filename}"
        save_path = os.path.join(SOP_FOLDER, safe_filename)
        
        # Save file with error handling
        try:
            file.save(save_path)
        except Exception as e:
            logger.error(f"File save error: {e}")
            return safe_json_response({"error": "File save failed"}, 500)
        
        # Verify file was saved correctly
        if not os.path.exists(save_path) or os.path.getsize(save_path) == 0:
            return safe_json_response({"error": "File save verification failed"}, 500)
        
        metadata = {
            "title": title,
            "company_id_slug": tenant,
            "filename": safe_filename,
            "uploaded_at": time.time(),
            "file_size": os.path.getsize(save_path)
        }
        
        update_status(safe_filename, {"status": "embedding...", **metadata})
        Thread(target=embed_sop_worker, args=(save_path, metadata), daemon=True).start()

        return safe_json_response({
            "message": "Document uploaded successfully",
            "doc_title": title,
            "company_id_slug": tenant,
            "sop_file_url": f"{request.host_url.rstrip('/')}/static/sop-files/{safe_filename}",
            "upload_id": secrets.token_hex(8)
        })
        
    except Exception as e:
        logger.error(f"Upload error: {traceback.format_exc()}")
        return safe_json_response({"error": "Upload failed"}, 500)

@app.route("/query", methods=["POST", "OPTIONS"])
def query_sop():
    """FIXED: Process text queries with proper error handling and response format"""
    if request.method == "OPTIONS":
        return "", 204
        
    start_time = time.time()
    
    try:
        # Parse request with better error handling
        try:
            payload = request.get_json(force=True, silent=True) or {}
        except Exception as e:
            logger.error(f"JSON parse error: {e}")
            return safe_json_response({
                "answer": "Invalid request format. Please try again.",
                "source": "error",
                "followups": [],
                "session_id": None
            }, 400)
        
        qtext = sanitize_text(payload.get("query", ""))
        tenant = payload.get("company_id_slug", "").strip()
        session_id = payload.get("session_id") or f"{tenant}_{int(time.time())}"
        session_id = re.sub(r'[^a-zA-Z0-9_\-\.]', '', str(session_id))[:64]

        # Validate inputs
        if not qtext or len(qtext.strip()) < 3:
            return safe_json_response({
                "answer": "Please ask a more specific question.",
                "source": "error",
                "followups": ["What procedure are you looking for?"],
                "session_id": session_id
            })
            
        if not validate_company_id(tenant):
            return safe_json_response({
                "answer": "Invalid company identifier.",
                "source": "error",
                "followups": [],
                "session_id": session_id
            }, 400)
            
        if not check_rate_limit(tenant):
            return safe_json_response({
                "answer": "Too many requests. Please wait a moment.",
                "source": "error",
                "followups": [],
                "session_id": session_id
            }, 429)

        # Check cache first
        cached_response = get_cached_response(qtext, tenant)
        if cached_response:
            cached_response["session_id"] = session_id
            update_metrics(time.time() - start_time, "cache")
            return safe_json_response(cached_response)

        # Ensure vectorstore is loaded
        if not ensure_vectorstore():
            return safe_json_response({
                "answer": "Service temporarily unavailable. Please try again in a moment.",
                "source": "error",
                "followups": ["Try again in a few seconds"],
                "session_id": session_id
            }, 503)

        # Handle vague queries
        if is_vague(qtext):
            response = {
                "answer": "Could you please be more specific? What procedure or policy are you looking for?",
                "source": "clarify",
                "followups": ["How do I handle customer complaints?", "What's the refund procedure?", "Where are the training documents?"],
                "session_id": session_id
            }
            update_metrics(time.time() - start_time, "clarify")
            return safe_json_response(response)

        # Filter off-topic queries
        off_topic_keywords = ["gmail", "facebook", "amazon", "weather", "news", "stock", "crypto", "youtube", "instagram"]
        if any(keyword in qtext.lower() for keyword in off_topic_keywords):
            response = {
                "answer": "Please ask questions about your company procedures and policies.",
                "source": "off_topic",
                "followups": ["What are our customer service procedures?", "How do I process a refund?", "What's the onboarding process?"],
                "session_id": session_id
            }
            update_metrics(time.time() - start_time, "off_topic")
            return safe_json_response(response)

        # FIXED: Process with RAG - Enhanced error handling
        try:
            complexity = get_query_complexity(qtext)
            optimal_llm = get_optimal_llm(complexity)
            expanded_query = expand_query_with_synonyms(qtext)
            
            logger.info(f"Processing query for {tenant}: {qtext[:50]}... (complexity: {complexity})")
            
            # Set up retriever with company filtering
            retriever = vectorstore.as_retriever(
                search_kwargs={
                    "k": 5,
                    "filter": {"company_id_slug": tenant}
                }
            )
            
            # Test retrieval first
            try:
                test_docs = retriever.get_relevant_documents(expanded_query)
                logger.info(f"Found {len(test_docs)} relevant documents for query")
                
                if not test_docs:
                    # Try without company filter for broader search
                    logger.info("No company-specific docs found, trying broader search...")
                    general_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
                    test_docs = general_retriever.get_relevant_documents(expanded_query)
                    logger.info(f"Broader search found {len(test_docs)} documents")
                    
            except Exception as retrieval_error:
                logger.error(f"Document retrieval error: {retrieval_error}")
                # Fall back to general response
                raise Exception("Document retrieval failed")
            
            # Get session memory
            memory = get_session_memory(session_id)
            
            # Create conversational chain with error handling
            try:
                qa = ConversationalRetrievalChain.from_llm(
                    optimal_llm,
                    retriever=retriever,
                    memory=memory,
                    return_source_documents=True
                )
                
                # Query the chain with timeout simulation
                result = qa.invoke({"question": expanded_query})
                
            except Exception as chain_error:
                logger.error(f"Chain invocation error: {chain_error}")
                raise Exception("AI processing failed")
            
            # Process result
            answer = clean_text(result.get("answer", ""))
            source_docs = result.get("source_documents", [])
            
            logger.info(f"Raw answer length: {len(answer)} chars, source docs: {len(source_docs)}")

            # FIXED: Better answer validation
            if answer and not is_unhelpful_answer(answer) and len(answer.strip()) > 10:
                # Smart truncation for long answers
                if len(answer.split()) > 150:
                    answer = smart_truncate(answer, 150)
                    
                response = {
                    "answer": answer,
                    "fallback_used": False,
                    "followups": generate_contextual_followups(qtext, answer),
                    "source": "sop",
                    "source_documents": len(source_docs),
                    "session_id": session_id,
                    "model_used": optimal_llm.model_name if hasattr(optimal_llm, 'model_name') else "unknown"
                }
                
                # Cache successful responses
                cache_response(qtext, tenant, response)
                update_metrics(time.time() - start_time, "sop")
                return safe_json_response(response)
                
        except Exception as rag_error:
            logger.error(f"RAG processing error: {rag_error}")
            # Continue to fallback instead of failing completely
        
        # FIXED: Enhanced fallback with company-specific responses
        logger.info(f"Using fallback for query: {qtext}")
        
        query_lower = qtext.lower()
        company_name = tenant.replace('-', ' ').title()
        
        if any(word in query_lower for word in ["angry", "upset", "customer", "complaint"]):
            fallback_answer = f"""I don't see specific customer service policies in your uploaded documents.

For handling difficult customers, here are general best practices:
• Listen actively and remain calm
• Acknowledge their concerns  
• Apologize for any inconvenience
• Focus on finding a solution
• Escalate to manager if needed

Please check your employee handbook or contact your supervisor for company-specific policies."""

        elif any(word in query_lower for word in ["cash", "money", "refund", "payment"]):
            fallback_answer = f"""I don't see specific financial procedures in your uploaded documents.

For financial processes, generally:
• Check with your manager for authorization limits
• Follow company refund policy procedures
• Contact accounting for guidance on payments
• Keep proper documentation for all transactions

Try asking about other topics from your uploaded company documents."""

        elif any(word in query_lower for word in ["onboard", "training", "first day", "new employee"]):
            fallback_answer = f"""I don't see specific onboarding procedures in your uploaded documents.

General onboarding best practices include:
• Welcome and introduction to team
• Provide company handbook and policies
• Set up workspace and accounts
• Schedule initial training sessions
• Assign a buddy or mentor

Please check with HR or your manager for your company's specific onboarding process."""

        else:
            fallback_answer = f"""I don't see specific information about that in your company documents.

This might be because:
• The document hasn't been uploaded yet
• It's covered under a different topic
• It requires manager approval

Try asking about procedures, policies, or customer service topics that might be in your uploaded documents."""
        
        response = {
            "answer": fallback_answer,
            "fallback_used": True,
            "followups": ["Can you be more specific?", "What department handles this?", "Try asking about a specific procedure"],
            "source": "fallback",
            "session_id": session_id
        }
        
        update_metrics(time.time() - start_time, "fallback")
        return safe_json_response(response)

    except Exception as e:
        logger.error(f"Query error: {traceback.format_exc()}")
        update_metrics(time.time() - start_time, "error")
        
        # Return user-friendly error with session preservation
        return safe_json_response({
            "answer": "I encountered an error processing your request. Please try asking your question differently.",
            "source": "error",
            "followups": ["Ask a different question", "Try a simpler query", "Check if your documents are uploaded"],
            "session_id": session_id,
            "error_details": str(e) if app.debug else None
        }, 500)

@app.route("/voice-reply", methods=["POST", "OPTIONS"])
def voice_reply():
    """FIXED: Convert text to speech with enhanced error handling"""
    if request.method == "OPTIONS":
        return "", 204

    try:
        try:
            payload = request.get_json(force=True, silent=True) or {}
        except Exception as e:
            logger.error(f"TTS JSON parse error: {e}")
            return safe_json_response({"error": "Invalid request format"}, 400)
        
        # Accept both 'query' and 'text' for compatibility
        text = payload.get("query") or payload.get("text", "")
        text = clean_text(text)
        
        if not text or len(text.strip()) < 1:
            return safe_json_response({"error": "No text provided"}, 400)
        
        tenant = payload.get("company_id_slug", "demo").strip()
        
        if not check_rate_limit(tenant):
            return safe_json_response({"error": "Rate limit exceeded"}, 429)
        
        # Limit text length for cost control
        tts_text = text[:500] if len(text) > 500 else text
        
        # Generate secure cache key
        content_hash = hashlib.md5(tts_text.encode()).hexdigest()
        cache_key = f"tts_{tenant}_{content_hash}.mp3"
        cache_path = os.path.join(AUDIO_CACHE_DIR, cache_key)
        
        # Serve from cache if available
        if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
            try:
                return send_file(cache_path, mimetype="audio/mp3", as_attachment=False)
            except Exception as e:
                logger.error(f"Cache serve error: {e}")
                # Remove corrupted cache file
                try:
                    os.remove(cache_path)
                except:
                    pass
        
        # Generate new audio
        elevenlabs_key = os.getenv("ELEVENLABS_API_KEY")
        if not elevenlabs_key:
            return safe_json_response({"error": "TTS service not configured"}, 503)
        
        try:
            tts_resp = requests.post(
                "https://api.elevenlabs.io/v1/text-to-speech/tnSpp4vdxKPjI9w0GnoV/stream",
                headers={
                    "xi-api-key": elevenlabs_key,
                    "Content-Type": "application/json"
                },
                json={
                    "text": tts_text,
                    "voice_settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.75,
                        "style": 0.0,
                        "use_speaker_boost": True
                    }
                },
                timeout=30
            )
            
            if tts_resp.status_code != 200:
                logger.error(f"ElevenLabs error: {tts_resp.status_code} - {tts_resp.text}")
                return safe_json_response({"error": "TTS generation failed"}, 502)
            
            # Validate response
            audio_data = tts_resp.content
            if len(audio_data) < 1000:  # Too small to be valid audio
                logger.error("TTS response too small")
                return safe_json_response({"error": "Invalid audio response"}, 502)
            
            # Cache the audio with error handling
            try:
                with open(cache_path, "wb") as f:
                    f.write(audio_data)
                logger.info(f"Cached TTS audio: {cache_key}")
            except Exception as cache_error:
                logger.error(f"TTS cache error: {cache_error}")
                # Continue without caching
            
            return send_file(io.BytesIO(audio_data), mimetype="audio/mp3", as_attachment=False)
            
        except requests.exceptions.Timeout:
            logger.error("TTS request timeout")
            return safe_json_response({"error": "TTS timeout"}, 504)
        except requests.exceptions.RequestException as e:
            logger.error(f"TTS request error: {e}")
            return safe_json_response({"error": "TTS service unavailable"}, 502)
        
    except Exception as e:
        logger.error(f"TTS error: {traceback.format_exc()}")
        return safe_json_response({"error": "TTS failed"}, 500)

@app.route("/company-docs/<company_id_slug>")
def company_docs(company_id_slug):
    """FIXED: Get documents with enhanced security and error handling"""
    # Validate company ID
    if not validate_company_id(company_id_slug):
        return safe_json_response({"error": "Invalid company identifier"}, 400)
    
    if not os.path.exists(STATUS_FILE): 
        return safe_json_response([])
    
    try:
        with open(STATUS_FILE, 'r') as f:
            data = json.load(f)
        
        company_docs = []
        
        for filename, metadata in data.items():
            if metadata.get("company_id_slug") == company_id_slug:
                # Security: validate filename before serving
                safe_filename = secure_filename(filename)
                if safe_filename != filename:
                    logger.warning(f"Unsafe filename detected: {filename}")
                    continue
                
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
        
        logger.info(f"Found {len(company_docs)} documents for {company_id_slug}")
        return safe_json_response(company_docs)
        
    except Exception as e:
        logger.error(f"Error fetching docs for {company_id_slug}: {e}")
        return safe_json_response({"error": "Failed to fetch documents"}, 500)

@app.route("/continue", methods=["POST", "OPTIONS"])
def continue_conversation():
    """FIXED: Continue conversation with proper error handling"""
    if request.method == "OPTIONS":
        return "", 204
        
    try:
        try:
            payload = request.get_json(force=True, silent=True) or {}
        except Exception as e:
            logger.error(f"Continue JSON parse error: {e}")
            return safe_json_response({"error": "Invalid request format"}, 400)
        
        session_id = payload.get("session_id", "").strip()
        tenant = payload.get("company_id_slug", "").strip()
        
        if not session_id or not validate_company_id(tenant):
            return safe_json_response({"error": "Invalid session or company"}, 400)
        
        session_id = re.sub(r'[^a-zA-Z0-9_\-\.]', '', session_id)[:64]
        
        if session_id not in conversation_sessions:
            return safe_json_response({
                "answer": "No conversation session found. Please start a new conversation.",
                "source": "error",
                "session_id": session_id,
                "followups": ["Ask a new question"]
            }, 404)
        
        memory = conversation_sessions[session_id]
        
        # Get last response from memory
        last_response = ""
        try:
            if hasattr(memory, 'chat_memory') and hasattr(memory.chat_memory, 'messages'):
                for msg in reversed(memory.chat_memory.messages):
                    if hasattr(msg, 'content') and len(msg.content) > 50:
                        last_response = msg.content
                        break
        except Exception as e:
            logger.error(f"Memory access error: {e}")
        
        if last_response and ("Would you like me to continue" in last_response or "Should I continue" in last_response):
            # Generate continuation by calling query endpoint internally
            continue_query = "Please continue from where you left off with the rest of the details."
            
            # Create internal request context
            with app.test_request_context('/query', json={
                "query": continue_query,
                "company_id_slug": tenant,
                "session_id": session_id
            }, method='POST'):
                return query_sop()
        else:
            return safe_json_response({
                "answer": "I don't see a previous response to continue from. What would you like to know?",
                "source": "clarification",
                "session_id": session_id,
                "followups": ["Ask a new question", "What procedures do you need?"]
            })
            
    except Exception as e:
        logger.error(f"Continue error: {traceback.format_exc()}")
        return safe_json_response({"error": "Continue failed"}, 500)

@app.route("/session-info/<session_id>")
def get_session_info(session_id):
    """Get session information with security validation"""
    # Validate and sanitize session ID
    session_id = re.sub(r'[^a-zA-Z0-9_\-\.]', '', session_id)[:64]
    
    if not session_id:
        return safe_json_response({"error": "Invalid session ID"}, 400)
    
    if session_id in conversation_sessions:
        memory = conversation_sessions[session_id]
        messages = []
        
        try:
            if hasattr(memory, 'chat_memory') and hasattr(memory.chat_memory, 'messages'):
                messages = [
                    {
                        "type": type(msg).__name__, 
                        "content": msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                    } 
                    for msg in memory.chat_memory.messages
                ]
        except Exception as e:
            logger.error(f"Session info error: {e}")
            messages = []
        
        return safe_json_response({
            "session_id": session_id,
            "message_count": len(messages),
            "messages": messages,
            "last_accessed": getattr(memory, 'last_accessed', 0)
        })
    else:
        return safe_json_response({"error": "Session not found"}, 404)

@app.route("/clear-cache", methods=["POST"])
def clear_cache():
    """Clear caches with enhanced error handling"""
    global query_cache
    try:
        cache_size = len(query_cache)
        query_cache.clear()
        
        audio_files_cleared = 0
        audio_errors = 0
        
        if os.path.exists(AUDIO_CACHE_DIR):
            for filename in os.listdir(AUDIO_CACHE_DIR):
                if filename.endswith('.mp3'):
                    try:
                        os.remove(os.path.join(AUDIO_CACHE_DIR, filename))
                        audio_files_cleared += 1
                    except Exception as e:
                        logger.error(f"Error removing audio file {filename}: {e}")
                        audio_errors += 1
        
        return safe_json_response({
            "message": "Cache cleared successfully",
            "query_cache_cleared": cache_size,
            "audio_files_cleared": audio_files_cleared,
            "audio_errors": audio_errors
        })
    except Exception as e:
        logger.error(f"Cache clear error: {e}")
        return safe_json_response({"error": "Cache clear failed"}, 500)

@app.route("/clear-sessions", methods=["POST"])
def clear_sessions():
    """Clear conversation sessions"""
    global conversation_sessions
    session_count = len(conversation_sessions)
    conversation_sessions.clear()
    return safe_json_response({
        "message": f"Cleared {session_count} conversation sessions"
    })

@app.route("/static/sop-files/<path:filename>")
def serve_sop(filename):
    """FIXED: Serve SOP files with enhanced security"""
    # Security: validate filename
    safe_filename = secure_filename(filename)
    if safe_filename != filename:
        logger.warning(f"Attempted access to unsafe filename: {filename}")
        return safe_json_response({"error": "Invalid filename"}, 400)
    
    # Check if file exists
    file_path = os.path.join(SOP_FOLDER, safe_filename)
    if not os.path.exists(file_path):
        return safe_json_response({"error": "File not found"}, 404)
    
    try:
        return send_from_directory(SOP_FOLDER, safe_filename)
    except Exception as e:
        logger.error(f"File serve error: {e}")
        return safe_json_response({"error": "File serve failed"}, 500)

# ---- Admin/Debug Routes ----
@app.route("/admin/status")
def admin_status():
    """Admin endpoint for system status"""
    return safe_json_response({
        "system": {
            "data_path": DATA_PATH,
            "persistent_storage": os.path.exists("/data"),
            "vectorstore_loaded": vectorstore is not None,
            "total_files": len(glob.glob(os.path.join(SOP_FOLDER, "*.*"))),
            "cache_size": len(query_cache),
            "active_sessions": len(conversation_sessions),
        },
        "metrics": performance_metrics,
        "demo_business": {
            "files": len([f for f in os.listdir(SOP_FOLDER) if f.startswith("demo-business-123_")]),
            "status": "active"
        }
    })

@app.route("/admin/reload-vectorstore", methods=["POST"])
def admin_reload_vectorstore():
    """Admin endpoint to reload vectorstore"""
    try:
        load_vectorstore()
        return safe_json_response({
            "message": "Vectorstore reloaded successfully",
            "loaded": vectorstore is not None
        })
    except Exception as e:
        logger.error(f"Vectorstore reload error: {e}")
        return safe_json_response({"error": "Reload failed"}, 500)

# ---- Error Handlers ----
@app.errorhandler(404)
def not_found(error):
    return safe_json_response({"error": "Endpoint not found"}, 404)

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return safe_json_response({"error": "Internal server error"}, 500)

@app.errorhandler(413)
def payload_too_large(error):
    return safe_json_response({"error": "Request too large"}, 413)

# ---- Startup Functions ----
def startup_checks():
    """Perform startup checks and initialization"""
    logger.info("=== OpsVoice API Startup ===")
    logger.info(f"Data path: {DATA_PATH}")
    logger.info(f"Persistent storage: {os.path.exists('/data')}")
    
    # Ensure demo documents exist
    ensure_demo_documents()
    
    # Load vectorstore
    logger.info("Loading vector store...")
    load_vectorstore()
    
    # Check existing files
    existing_files = glob.glob(os.path.join(SOP_FOLDER, "*.*"))
    logger.info(f"Found {len(existing_files)} existing files")
    
    # Load existing metrics if available
    metrics_file = os.path.join(DATA_PATH, "metrics.json")
    try:
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                saved_metrics = json.load(f)
                performance_metrics.update(saved_metrics)
                logger.info(f"Loaded existing metrics: {performance_metrics['total_queries']} total queries")
    except Exception as e:
        logger.error(f"Could not load metrics: {e}")
    
    logger.info("=== Startup Complete ===")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    
    # Perform startup checks
    startup_checks()
    
    logger.info(f"Starting OpsVoice API v1.5.0 on port {port}")
    logger.info("Features: Data Persistence, Enhanced Security, Session Memory, Smart Truncation")
    app.run(host="0.0.0.0", port=port, debug=False)