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

load_dotenv()

# ---- MVP SECURITY ADDITIONS ----
# Session memory storage
conversation_sessions = {}

# Simple rate limiting for MVP
rate_limits = {}
MAX_REQUESTS_PER_MINUTE = 20  # Generous for MVP testing
MAX_CACHE_SIZE = 200  # Reasonable for MVP

# File upload security
ALLOWED_EXTENSIONS = {'pdf', 'docx'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# ---- Paths & Setup ----
DATA_PATH = "/opt/render/project/src/data"
SOP_FOLDER = os.path.join(DATA_PATH, "sop-files")
CHROMA_DIR = os.path.join(DATA_PATH, "chroma_db")
AUDIO_CACHE_DIR = os.path.join(DATA_PATH, "audio_cache")
STATUS_FILE = os.path.join(SOP_FOLDER, "status.json")

# Ensure directories exist (without deleting existing data)
os.makedirs(SOP_FOLDER, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)
os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)

# Performance tracking
performance_metrics = {
    "total_queries": 0,
    "cache_hits": 0,
    "avg_response_time": 0,
    "model_usage": {"gpt-3.5-turbo": 0, "gpt-4": 0},
    "response_sources": {"sop": 0, "fallback": 0, "cache": 0}
}

# Enhanced caching & memory
query_cache = {}

embedding = OpenAIEmbeddings()
vectorstore = None

# ---- Flask Setup ----
app = Flask(__name__)
CORS(app, origins=["https://opsvoice-widget.vercel.app", "http://localhost:3000", "*"], supports_credentials=True)

@app.after_request
def add_cors(response):
    origin = request.headers.get("Origin", "")
    allowed_origins = ["https://opsvoice-widget.vercel.app", "http://localhost:3000"]
    
    # Allow all origins for debugging, but log them
    response.headers["Access-Control-Allow-Origin"] = origin or "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    return response

# ---- MVP SECURITY FUNCTIONS ----
def validate_file_upload(file):
    """Basic file validation for MVP"""
    if not file or not file.filename:
        return False, "No file uploaded"
    
    # Use secure_filename for safety
    filename = secure_filename(file.filename)
    if not filename:
        return False, "Invalid filename"
    
    # Check extension
    if '.' not in filename:
        return False, "File must have extension"
    
    ext = filename.rsplit('.', 1)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Only PDF and DOCX files allowed, got .{ext}"
    
    # Check file size
    file.seek(0, 2)
    size = file.tell()
    file.seek(0)
    if size > MAX_FILE_SIZE:
        return False, f"File too large (max 10MB)"
    
    return True, filename

def check_rate_limit_mvp(tenant: str) -> bool:
    """Simple rate limiting for MVP"""
    global rate_limits
    
    current_minute = int(time.time() // 60)
    key = f"{tenant}:{current_minute}"
    
    # Clean old entries (keep only last 5 minutes)
    cutoff = current_minute - 5
    rate_limits = {k: v for k, v in rate_limits.items() 
                   if int(k.split(':')[1]) > cutoff}
    
    # Check current minute rate
    count = rate_limits.get(key, 0)
    if count >= MAX_REQUESTS_PER_MINUTE:
        return False
    
    rate_limits[key] = count + 1
    return True

def validate_company_id_mvp(company_id: str) -> bool:
    """Basic company ID validation for MVP"""
    if not company_id or len(company_id) < 3:
        return False
    
    # Allow alphanumeric, hyphens, underscores only
    if not re.match(r'^[a-zA-Z0-9_-]+$', company_id):
        return False
    
    # Prevent path traversal
    if '..' in company_id or '/' in company_id:
        return False
    
    return True

def sanitize_query_mvp(query: str) -> str:
    """Basic query sanitization for MVP"""
    if not query:
        return ""
    
    # Remove dangerous characters but keep normal punctuation
    query = re.sub(r'[<>"\'\x00\r\n]', '', query)
    
    # Limit length
    return query[:500].strip()

def safe_error_mvp(message: str, status_code: int = 500):
    """Safe error responses for MVP"""
    safe_messages = {
        400: "Invalid request",
        403: "Access denied", 
        413: "File too large",
        429: "Too many requests - please wait a moment",
        500: "Service temporarily unavailable"
    }
    
    return jsonify({
        "error": safe_messages.get(status_code, "Service error"),
        "timestamp": int(time.time())
    }), status_code

def intelligent_query_expansion(query):
    """Enhanced query expansion with AI"""
    # This function should expand/improve the query using AI
    # For now, return the original query if AI expansion fails
    return query

# ---- Enhanced Utility Functions ----
def clean_text(txt: str) -> str:
    """Clean text for TTS - remove problematic characters"""
    if not txt:
        return ""
    txt = re.sub(r"\s+", " ", txt)
    txt = txt.replace("\u2022", "-").replace("\t", " ")
    txt = re.sub(r"[*#]+", "", txt)
    txt = re.sub(r"\[.*?\]", "", txt)
    txt = txt.strip()
    return txt

def smart_truncate(text, max_words=150):
    """Smart truncation that preserves complete sentences"""
    if not text:
        return text
    
    words = text.split()
    if len(words) <= max_words:
        return text
    
    # Find the last complete sentence within limit
    truncated = " ".join(words[:max_words])
    
    # Find last sentence ending
    sentence_endings = ['.', '!', '?', ':']
    last_ending = -1
    
    for ending in sentence_endings:
        pos = truncated.rfind(ending)
        if pos > last_ending:
            last_ending = pos
    
    # If we found a good sentence break
    if last_ending > len(truncated) * 0.6:
        return truncated[:last_ending + 1] + " Would you like me to continue with more details?"
    else:
        # Fallback to word truncation with better ending
        return " ".join(words[:130]) + "... Should I continue with the rest of the details?"

def get_session_memory(session_id):
    """Get or create conversation memory for session"""
    global conversation_sessions
    
    if session_id not in conversation_sessions:
        conversation_sessions[session_id] = ConversationBufferMemory(
            memory_key="chat_history", 
            return_messages=True,
            output_key="answer"
        )
    return conversation_sessions[session_id]

def get_query_complexity(query: str) -> str:
    """Determine if query is simple or complex for model selection"""
    words = query.lower().split()
    
    # Simple query indicators
    simple_indicators = [
        len(words) <= 8,
        any(word in query.lower() for word in ['what', 'when', 'where', 'who', 'how many']),
        query.endswith('?') and len(words) <= 6
    ]
    
    # Complex query indicators  
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
    """Select optimal LLM based on query complexity for performance"""
    global performance_metrics
    
    if complexity == "simple":
        performance_metrics["model_usage"]["gpt-3.5-turbo"] += 1
        return ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    else:
        performance_metrics["model_usage"]["gpt-4"] += 1
        return ChatOpenAI(temperature=0, model="gpt-4")

def get_cache_key(query: str, company_id: str) -> str:
    """Generate cache key for query"""
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
    """Cache response for future use"""
    cache_key = get_cache_key(query, company_id)
    query_cache[cache_key] = {
        'response': response,
        'timestamp': time.time()
    }
    
    # Simple cache size management
    if len(query_cache) > MAX_CACHE_SIZE:
        # Remove oldest entries
        oldest_keys = sorted(query_cache.keys(), key=lambda k: query_cache[k]['timestamp'])[:50]
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
    """Enhanced unhelpful answer detection"""
    if not text or not text.strip(): 
        return True
        
    low = text.lower()
    triggers = ["don't know", "no information", "i'm not sure", "sorry", 
                "unavailable", "not covered", "cannot find", "no specific information"]
    
    has_trigger = any(t in low for t in triggers)
    is_too_short = len(low.split()) < 8
    
    return has_trigger or is_too_short

def generate_contextual_followups(query: str, answer: str) -> list:
    """Generate smarter contextual follow-up questions"""
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
    """Enhanced vague query detection"""
    if not query or len(query.strip()) < 3:
        return True
        
    # Very short queries without context
    if len(query.split()) < 2:
        return True
    
    # Greetings without questions
    greetings = ['hello', 'hi', 'hey', 'good morning', 'good afternoon']
    if any(greeting in query.lower() for greeting in greetings) and '?' not in query:
        return True
        
    return False

def expand_query_with_synonyms(query):
    """Expand query with common synonyms for better matching"""
    synonyms = {
        "angry": "angry upset difficult frustrated mad",
        "customer": "customer client guest patron",
        "handle": "handle deal manage respond",
        "procedure": "procedure process protocol steps",
        "policy": "policy rule guideline standard",
        "refund": "refund return money back exchange",
        "cash": "cash money payment transaction",
        "first day": "first day onboarding orientation training new employee"
    }
    
    expanded = query
    for word, expansion in synonyms.items():
        if word in query.lower():
            expanded += f" {expansion}"
    
    return expanded

# ---- Embedding Worker ----
def embed_sop_worker(fpath, metadata=None):
    """Enhanced embedding worker with better chunking"""
    fname = os.path.basename(fpath)
    try:
        ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""
        if ext == "docx":
            docs = UnstructuredWordDocumentLoader(fpath).load()
        elif ext == "pdf":
            docs = PyPDFLoader(fpath).load()
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        
        # Enhanced chunking for better context
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Increased for better context
            chunk_overlap=200,  # Increased overlap
            separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]
        ).split_documents(docs)
        
        # Add enhanced metadata
        for i, chunk in enumerate(chunks):
            # Detect content type
            content_lower = chunk.page_content.lower()
            if any(word in content_lower for word in ["angry", "upset", "difficult", "complaint"]):
                chunk_type = "customer_service"
            elif any(word in content_lower for word in ["cash", "money", "payment", "refund"]):
                chunk_type = "financial"
            elif any(word in content_lower for word in ["first day", "onboard", "training"]):
                chunk_type = "onboarding"
            else:
                chunk_type = "general"
            
            # Extract keywords as string (not list - prevents ChromaDB error)
            words = re.findall(r'\b[a-zA-Z]{4,}\b', chunk.page_content.lower())
            stopwords = {'that', 'this', 'with', 'they', 'have', 'will', 'from', 'been', 'were'}
            keywords = [word for word in set(words) if word not in stopwords][:5]
            keywords_string = " ".join(keywords)  # Convert to string for ChromaDB

            chunk.metadata.update({
                **(metadata or {}),
                "chunk_id": f"{fname}_{i}",
                "chunk_type": chunk_type,
                "keywords": keywords_string  # String, not list
            })
       
        db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding)
        db.add_documents(chunks)
        db.persist()
        
        print(f"[EMBED] Successfully embedded {len(chunks)} chunks from {fname}")
        update_status(fname, {"status": "embedded", "chunk_count": len(chunks), **(metadata or {})})
        
    except Exception as e:
        print(f"[EMBED] Error with {fname}: {e}")
        update_status(fname, {"status": f"error: {str(e)}", **(metadata or {})})

def update_status(filename, status):
    """Update document processing status"""
    try:
        data = json.load(open(STATUS_FILE)) if os.path.exists(STATUS_FILE) else {}
        data[filename] = {**status, "updated_at": time.time()}
        with open(STATUS_FILE, "w") as f: 
            json.dump(data, f, indent=2)
        print(f"[STATUS] Updated {filename}: {status.get('status', 'unknown')}")
    except Exception as e:
        print(f"[STATUS] Error updating {filename}: {e}")

def load_vectorstore():
    """Load the vector database"""
    global vectorstore
    try:
        vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding)
        print("[DB] Vector store loaded successfully")
    except Exception as e:
        print(f"[DB] Error loading vector store: {e}")
        vectorstore = None

def ensure_vectorstore():
    """Ensure vectorstore is available and healthy"""
    global vectorstore
    try:
        if not vectorstore:
            load_vectorstore()
        return vectorstore is not None
    except Exception as e:
        print(f"[DB] Vectorstore health check failed: {e}")
        load_vectorstore()
        return vectorstore is not None

# ---- Routes ----
@app.route("/")
def home(): 
    return jsonify({
        "status": "ok", 
        "message": "🚀 OpsVoice RAG API is live!",
        "version": "1.3.0-production",
        "features": ["session_memory", "smart_truncation", "performance_optimization", "mvp_security", "n8n_compatible"]
    })

@app.route("/healthz")
def healthz(): 
    return jsonify({
        "status": "healthy",
        "timestamp": time.time(),
        "vectorstore": "loaded" if vectorstore else "not_loaded",
        "cache_size": len(query_cache),
        "active_sessions": len(conversation_sessions),
        "avg_response_time": performance_metrics["avg_response_time"]
    })

@app.route("/metrics")
def get_metrics():
    """Get performance metrics"""
    return jsonify(performance_metrics)

@app.route("/list-sops")
def list_sops():
    """List all uploaded SOP files"""
    docs = glob.glob(os.path.join(SOP_FOLDER, "*.docx")) + glob.glob(os.path.join(SOP_FOLDER, "*.pdf"))
    return jsonify({"files": docs, "count": len(docs)})

@app.route("/static/sop-files/<path:filename>")
def serve_sop(filename): 
    """Serve SOP files with basic security"""
    # Basic security: only allow safe filenames
    safe_filename = secure_filename(filename)
    if safe_filename != filename:
        return safe_error_mvp("Invalid filename", 400)
    
    return send_from_directory(SOP_FOLDER, safe_filename)

@app.route("/upload-sop", methods=["POST"])
def upload_sop():
    """Secure document upload with MVP validation"""
    try:
        # Validate file upload
        file = request.files.get("file")
        is_valid, result = validate_file_upload(file)
        if not is_valid:
            return safe_error_mvp(result, 400)
        
        filename = result
        
        # Validate and sanitize company ID
        tenant = request.form.get("company_id_slug", "").strip()
        if not validate_company_id_mvp(tenant):
            return safe_error_mvp("Invalid company identifier", 400)
        
        # Rate limiting
        if not check_rate_limit_mvp(tenant):
            return safe_error_mvp("Too many requests - please wait a moment", 429)
        
        # Get and sanitize title
        title = request.form.get("doc_title", filename)[:100]
        title = re.sub(r'[<>"\']', '', title)  # Remove dangerous chars
        
        # Create secure filename with timestamp to avoid conflicts
        timestamp = int(time.time())
        safe_filename = f"{tenant}_{timestamp}_{filename}"
        save_path = os.path.join(SOP_FOLDER, safe_filename)
        
        # Save file
        file.save(save_path)
        
        # Prepare metadata
        metadata = {
            "title": title,
            "company_id_slug": tenant,
            "filename": safe_filename,
            "uploaded_at": time.time(),
            "file_size": os.path.getsize(save_path)
        }
        
        # Start background embedding
        update_status(safe_filename, {"status": "embedding...", **metadata})
        Thread(target=embed_sop_worker, args=(save_path, metadata), daemon=True).start()

        return jsonify({
            "message": f"Document uploaded successfully",
            "doc_title": title,
            "company_id_slug": tenant,
            "sop_file_url": f"{request.host_url.rstrip('/')}/static/sop-files/{safe_filename}",
            "upload_id": secrets.token_hex(8)
        })
        
    except Exception as e:
        print(f"[UPLOAD ERROR] {e}")
        return safe_error_mvp("Upload failed", 500)

@app.route("/query", methods=["POST", "OPTIONS"])
def query_sop():
    """Enhanced query processing with widget/n8n compatibility for voice and text Ask"""
    if request.method == "OPTIONS":
        return "", 204
        
    start_time = time.time()
    try:
        payload = {}
        if request.is_json:
            payload = request.get_json()
        if not payload:
            try: payload = request.get_json(force=True)
            except: pass
        if not payload and request.data:
            try: payload = json.loads(request.data.decode('utf-8'))
            except: pass

        if not payload:
            print("[QUERY] No payload found after all attempts")
            return jsonify({"error": "No data received"}), 400
        
        qtext = sanitize_query_mvp(str(payload.get("query", "")))
        tenant = str(payload.get("company_id_slug", "")).strip()
        session_id = payload.get("session_id") or f"{tenant}_{int(time.time())}"
        session_id = re.sub(r'[^a-zA-Z0-9_\-\.]', '', str(session_id))[:64]

        # Check/validate input
        if not qtext or len(qtext.strip()) < 3:
            return jsonify({"answer": "Please type a longer question.", "source": "error", "followups": []}), 400
        if not validate_company_id_mvp(tenant):
            return jsonify({"answer": "Missing or invalid company.", "source": "error", "followups": []}), 400
        if not check_rate_limit_mvp(tenant):
            return jsonify({"answer": "Too many requests. Wait and try again.", "source": "error", "followups": []}), 429

        # 1. Check cache first
        cached_response = get_cached_response(qtext, tenant)
        if cached_response:
            cached_response["session_id"] = session_id
            update_metrics(time.time() - start_time, "cache")
            return jsonify(cached_response)

        # 2. Ensure vectorstore is loaded
        if not ensure_vectorstore():
            return jsonify({
                "answer": "Service temporarily unavailable. Try again shortly.",
                "source": "error",
                "followups": []
            }), 503

        # 3. Handle vague queries
        if is_vague(qtext):
            response = {
                "answer": "Please provide more specific details about what you're looking for.",
                "source": "clarify",
                "followups": ["Can you describe your task more clearly?"],
                "session_id": session_id
            }
            update_metrics(time.time() - start_time, "clarify")
            return jsonify(response)

        # 4. Filter off-topic queries
        off_topic_keywords = ["gmail", "facebook", "amazon", "weather", "news", "stock", "crypto"]
        if any(keyword in qtext.lower() for keyword in off_topic_keywords):
            response = {
                "answer": "Please ask questions related to your company procedures and policies.",
                "source": "off_topic",
                "followups": ["Ask about procedures", "Ask about training"],
                "session_id": session_id
            }
            update_metrics(time.time() - start_time, "off_topic")
            return jsonify(response)

        # 5. RAG Model selection and query
        complexity = get_query_complexity(qtext)
        optimal_llm = get_optimal_llm(complexity)
        try:
            expanded_query = expand_query_with_synonyms(qtext)
        except Exception as e:
            print(f"[QUERY] Expansion failed: {e}")
            expanded_query = qtext

        retriever = vectorstore.as_retriever(
            search_kwargs={
                "k": 5,
                "filter": {"company_id_slug": tenant}
            }
        )
        memory = get_session_memory(session_id)
        qa = ConversationalRetrievalChain.from_llm(
            optimal_llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True
        )
        result = qa.invoke({"question": expanded_query})
        answer = clean_text(result.get("answer", ""))

        if answer and not is_unhelpful_answer(answer):
            if len(answer.split()) > 150:
                answer = smart_truncate(answer, 150)
            response = {
                "answer": answer,
                "fallback_used": False,
                "followups": generate_contextual_followups(qtext, answer),
                "source": "sop",
                "source_documents": len(result.get("source_documents", [])),
                "session_id": session_id,
                "model_used": optimal_llm.model_name
            }
            cache_response(qtext, tenant, response)
            update_metrics(time.time() - start_time, "sop")
            return jsonify(response)
        else:
            # fallback (show text, followups always)
            fallback_answer = """I don't see information about that in your company documents.
This might be because:
- The document hasn't been uploaded yet
- It's covered under a different topic
- It requires manager approval
Try asking about procedures, policies, or customer service topics that might be in your uploaded documents."""
            response = {
                "answer": fallback_answer,
                "fallback_used": True,
                "followups": ["Can you be more specific?", "What department handles this?", "Try asking about a specific procedure"],
                "source": "fallback",
                "session_id": session_id
            }
            update_metrics(time.time() - start_time, "fallback")
            return jsonify(response)

    except Exception as e:
        print(f"[QUERY ERROR] Full trace: {traceback.format_exc()}")
        return jsonify({
            "answer": "Error processing your request. Please try again.",
            "source": "error",
            "followups": [],
            "debug": str(e)
        }), 500

@app.route("/debug-request", methods=["POST"])
def debug_request():
    """Debug endpoint to see what n8n is sending"""
    return jsonify({
        "headers": dict(request.headers),
        "content_type": request.content_type,
        "is_json": request.is_json,
        "data": request.data.decode() if request.data else None,
        "json": request.get_json(force=True, silent=True),
        "form": dict(request.form) if request.form else None,
        "args": dict(request.args) if request.args else None
    })

@app.route("/voice-reply", methods=["POST", "OPTIONS"])
def voice_reply():
    """Convert text to speech with enhanced security and caching"""
    # Handle CORS preflight
    if request.method == "OPTIONS":
        return "", 204

    try:
        # More flexible request handling
        payload = {}
        if request.is_json:
            payload = request.get_json()
        else:
            try:
                payload = request.get_json(force=True)
            except:
                payload = {}
        
        # Extract and validate text
        text = payload.get("text", "").strip()
        if not text:
            return jsonify({"error": "Missing text parameter"}), 400
        
        # Clean text for TTS
        text = clean_text(text)
        if not text:
            return jsonify({"error": "Text too short after cleaning"}), 400
        
        # Validate company ID
        tenant = payload.get("company_id_slug", "").strip()
        if tenant and not validate_company_id_mvp(tenant):
            return jsonify({"error": "Invalid company identifier"}), 400
        
        # Rate limiting for TTS
        if tenant and not check_rate_limit_mvp(tenant):
            return jsonify({"error": "Too many requests - please wait a moment"}), 429
        
        # Limit text length for TTS (cost control)
        tts_text = text[:500] if len(text) > 500 else text
        
        # Generate secure cache key
        content_hash = hashlib.md5(tts_text.encode()).hexdigest()
        cache_key = f"tts_{tenant}_{content_hash}.mp3"
        cache_path = os.path.join(AUDIO_CACHE_DIR, cache_key)
        
        # Serve from cache if available
        if os.path.exists(cache_path):
            return send_file(cache_path, mimetype="audio/mp3", as_attachment=False)
        
        # Generate new audio with API key from environment
        elevenlabs_key = os.getenv("ELEVENLABS_API_KEY")
        if not elevenlabs_key:
            return jsonify({"error": "TTS service not configured"}), 503
        
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
            return jsonify({"error": "TTS service error"}), 502
        
        # Cache the audio
        audio_data = tts_resp.content
        with open(cache_path, "wb") as f:
            f.write(audio_data)
        
        return send_file(io.BytesIO(audio_data), mimetype="audio/mp3", as_attachment=False)
        
    except requests.exceptions.Timeout:
        return jsonify({"error": "TTS request timeout"}), 504
    except Exception as e:
        print(f"[TTS ERROR] {e}")
        return jsonify({"error": "TTS request failed"}), 500

@app.route("/sop-status")
def sop_status():
    """Get document processing status"""
    if os.path.exists(STATUS_FILE): 
        return send_file(STATUS_FILE)
    return jsonify({})

@app.route("/company-docs/<company_id_slug>")
def company_docs(company_id_slug):
    """Get documents for a specific company with security validation"""
    # Validate company ID
    if not validate_company_id_mvp(company_id_slug):
        return jsonify({"error": "Invalid company identifier"}), 400
    
    if not os.path.exists(STATUS_FILE): 
        return jsonify([])
    
    try:
        data = json.load(open(STATUS_FILE))
        company_docs = []
        
        for filename, metadata in data.items():
            if metadata.get("company_id_slug") == company_id_slug:
                doc_info = {
                    "filename": filename,
                    "title": metadata.get("title", filename),
                    "status": metadata.get("status", "unknown"),
                    "company_id_slug": company_id_slug,
                    "uploaded_at": metadata.get("uploaded_at"),
                    "sop_file_url": f"{request.host_url}static/sop-files/{filename}"
                }
                company_docs.append(doc_info)
        
        print(f"[DOCS] Found {len(company_docs)} documents for {company_id_slug}")
        return jsonify(company_docs)
        
    except Exception as e:
        print(f"[DOCS] Error: {e}")
        return jsonify({"error": "Failed to fetch documents"}), 500

@app.route("/lookup-slug")
def lookup_slug():
    """Lookup company slug by email with basic validation"""
    email = request.args.get("email", "").strip().lower()
    
    # Basic email validation
    if not email or '@' not in email or len(email) < 5:
        return jsonify({"error": "Invalid email"}), 400
    
    if not os.path.exists(STATUS_FILE):
        return jsonify({"error": "No data available"}), 404

    try:
        data = json.load(open(STATUS_FILE))
        for metadata in data.values():
            if metadata.get("uploaded_by", "").strip().lower() == email:
                return jsonify({"slug": metadata.get("company_id_slug")})
        
        return jsonify({"error": "Not found"}), 404
    except Exception as e:
        print(f"[LOOKUP] Error: {e}")
        return jsonify({"error": "Lookup failed"}), 500

@app.route("/reload-db", methods=["POST"])
def reload_db():
    """Reload the vector database"""
    load_vectorstore()
    return jsonify({"message": "Vectorstore reloaded successfully."})

@app.route("/clear-cache", methods=["POST"])
def clear_cache():
    """Clear query and audio cache"""
    global query_cache
    try:
        # Clear query cache
        cache_size = len(query_cache)
        query_cache.clear()
        
        # Clear audio cache
        audio_files_cleared = 0
        if os.path.exists(AUDIO_CACHE_DIR):
            for filename in os.listdir(AUDIO_CACHE_DIR):
                if filename.endswith('.mp3'):
                    os.remove(os.path.join(AUDIO_CACHE_DIR, filename))
                    audio_files_cleared += 1
        
        return jsonify({
            "message": "Cache cleared successfully",
            "query_cache_cleared": cache_size,
            "audio_files_cleared": audio_files_cleared
        })
    except Exception as e:
        return jsonify({"error": "Cache clear failed"}), 500

@app.route("/clear-sessions", methods=["POST"])
def clear_sessions():
    """Clear conversation sessions"""
    global conversation_sessions
    session_count = len(conversation_sessions)
    conversation_sessions.clear()
    return jsonify({
        "message": f"Cleared {session_count} conversation sessions"
    })

@app.route("/session-info/<session_id>")
def get_session_info(session_id):
    """Get information about a conversation session"""
    # Sanitize session ID
    session_id = re.sub(r'[^a-zA-Z0-9_\-\.]', '', session_id)[:64]
    
    if session_id in conversation_sessions:
        memory = conversation_sessions[session_id]
        messages = []
        if hasattr(memory, 'chat_memory') and hasattr(memory.chat_memory, 'messages'):
            messages = [{"type": type(msg).__name__, "content": msg.content[:100]} for msg in memory.chat_memory.messages]
        
        return jsonify({
            "session_id": session_id,
            "message_count": len(messages),
            "messages": messages
        })
    else:
        return jsonify({"error": "Session not found"}), 404

@app.route("/continue", methods=["POST"])
def continue_conversation():
    """Explicitly continue from where we left off"""
    try:
        payload = {}
        if request.is_json:
            payload = request.get_json()
        else:
            try:
                payload = request.get_json(force=True)
            except:
                payload = {}
        
        session_id = payload.get("session_id", "").strip()
        tenant = payload.get("company_id_slug", "").strip()
        
        # Validate inputs
        if not session_id or not validate_company_id_mvp(tenant):
            return jsonify({"error": "Invalid session or company identifier"}), 400
        
        # Sanitize session ID
        session_id = re.sub(r'[^a-zA-Z0-9_\-\.]', '', session_id)[:64]
        
        if session_id not in conversation_sessions:
            return jsonify({"error": "No conversation session found"}), 400
        
        memory = conversation_sessions[session_id]
        
        # Get last AI response from memory
        last_response = ""
        if hasattr(memory, 'chat_memory') and hasattr(memory.chat_memory, 'messages'):
            for msg in reversed(memory.chat_memory.messages):
                if hasattr(msg, 'content') and len(msg.content) > 50:
                    last_response = msg.content
                    break
        
        if last_response:
            # Check if it was truncated
            if "Would you like me to continue" in last_response or "Should I continue" in last_response:
                # Generate continuation using the query endpoint
                continue_payload = {
                    "query": "Please continue from where you left off with the rest of the details.",
                    "company_id_slug": tenant,
                    "session_id": session_id
                }
                
                # Call query endpoint internally
                with app.test_request_context('/query', json=continue_payload, method='POST'):
                    return query_sop()
            else:
                return jsonify({
                    "answer": "I don't see that the previous response was truncated. What specific aspect would you like me to elaborate on?",
                    "source": "clarification",
                    "session_id": session_id
                })
        else:
            return jsonify({
                "answer": "I don't have a previous response to continue from. What would you like to know?",
                "source": "clarification",
                "session_id": session_id
            })
            
    except Exception as e:
        print(f"[CONTINUE ERROR] {e}")
        return jsonify({"error": "Continue failed"}), 500

@app.route("/force-embed-demo", methods=["POST"])
def force_embed_demo():
    """Force embed all files for demo-business-123"""
    try:
        all_files = glob.glob(os.path.join(SOP_FOLDER, "*.pdf")) + glob.glob(os.path.join(SOP_FOLDER, "*.docx"))
        
        if not all_files:
            return jsonify({"message": "No files found", "sop_folder": SOP_FOLDER})
        
        embedded_count = 0
        for file_path in all_files:
            filename = os.path.basename(file_path)
            title = filename.rsplit('.', 1)[0]
            
            metadata = {
                "title": title,
                "company_id_slug": "demo-business-123", 
                "filename": filename,
                "uploaded_at": time.time(),
                "file_size": os.path.getsize(file_path)
            }
            
            print(f"[FORCE-EMBED] {filename} -> demo-business-123")
            update_status(filename, {"status": "force-embedding...", **metadata})
            Thread(target=embed_sop_worker, args=(file_path, metadata), daemon=True).start()
            embedded_count += 1
        
        return jsonify({
            "message": f"Started embedding {embedded_count} files for demo-business-123",
            "files": [os.path.basename(f) for f in all_files]
        })
        
    except Exception as e:
        print(f"[FORCE-EMBED ERROR] {e}")
        return jsonify({"error": "Force embedding failed"}), 500

# Simple memory cleanup - runs every 100 requests
request_counter = 0

@app.before_request
def before_request():
    """Simple cleanup before each request"""
    global request_counter, conversation_sessions
    request_counter += 1
    
    # Simple cleanup every 100 requests
    if request_counter % 100 == 0:
        # Clean old sessions (simple approach for MVP)
        if len(conversation_sessions) > 50:  # Keep only 50 most recent sessions
            # Remove oldest sessions (keep newest 25)
            session_items = list(conversation_sessions.items())
            conversation_sessions.clear()
            for session_id, memory in session_items[-25:]:  # Keep last 25
                conversation_sessions[session_id] = memory

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(413)
def too_large(error):
    return jsonify({"error": "File too large"}), 413

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    
    # Load vectorstore on startup
    print("[STARTUP] Loading vector store...")
    load_vectorstore()
    
    print(f"[STARTUP] Starting Production OpsVoice API v1.3.0 on port {port}")
    print(f"[STARTUP] Features: Session Memory, Smart Truncation, Performance Optimization, MVP Security, n8n Compatibility")
    print(f"[STARTUP] Security: File validation, Rate limiting, Input sanitization, Safe errors")
    print(f"[STARTUP] Target response time: 3-5 seconds")
    app.run(host="0.0.0.0", port=port, debug=False)