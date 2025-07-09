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

# ---- Configuration ----
# Session memory storage
conversation_sessions = {}

# Rate limiting
rate_limits = {}
MAX_REQUESTS_PER_MINUTE = 50  # Higher for demo
MAX_REQUESTS_PER_MINUTE_DEMO = 999  # Unlimited for demo
MAX_CACHE_SIZE = 500

# File upload security
ALLOWED_EXTENSIONS = {'pdf', 'docx'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# ---- Paths & Setup ----
DATA_PATH = os.getenv("DATA_PATH", "/opt/render/project/src/data")
SOP_FOLDER = os.path.join(DATA_PATH, "sop-files")
CHROMA_DIR = os.path.join(DATA_PATH, "chroma_db")
AUDIO_CACHE_DIR = os.path.join(DATA_PATH, "audio_cache")
STATUS_FILE = os.path.join(SOP_FOLDER, "status.json")

# Ensure directories exist
os.makedirs(SOP_FOLDER, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)
os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)

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

# Enhanced CORS configuration
CORS(app, resources={
    r"/*": {
        "origins": ["*"],  # Allow all origins for MVP
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"],
        "supports_credentials": True,
        "max_age": 3600
    }
})

@app.after_request
def after_request(response):
    """Enhanced CORS headers"""
    origin = request.headers.get('Origin', '*')
    response.headers['Access-Control-Allow-Origin'] = origin
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With'
    
    # Handle preflight
    if request.method == 'OPTIONS':
        response.headers['Access-Control-Max-Age'] = '3600'
        response.status_code = 200
        
    return response

# ---- Security Functions ----
def validate_file_upload(file):
    """Validate uploaded files"""
    if not file or not file.filename:
        return False, "No file uploaded"
    
    filename = secure_filename(file.filename)
    if not filename:
        return False, "Invalid filename"
    
    if '.' not in filename:
        return False, "File must have extension"
    
    ext = filename.rsplit('.', 1)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Only PDF and DOCX files allowed"
    
    file.seek(0, 2)
    size = file.tell()
    file.seek(0)
    if size > MAX_FILE_SIZE:
        return False, "File too large (max 10MB)"
    
    return True, filename

def check_rate_limit(tenant: str) -> bool:
    """Rate limiting with demo bypass"""
    global rate_limits
    
    # No rate limit for demo
    if tenant == "demo-business-123":
        return True
    
    current_minute = int(time.time() // 60)
    key = f"{tenant}:{current_minute}"
    
    # Clean old entries
    cutoff = current_minute - 5
    rate_limits = {k: v for k, v in rate_limits.items() 
                   if int(k.split(':')[1]) > cutoff}
    
    count = rate_limits.get(key, 0)
    limit = MAX_REQUESTS_PER_MINUTE_DEMO if tenant == "demo-business-123" else MAX_REQUESTS_PER_MINUTE
    
    if count >= limit:
        return False
    
    rate_limits[key] = count + 1
    return True

def validate_company_id(company_id: str) -> bool:
    """Validate company ID format"""
    if not company_id or len(company_id) < 3:
        return False
    
    if not re.match(r'^[a-zA-Z0-9_-]+$', company_id):
        return False
    
    if '..' in company_id or '/' in company_id:
        return False
    
    return True

def sanitize_text(text: str) -> str:
    """Sanitize text input"""
    if not text:
        return ""
    
    # Remove dangerous characters
    text = re.sub(r'[<>"\'\x00\r\n]', '', text)
    
    # Limit length
    return text[:1000].strip()

# ---- Utility Functions ----
def clean_text(txt: str) -> str:
    """Clean text for TTS"""
    if not txt:
        return ""
    txt = re.sub(r"\s+", " ", txt)
    txt = txt.replace("\u2022", "-").replace("\t", " ")
    txt = re.sub(r"[*#]+", "", txt)
    txt = re.sub(r"\[.*?\]", "", txt)
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
    """Get or create conversation memory"""
    global conversation_sessions
    
    if session_id not in conversation_sessions:
        conversation_sessions[session_id] = ConversationBufferMemory(
            memory_key="chat_history", 
            return_messages=True,
            output_key="answer"
        )
    return conversation_sessions[session_id]

def get_query_complexity(query: str) -> str:
    """Determine query complexity"""
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
    """Generate cache key"""
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
    """Cache response"""
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
                "i don't have", "unable to find"]
    
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
        if ext == "docx":
            docs = UnstructuredWordDocumentLoader(fpath).load()
        elif ext == "pdf":
            docs = PyPDFLoader(fpath).load()
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
       
        db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding)
        db.add_documents(chunks)
        db.persist()
        
        logger.info(f"Successfully embedded {len(chunks)} chunks from {fname}")
        update_status(fname, {"status": "embedded", "chunk_count": len(chunks), **(metadata or {})})
        
    except Exception as e:
        logger.error(f"Error embedding {fname}: {e}")
        update_status(fname, {"status": f"error: {str(e)}", **(metadata or {})})

def update_status(filename, status):
    """Update document processing status"""
    try:
        data = json.load(open(STATUS_FILE)) if os.path.exists(STATUS_FILE) else {}
        data[filename] = {**status, "updated_at": time.time()}
        with open(STATUS_FILE, "w") as f: 
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Error updating status for {filename}: {e}")

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
    return jsonify({
        "status": "ok", 
        "message": "🚀 OpsVoice RAG API is live!",
        "version": "1.4.0-production",
        "features": ["session_memory", "smart_truncation", "performance_optimization", "security_enhanced", "demo_mode"]
    })

@app.route("/healthz")
def healthz(): 
    return jsonify({
        "status": "healthy",
        "timestamp": time.time(),
        "vectorstore": "loaded" if vectorstore else "not_loaded",
        "cache_size": len(query_cache),
        "active_sessions": len(conversation_sessions),
        "avg_response_time": performance_metrics.get("avg_response_time", 0)
    })

@app.route("/metrics")
def get_metrics():
    """Get performance metrics"""
    return jsonify(performance_metrics)

@app.route("/upload-sop", methods=["POST", "OPTIONS"])
def upload_sop():
    """Upload and process documents"""
    if request.method == "OPTIONS":
        return "", 204
        
    try:
        file = request.files.get("file")
        is_valid, result = validate_file_upload(file)
        if not is_valid:
            return jsonify({"error": result}), 400
        
        filename = result
        
        tenant = request.form.get("company_id_slug", "").strip()
        if not validate_company_id(tenant):
            return jsonify({"error": "Invalid company identifier"}), 400
        
        if not check_rate_limit(tenant):
            return jsonify({"error": "Too many requests - please wait"}), 429
        
        title = request.form.get("doc_title", filename)[:100]
        title = re.sub(r'[<>"\']', '', title)
        
        timestamp = int(time.time())
        safe_filename = f"{tenant}_{timestamp}_{filename}"
        save_path = os.path.join(SOP_FOLDER, safe_filename)
        
        file.save(save_path)
        
        metadata = {
            "title": title,
            "company_id_slug": tenant,
            "filename": safe_filename,
            "uploaded_at": time.time(),
            "file_size": os.path.getsize(save_path)
        }
        
        update_status(safe_filename, {"status": "embedding...", **metadata})
        Thread(target=embed_sop_worker, args=(save_path, metadata), daemon=True).start()

        return jsonify({
            "message": "Document uploaded successfully",
            "doc_title": title,
            "company_id_slug": tenant,
            "sop_file_url": f"{request.host_url.rstrip('/')}/static/sop-files/{safe_filename}",
            "upload_id": secrets.token_hex(8)
        })
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({"error": "Upload failed"}), 500

@app.route("/query", methods=["POST", "OPTIONS"])
def query_sop():
    """Process text queries with RAG"""
    if request.method == "OPTIONS":
        return "", 204
        
    start_time = time.time()
    
    try:
        # Parse request
        payload = request.get_json(force=True, silent=True) or {}
        
        qtext = sanitize_text(payload.get("query", ""))
        tenant = payload.get("company_id_slug", "").strip()
        session_id = payload.get("session_id") or f"{tenant}_{int(time.time())}"
        session_id = re.sub(r'[^a-zA-Z0-9_\-\.]', '', str(session_id))[:64]

        # Validate inputs
        if not qtext or len(qtext.strip()) < 3:
            return jsonify({
                "answer": "Please ask a more specific question.",
                "source": "error",
                "followups": ["What procedure are you looking for?"],
                "session_id": session_id
            }), 400
            
        if not validate_company_id(tenant):
            return jsonify({
                "answer": "Invalid company identifier.",
                "source": "error",
                "followups": []
            }), 400
            
        if not check_rate_limit(tenant):
            return jsonify({
                "answer": "Too many requests. Please wait a moment.",
                "source": "error",
                "followups": []
            }), 429

        # Check cache
        cached_response = get_cached_response(qtext, tenant)
        if cached_response:
            cached_response["session_id"] = session_id
            update_metrics(time.time() - start_time, "cache")
            return jsonify(cached_response)

        # Ensure vectorstore
        if not ensure_vectorstore():
            return jsonify({
                "answer": "Service temporarily unavailable. Please try again.",
                "source": "error",
                "followups": []
            }), 503

        # Handle vague queries
        if is_vague(qtext):
            response = {
                "answer": "Could you please be more specific? What procedure or policy are you looking for?",
                "source": "clarify",
                "followups": ["How do I handle customer complaints?", "What's the refund procedure?", "Where are the training documents?"],
                "session_id": session_id
            }
            update_metrics(time.time() - start_time, "clarify")
            return jsonify(response)

        # Filter off-topic
        off_topic_keywords = ["gmail", "facebook", "amazon", "weather", "news", "stock", "crypto", "youtube", "instagram"]
        if any(keyword in qtext.lower() for keyword in off_topic_keywords):
            response = {
                "answer": "Please ask questions about your company procedures and policies.",
                "source": "off_topic",
                "followups": ["What are our customer service procedures?", "How do I process a refund?", "What's the onboarding process?"],
                "session_id": session_id
            }
            update_metrics(time.time() - start_time, "off_topic")
            return jsonify(response)

        # Process with RAG
        complexity = get_query_complexity(qtext)
        optimal_llm = get_optimal_llm(complexity)
        expanded_query = expand_query_with_synonyms(qtext)
        
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

        # Check if answer is helpful
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
            # Fallback response
            fallback_answer = f"""I don't see specific information about that in your company documents.

Here are some general guidelines:
- Check with your manager or supervisor
- Look for the relevant department's procedures
- Review your employee handbook

Would you like to ask about something else?"""
            
            response = {
                "answer": fallback_answer,
                "fallback_used": True,
                "followups": ["What procedures are available?", "How do I find training documents?", "What policies are uploaded?"],
                "source": "fallback",
                "session_id": session_id
            }
            
            update_metrics(time.time() - start_time, "fallback")
            return jsonify(response)

    except Exception as e:
        logger.error(f"Query error: {traceback.format_exc()}")
        update_metrics(time.time() - start_time, "error")
        return jsonify({
            "answer": "I encountered an error processing your request. Please try again.",
            "source": "error",
            "followups": ["Ask a different question", "Try a simpler query"],
            "error_details": str(e) if app.debug else None
        }), 500

@app.route("/voice-reply", methods=["POST", "OPTIONS"])
def voice_reply():
    """Convert text to speech"""
    if request.method == "OPTIONS":
        return "", 204

    try:
        payload = request.get_json(force=True, silent=True) or {}
        
        # Accept both 'query' and 'text' for compatibility
        text = payload.get("query") or payload.get("text", "")
        text = clean_text(text)
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        tenant = payload.get("company_id_slug", "demo").strip()
        
        if not check_rate_limit(tenant):
            return jsonify({"error": "Rate limit exceeded"}), 429
        
        # Limit text length
        tts_text = text[:500] if len(text) > 500 else text
        
        # Cache key
        content_hash = hashlib.md5(tts_text.encode()).hexdigest()
        cache_key = f"tts_{tenant}_{content_hash}.mp3"
        cache_path = os.path.join(AUDIO_CACHE_DIR, cache_key)
        
        # Serve from cache
        if os.path.exists(cache_path):
            return send_file(cache_path, mimetype="audio/mp3", as_attachment=False)
        
        # Generate new audio
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
            logger.error(f"ElevenLabs error: {tts_resp.status_code}")
            return jsonify({"error": "TTS generation failed"}), 502
        
        # Cache audio
        audio_data = tts_resp.content
        with open(cache_path, "wb") as f:
            f.write(audio_data)
        
        return send_file(io.BytesIO(audio_data), mimetype="audio/mp3", as_attachment=False)
        
    except requests.exceptions.Timeout:
        return jsonify({"error": "TTS timeout"}), 504
    except Exception as e:
        logger.error(f"TTS error: {e}")
        return jsonify({"error": "TTS failed"}), 500

@app.route("/company-docs/<company_id_slug>")
def company_docs(company_id_slug):
   """Get documents for a specific company"""
   if not validate_company_id(company_id_slug):
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
       
       return jsonify(company_docs)
       
   except Exception as e:
       logger.error(f"Error fetching docs: {e}")
       return jsonify({"error": "Failed to fetch documents"}), 500

@app.route("/continue", methods=["POST", "OPTIONS"])
def continue_conversation():
   """Continue from previous response"""
   if request.method == "OPTIONS":
       return "", 204
       
   try:
       payload = request.get_json(force=True, silent=True) or {}
       
       session_id = payload.get("session_id", "").strip()
       tenant = payload.get("company_id_slug", "").strip()
       
       if not session_id or not validate_company_id(tenant):
           return jsonify({"error": "Invalid session or company"}), 400
       
       session_id = re.sub(r'[^a-zA-Z0-9_\-\.]', '', session_id)[:64]
       
       if session_id not in conversation_sessions:
           return jsonify({"error": "Session not found"}), 404
       
       memory = conversation_sessions[session_id]
       
       # Get last response
       last_response = ""
       if hasattr(memory, 'chat_memory') and hasattr(memory.chat_memory, 'messages'):
           for msg in reversed(memory.chat_memory.messages):
               if hasattr(msg, 'content') and len(msg.content) > 50:
                   last_response = msg.content
                   break
       
       if last_response and ("Would you like me to continue" in last_response or "Should I continue" in last_response):
           # Generate continuation
           continue_query = "Please continue from where you left off with the rest of the details."
           
           # Use query endpoint
           with app.test_request_context():
               request.json = {
                   "query": continue_query,
                   "company_id_slug": tenant,
                   "session_id": session_id
               }
               return query_sop()
       else:
           return jsonify({
               "answer": "I don't see a previous response to continue from. What would you like to know?",
               "source": "clarification",
               "session_id": session_id,
               "followups": ["Ask a new question", "What procedures do you need?"]
           })
           
   except Exception as e:
       logger.error(f"Continue error: {e}")
       return jsonify({"error": "Continue failed"}), 500

@app.route("/session-info/<session_id>")
def get_session_info(session_id):
   """Get session information"""
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

@app.route("/clear-cache", methods=["POST"])
def clear_cache():
   """Clear caches"""
   global query_cache
   try:
       cache_size = len(query_cache)
       query_cache.clear()
       
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
       return jsonify({"error": str(e)}), 500

@app.route("/static/sop-files/<path:filename>")
def serve_sop(filename):
   """Serve SOP files"""
   safe_filename = secure_filename(filename)
   return send_from_directory(SOP_FOLDER, safe_filename)

# Cleanup old sessions periodically
def cleanup_old_sessions():
   """Remove old conversation sessions"""
   global conversation_sessions
   if len(conversation_sessions) > 100:
       # Keep only 50 most recent
       sorted_sessions = sorted(conversation_sessions.items(), key=lambda x: id(x[1]))
       conversation_sessions = dict(sorted_sessions[-50:])

# Error handlers
@app.errorhandler(404)
def not_found(error):
   return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
   return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
   port = int(os.environ.get("PORT", 10000))
   
   # Load vectorstore on startup
   logger.info("Loading vector store...")
   load_vectorstore()
   
   logger.info(f"Starting OpsVoice API v1.4.0 on port {port}")
   app.run(host="0.0.0.0", port=port, debug=False)