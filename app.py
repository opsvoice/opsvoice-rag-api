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

load_dotenv()

# ---- Paths & Setup ----
DATA_PATH = "/opt/render/project/src/data"
SOP_FOLDER = os.path.join(DATA_PATH, "sop-files")
CHROMA_DIR = os.path.join(DATA_PATH, "chroma_db")
AUDIO_CACHE_DIR = os.path.join(DATA_PATH, "audio_cache")
STATUS_FILE = os.path.join(SOP_FOLDER, "status.json")
METRICS_FILE = os.path.join(DATA_PATH, "metrics.json")

os.makedirs(SOP_FOLDER, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)
os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)

# ---- Enhanced Caching & Memory ----
query_cache = {}  # Simple in-memory cache
conversation_sessions = {}  # Session-based memory
performance_metrics = {
    "total_queries": 0,
    "cache_hits": 0,
    "avg_response_time": 0,
    "model_usage": {"gpt-3.5-turbo": 0, "gpt-4": 0},
    "response_sources": {"sop": 0, "fallback": 0, "cache": 0}
}

# Clean Chroma on startup
if os.path.exists(CHROMA_DIR):
    try:
        shutil.rmtree(CHROMA_DIR)
        os.makedirs(CHROMA_DIR, exist_ok=True)
        print(f"[CLEANUP] Cleaned ChromaDB at {CHROMA_DIR}")
    except Exception as e:
        print(f"[CLEANUP] Error: {e}")

embedding = OpenAIEmbeddings()
vectorstore = None

# ---- Flask Setup ----
app = Flask(__name__)
CORS(app, origins=["https://opsvoice-widget.vercel.app", "http://localhost:3000"], supports_credentials=True)

@app.after_request
def add_cors(response):
    origin = request.headers.get("Origin", "")
    allowed_origins = ["https://opsvoice-widget.vercel.app", "http://localhost:3000"]
    if origin in allowed_origins:
        response.headers["Access-Control-Allow-Origin"] = origin
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    return response

# ---- Enhanced Utility Functions ----
def clean_text(txt: str) -> str:
    """Clean text for TTS - remove problematic characters"""
    if not txt:
        return ""
    txt = re.sub(r"\s+", " ", txt)
    txt = txt.replace("\u2022", "-").replace("\t", " ")
    txt = re.sub(r"[*#]+", "", txt)  # Remove markdown
    txt = re.sub(r"\[.*?\]", "", txt)  # Remove brackets
    txt = txt.strip()
    return txt

def smart_truncate(text, max_words=150):
    """Smart truncation that preserves complete sentences and adds continuation prompts"""
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
    
    # If we found a good sentence break (and it's not too short)
    if last_ending > len(truncated) * 0.6:
        return truncated[:last_ending + 1] + " Would you like me to continue with more details?"
    else:
        # Fallback: find last complete phrase
        last_comma = truncated.rfind(',')
        if last_comma > len(truncated) * 0.7:
            return truncated[:last_comma] + ". For the complete procedure, please ask me to continue."
        else:
            # Final fallback
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
        len(words) <= 8,  # Short queries
        any(word in query.lower() for word in ['what', 'when', 'where', 'who', 'how many']),
        query.endswith('?') and len(words) <= 6
    ]
    
    # Complex query indicators  
    complex_indicators = [
        len(words) > 15,  # Long queries
        any(word in query.lower() for word in ['analyze', 'compare', 'explain why', 'walk me through', 'break down']),
        query.count('?') > 1,  # Multiple questions
        any(word in query.lower() for word in ['because', 'therefore', 'however', 'although'])
    ]
    
    if sum(complex_indicators) > 0:
        return "complex"
    elif sum(simple_indicators) >= 2:
        return "simple"
    else:
        return "medium"

def get_optimal_llm(complexity: str) -> ChatOpenAI:
    """Select optimal LLM based on query complexity - PERFORMANCE OPTIMIZATION"""
    global performance_metrics
    
    if complexity == "simple":
        performance_metrics["model_usage"]["gpt-3.5-turbo"] += 1
        return ChatOpenAI(temperature=0, model="gpt-3.5-turbo")  # 2x faster
    else:  # medium or complex
        performance_metrics["model_usage"]["gpt-4"] += 1
        return ChatOpenAI(temperature=0, model="gpt-4")

def get_cache_key(query: str, company_id: str) -> str:
    """Generate cache key for query"""
    combined = f"{company_id}:{query.lower().strip()}"
    return hashlib.md5(combined.encode()).hexdigest()

def get_cached_response(query: str, company_id: str) -> dict:
    """Get cached response if available - PERFORMANCE OPTIMIZATION"""
    cache_key = get_cache_key(query, company_id)
    cached = query_cache.get(cache_key)
    
    if cached and time.time() - cached['timestamp'] < 3600:  # 1 hour cache
        performance_metrics["cache_hits"] += 1
        performance_metrics["response_sources"]["cache"] += 1
        print(f"[CACHE] Cache hit for query: {query[:50]}...")
        return cached['response']
    
    return None

def cache_response(query: str, company_id: str, response: dict):
    """Cache response for future use"""
    cache_key = get_cache_key(query, company_id)
    query_cache[cache_key] = {
        'response': response,
        'timestamp': time.time()
    }
    
    # Limit cache size
    if len(query_cache) > 500:
        # Remove oldest entries
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
    
    # Save metrics periodically
    if performance_metrics["total_queries"] % 10 == 0:
        try:
            with open(METRICS_FILE, 'w') as f:
                json.dump(performance_metrics, f, indent=2)
        except Exception as e:
            print(f"[METRICS] Error saving: {e}")

def is_unhelpful_answer(text):
    """Enhanced unhelpful answer detection"""
    if not text or not text.strip(): 
        return True
        
    low = text.lower()
    
    # Definitive unhelpful phrases
    definitive_triggers = [
        "don't know", "no information", "i'm not sure", "sorry", 
        "unavailable", "not covered", "cannot find", "no specific information",
        "not mentioned", "doesn't provide", "no details", "not included",
        "context provided does not include", "text does not provide"
    ]
    
    # Must have substantial content (more than just a trigger phrase)
    has_trigger = any(t in low for t in definitive_triggers)
    is_too_short = len(low.split()) < 8
    
    return has_trigger or is_too_short

def contains_sensitive(text):
    """Temporarily disable sensitivity filtering for MVP"""
    return False

def generate_contextual_followups(query: str, answer: str, previous_queries: list = None) -> list:
    """Generate smarter contextual follow-up questions"""
    q = query.lower()
    a = answer.lower()
    base_followups = []
    
    # Context from previous queries
    context_words = []
    if previous_queries:
        for prev_q in previous_queries[-2:]:  # Last 2 queries
            context_words.extend(prev_q.lower().split())
    
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
    
    # Context-aware followups
    if "training" in context_words and "procedure" in q:
        base_followups.append("Would you like the training checklist for this procedure?")
    
    # Default fallbacks
    if not base_followups:
        base_followups.extend([
            "Do you want to know more details?",
            "Would you like steps for a related task?",
            "Need help finding a specific document?"
        ])
    
    return base_followups[:3]  # Return top 3

def is_vague(query): 
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

def ensure_vectorstore():
    """Ensure vectorstore is available and healthy"""
    global vectorstore
    try:
        if not vectorstore:
            load_vectorstore()
        
        # Test vectorstore health
        if vectorstore and hasattr(vectorstore, '_collection'):
            test_results = vectorstore.similarity_search("test", k=1)
            print(f"[DB] Vectorstore healthy, {len(test_results)} test results")
        
        return vectorstore is not None
    except Exception as e:
        print(f"[DB] Vectorstore health check failed: {e}")
        load_vectorstore()
        return vectorstore is not None

def get_company_documents_internal(company_id_slug):
    """Internal function to get company documents without HTTP request"""
    if not os.path.exists(STATUS_FILE): 
        return []
    
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
                    "uploaded_at": metadata.get("uploaded_at")
                }
                company_docs.append(doc_info)
        
        return company_docs
        
    except Exception as e:
        print(f"[DOCS_INTERNAL] Error: {e}")
        return []

# ---- Embedding Worker ----
def embed_sop_worker(fpath, metadata=None):
    """Enhanced embedding worker with better chunking for policies"""
    fname = os.path.basename(fpath)
    try:
        ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""
        if ext == "docx":
            docs = UnstructuredWordDocumentLoader(fpath).load()
        elif ext == "pdf":
            docs = PyPDFLoader(fpath).load()
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        
        # ENHANCED CHUNKING - This is the key fix
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # INCREASED from 500
            chunk_overlap=200,  # INCREASED from 100
            separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]
        ).split_documents(docs)
        
        # Add enhanced metadata for better search
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
            
            # Extract simple keywords as a string (not list)
            words = re.findall(r'\b[a-zA-Z]{4,}\b', chunk.page_content.lower())
            stopwords = {'that', 'this', 'with', 'they', 'have', 'will', 'from', 'been', 'were'}
            keywords = [word for word in set(words) if word not in stopwords][:5]
            keywords_string = " ".join(keywords)  # Convert to string

            chunk.metadata.update({
                **(metadata or {}),
                "chunk_id": f"{fname}_{i}",
                "chunk_type": chunk_type,
                "keywords": keywords_string  # ← NOW IT'S A STRING
            })
       
        db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding)
        db.add_documents(chunks)
        db.persist()
        
        print(f"[EMBED] Successfully embedded {len(chunks)} chunks from {fname}")
        update_status(fname, {"status": "embedded", "chunk_count": len(chunks), **(metadata or {})})
        
    except Exception as e:
        print(f"[EMBED] Error with {fname}: {e}")
        update_status(fname, {"status": f"error: {str(e)}", **(metadata or {})})

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
    """Load the vector database with better error handling"""
    global vectorstore
    try:
        vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding)
        print("[DB] Vector store loaded successfully")
    except Exception as e:
        print(f"[DB] Error loading vector store: {e}")
        vectorstore = None

# ---- Routes ----
@app.route("/")
def home(): 
    return jsonify({
        "status": "ok", 
        "message": "🚀 OpsVoice RAG API is live!",
        "version": "1.2.0",
        "features": ["smart_caching", "model_optimization", "session_memory", "smart_truncation"]
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
    """Serve SOP files"""
    return send_from_directory(SOP_FOLDER, filename)

@app.route("/upload-sop", methods=["POST"])
def upload_sop():
    """Upload and process SOP documents"""
    file = request.files.get("file")
    if not file or not file.filename: 
        return jsonify({"error": "No file uploaded"}), 400
        
    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in ("docx", "pdf"): 
        return jsonify({"error": "Only .docx and .pdf files allowed"}), 400

    # Get metadata
    tenant = re.sub(r"[^\w\-]", "", request.form.get("company_id_slug", ""))
    title = request.form.get("doc_title", file.filename)
    
    if not tenant:
        return jsonify({"error": "company_id_slug is required"}), 400
        
    # Check file size (10MB limit)
    file.seek(0, 2)  # Seek to end
    size = file.tell()
    file.seek(0)  # Reset
    if size > 10 * 1024 * 1024:  # 10MB
        return jsonify({"error": "File too large (max 10MB)"}), 400

    # Save file
    safe_filename = re.sub(r'[^a-zA-Z0-9._-]', '_', file.filename)
    save_path = os.path.join(SOP_FOLDER, safe_filename)
    file.save(save_path)

    # Prepare metadata
    metadata = {
        "title": title,
        "company_id_slug": tenant,
        "filename": safe_filename,
        "uploaded_at": time.time()
    }
    
    # Start background embedding
    update_status(safe_filename, {"status": "embedding...", **metadata})
    Thread(target=embed_sop_worker, args=(save_path, metadata), daemon=True).start()

    return jsonify({
        "message": f"Uploaded {safe_filename}, embedding in background.",
        "doc_title": title,
        "company_id_slug": tenant,
        "sop_file_url": f"{request.host_url.rstrip('/')}/static/sop-files/{safe_filename}"
    })

# Rate limiting - simplified for MVP
def check_rate_limit(tenant: str) -> bool:
    """Simplified rate limiting for MVP"""
    return True  # Disable rate limiting for now

@app.route("/query", methods=["POST"])
def query_sop():
    """ENHANCED query processing with session memory, smart truncation, and performance optimization"""
    start_time = time.time()
    global vectorstore
    
    if vectorstore is None: 
        load_vectorstore()

    payload = request.get_json() or {}
    qtext = clean_text(payload.get("query", ""))
    tenant = re.sub(r"[^\w\-]", "", payload.get("company_id_slug", ""))
    session_id = payload.get("session_id", f"{tenant}_{int(time.time())}")  # SESSION SUPPORT

    if not qtext or not tenant:
        return jsonify({"error": "Missing query or company_id_slug"}), 400

    # Check cache first - PERFORMANCE OPTIMIZATION
    cached_response = get_cached_response(qtext, tenant)
    if cached_response:
        update_metrics(time.time() - start_time, "cache")
        return jsonify(cached_response)

    # Rate limiting
    if not check_rate_limit(tenant):
        return jsonify({"error": "Too many requests, try again in a minute."}), 429

    # Check for vague queries
    if is_vague(qtext):
        response = {
            "answer": "Can you give me more detail—like the specific SOP or process you're referring to?",
            "source": "clarify"
        }
        update_metrics(time.time() - start_time, "clarify")
        return jsonify(response)

    # Filter out off-topic queries
    off_topic_keywords = ["gmail", "facebook", "amazon account", "weather", "news", "stock price"]
    if any(keyword in qtext.lower() for keyword in off_topic_keywords):
        response = {
            "answer": "That's outside your company SOPs—please use the official help portal or ask about your internal procedures.",
            "source": "off_topic"
        }
        update_metrics(time.time() - start_time, "off_topic")
        return jsonify(response)

    try:
        # PERFORMANCE OPTIMIZATION - Select optimal model
        complexity = get_query_complexity(qtext)
        optimal_llm = get_optimal_llm(complexity)
        
        print(f"[QUERY] Using {optimal_llm.model_name} for {complexity} query: {qtext[:50]}...")
        
        # Expand query with synonyms for better matching
        expanded_query = expand_query_with_synonyms(qtext)
        print(f"[QUERY] Expanded query: {expanded_query}")
        
        # PERFORMANCE OPTIMIZATION - Reduced retrieval for faster processing
        retriever = vectorstore.as_retriever(
            search_kwargs={
                "k": 5,  # REDUCED from 8 for faster processing
                "filter": {"company_id_slug": tenant}
            }
        )
        
        # Try to get documents
        relevant_docs = retriever.get_relevant_documents(expanded_query)
        
        if not relevant_docs:
            print(f"[QUERY] No docs with company filter, trying without filter")
            # Try without company filter
            general_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # Reduced for speed
            relevant_docs = general_retriever.get_relevant_documents(expanded_query)
        
        print(f"[QUERY] Found {len(relevant_docs)} relevant documents")
        
        if relevant_docs:
            # SESSION MEMORY - Get persistent memory for this session
            memory = get_session_memory(session_id)
            
            qa = ConversationalRetrievalChain.from_llm(
                optimal_llm,  # Use optimal model based on complexity
                retriever=vectorstore.as_retriever(search_kwargs={"k": len(relevant_docs)}),
                memory=memory,  # SESSION MEMORY
                return_source_documents=True
            )
            
            # Query the chain
            result = qa.invoke({"question": expanded_query})
            answer = clean_text(result.get("answer", ""))

            # Enhanced answer validation
            generic_phrases = ["i don't have", "i'm not sure", "sorry, i don't", "i cannot", "i'm unable"]
            is_generic = any(phrase in answer.lower() for phrase in generic_phrases) if answer else True
            
            if answer and len(answer.split()) > 10 and not is_generic:
                print(f"[QUERY] Success - found relevant answer")
                
                # SMART TRUNCATION - Preserves complete sentences
                if len(answer.split()) > 150:
                    answer = smart_truncate(answer, 150)
               
                # CLEAN RESPONSES - No company prefix
                response = {
                    "answer": answer,  # NO MORE COMPANY PREFIX
                    "fallback_used": False,
                    "followups": generate_contextual_followups(qtext, answer),
                    "source": "sop",
                    "source_documents": len(relevant_docs),
                    "session_id": session_id,  # SESSION SUPPORT
                    "model_used": optimal_llm.model_name  # DEBUG INFO
                }
                
                # Cache successful responses
                cache_response(qtext, tenant, response)
                update_metrics(time.time() - start_time, "sop")
                return jsonify(response)

        # Enhanced fallback
        print(f"[QUERY] Using enhanced fallback")
        
        query_lower = qtext.lower()
        if any(word in query_lower for word in ["angry", "upset", "customer"]):
            fallback_answer = f"""I don't see specific customer service policies in your uploaded documents for {tenant.replace('-', ' ').title()}. 

For handling difficult customers, here are general best practices:
1. Listen actively and remain calm
2. Acknowledge their concerns
3. Apologize for any inconvenience  
4. Focus on finding a solution
5. Escalate to a manager if needed

Please check your employee handbook or contact your supervisor for company-specific policies."""

        elif any(word in query_lower for word in ["cash", "money", "refund"]):
            fallback_answer = f"""I don't see specific financial procedures in your uploaded documents for {tenant.replace('-', ' ').title()}.

For financial processes, please:
1. Refer to your company's financial procedures manual
2. Check with your manager for authorization limits
3. Contact the accounting department for guidance

Try asking about other topics from your uploaded company documents."""

        else:
            fallback_answer = f"""I don't see information about that in your company documents for {tenant.replace('-', ' ').title()}.

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
        print(f"[QUERY] Error: {e}")
        update_metrics(time.time() - start_time, "error")
        return jsonify({"error": "Query failed", "details": str(e)}), 500

@app.route("/voice-reply", methods=["POST", "OPTIONS"])
def voice_reply():
    """Convert text to speech for voice responses with enhanced caching"""
    # Handle CORS preflight
    if request.method == "OPTIONS":
        response = make_response()
        response.headers["Access-Control-Allow-Origin"] = request.headers.get("Origin", "")
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        return response

    data = request.get_json() or {}
    text = clean_text(data.get("query", ""))
    
    if not text: 
        return jsonify({"error": "Empty text"}), 400

    # Generate cache key with content hash for better caching
    content_hash = hashlib.md5(text.encode()).hexdigest()
    tenant = re.sub(r"[^\w\-]", "", data.get("company_id_slug", ""))
    cache_key = f"{tenant}_{content_hash}.mp3"
    cache_path = os.path.join(AUDIO_CACHE_DIR, cache_key)

    # Serve from cache if available
    if os.path.exists(cache_path):
        print(f"[TTS] Serving cached audio: {cache_key}")
        return send_file(cache_path, mimetype="audio/mp3", as_attachment=False)

    # Generate new audio
    try:
        # Limit text length for TTS
        tts_text = text[:500] if len(text) > 500 else text
        
        print(f"[TTS] Generating audio for: {tts_text[:50]}...")
        
        tts_resp = requests.post(
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
                }
            },
            timeout=30
        )
        
        if tts_resp.status_code != 200:
            print(f"[TTS] ElevenLabs error: {tts_resp.status_code} - {tts_resp.text}")
            return jsonify({"error": "TTS service error"}), 502

        # Cache the audio
        audio_data = tts_resp.content
        with open(cache_path, "wb") as f:
            f.write(audio_data)
        
        print(f"[TTS] Generated and cached: {cache_key}")
        return send_file(io.BytesIO(audio_data), mimetype="audio/mp3", as_attachment=False)

    except requests.exceptions.Timeout:
        print("[TTS] Request timeout")
        return jsonify({"error": "TTS request timeout"}), 504
    except Exception as e:
        print(f"[TTS] Error: {e}")
        return jsonify({"error": "TTS request failed", "details": str(e)}), 500

@app.route("/sop-status")
def sop_status():
    """Get document processing status"""
    if os.path.exists(STATUS_FILE): 
        return send_file(STATUS_FILE)
    return jsonify({})

@app.route("/company-docs/<company_id_slug>")
def company_docs(company_id_slug):
    """Get documents for a specific company"""
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
    """Lookup company slug by email"""
    email = request.args.get("email", "").strip().lower()
    if not email or not os.path.exists(STATUS_FILE):
        return jsonify({"error": "Invalid email or missing status"}), 400

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
        return jsonify({"error": str(e)}), 500

@app.route("/clear-sessions", methods=["POST"])
def clear_sessions():
    """Clear conversation sessions"""
    global conversation_sessions
    session_count = len(conversation_sessions)
    conversation_sessions.clear()
    return jsonify({
        "message": f"Cleared {session_count} conversation sessions"
    })

# NEW ENDPOINT: Get session info for debugging
@app.route("/session-info/<session_id>")
def get_session_info(session_id):
    """Get information about a conversation session"""
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

# NEW ENDPOINT: Continue conversation explicitly
@app.route("/continue", methods=["POST"])
def continue_conversation():
    """Explicitly continue from where we left off"""
    payload = request.get_json() or {}
    session_id = payload.get("session_id")
    tenant = payload.get("company_id_slug", "")
    
    if not session_id or session_id not in conversation_sessions:
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
            # Generate continuation
            continue_prompt = "Please continue from where you left off with the rest of the details."
            
            # Use the same query processing but with continuation context
            payload["query"] = continue_prompt
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
    print("[STARTUP] Loading vector store...")
    load_vectorstore()
    
    # Load existing metrics if available
    try:
        if os.path.exists(METRICS_FILE):
            with open(METRICS_FILE, 'r') as f:
                saved_metrics = json.load(f)
                performance_metrics.update(saved_metrics)
                print(f"[STARTUP] Loaded existing metrics: {performance_metrics['total_queries']} total queries")
    except Exception as e:
        print(f"[STARTUP] Could not load metrics: {e}")
    
    print(f"[STARTUP] Starting Optimized OpsVoice API v1.2.0 on port {port}")
    print(f"[STARTUP] Features: Smart Caching, Session Memory, Smart Truncation, Model Optimization")
    print(f"[STARTUP] Performance: Target 3-5s response time (down from 12s)")
    app.run(host="0.0.0.0", port=port, debug=False)