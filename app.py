"""
OpsVoice RAG API - Production Deployment Script for Render.com

This script is the main entrypoint for the optimized OpsVoice API.
It ensures all optimizations are initialized and visible in production.

Deployment Checklist (for Render.com):
1. Ensure all code is committed to main branch (including this file and requirements.txt).
2. requirements.txt must include: psutil>=5.9.0
3. Push to main branch to trigger Render auto-deploy.
4. After deploy, verify:
   - /healthz shows precomputed_responses >= 6
   - vectorstore_loaded: true
   - All optimization features active
   - 1-2s response times

If you see any issues in production, check logs and confirm that initialize_optimized_app() is called at startup.
"""

import os, glob, json, re, time, io, shutil, requests, hashlib, traceback, secrets
import logging
import pickle
import gc
from dotenv import load_dotenv
from threading import Thread, Timer, Lock
from functools import lru_cache, wraps
from collections import OrderedDict
from datetime import datetime, timedelta

from flask import Flask, request, jsonify, send_file, make_response
from flask_cors import CORS
from werkzeug.utils import secure_filename

import psutil  # For system monitoring

# LangChain imports
from langchain_community.document_loaders import UnstructuredWordDocumentLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# ==== ENVIRONMENT & LOGGING ====
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==== PERFORMANCE OPTIMIZATIONS ====
executor = ThreadPoolExecutor = __import__('concurrent.futures').futures.ThreadPoolExecutor
executor = executor(max_workers=4)
gc.set_threshold(700, 10, 10)
cache_lock = Lock()
session_lock = Lock()

# ==== PATHS & DIRECTORIES ====
if os.path.exists("/data"):
    DATA_PATH = "/data"
else:
    DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    print(f"🔧 DATA_PATH set to: {DATA_PATH}")

SOP_FOLDER = os.path.join(DATA_PATH, "sop-files")
CHROMA_DIR = os.path.join(DATA_PATH, "chroma_db")
AUDIO_CACHE_DIR = os.path.join(DATA_PATH, "audio_cache")
STATUS_FILE = os.path.join(SOP_FOLDER, "status.json")
METRICS_FILE = os.path.join(DATA_PATH, "metrics.json")
CACHE_FILE = os.path.join(DATA_PATH, "query_cache.pkl")

os.makedirs(SOP_FOLDER, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)
os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)

# ==== CONSTANTS ====
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}
MAX_FILE_SIZE = 15 * 1024 * 1024
MAX_CACHE_SIZE = 2000
QUERY_CACHE_TTL = 7200
AUDIO_CACHE_TTL = 86400 * 3
SESSION_TTL = 14400
PRECOMPUTE_CACHE_SIZE = 100

# ==== RATE LIMITS ====
DEMO_RATE_LIMITS = {
    'queries_per_minute': 300,
    'queries_per_hour': 10000,
    'uploads_per_hour': 200,
    'max_documents': 100
}
PRODUCTION_RATE_LIMITS = {
    'queries_per_minute': 60,
    'queries_per_hour': 1000,
    'uploads_per_hour': 50,
    'max_documents': 25
}

# ==== GLOBAL STATE ====
query_cache = OrderedDict()
conversation_sessions = {}
rate_limit_tracker = {}
precomputed_responses = {}
business_knowledge_cache = {}

performance_metrics = {
    "total_queries": 0,
    "cache_hits": 0,
    "precompute_hits": 0,
    "avg_response_time": 0,
    "response_times": [],
    "model_usage": {"gpt-3.5-turbo": 0, "gpt-4": 0, "cached": 0, "precomputed": 0},
    "response_sources": {"sop": 0, "fallback": 0, "cache": 0, "error": 0, "business": 0, "precomputed": 0},
    "companies": {},
    "error_count": 0,
    "system_health": {"cpu_percent": 0, "memory_percent": 0, "active_threads": 0}
}

embedding = OpenAIEmbeddings()
vectorstore = None

# ==== FLASK SETUP ====
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', secrets.token_hex(32))
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 300

ALLOWED_ORIGINS = [
    "https://opsvoice-widget.vercel.app",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://*.vercel.app"
]
CORS(app, resources={
    r"/*": {
        "origins": ALLOWED_ORIGINS,
        "supports_credentials": True,
        "allow_headers": ["Content-Type", "Authorization", "X-Requested-With", "Accept", "Origin", "X-API-Key", "Cache-Control"],
        "methods": ["GET", "POST", "OPTIONS", "PUT", "DELETE"],
        "expose_headers": ["Content-Range", "X-Content-Range"]
    }
})

# ==== PERFORMANCE DECORATORS ====
def async_task(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        future = executor.submit(f, *args, **kwargs)
        return future
    return wrapper

def timed_lru_cache(seconds: int, maxsize: int = 128):
    def wrapper_cache(func):
        func = lru_cache(maxsize=maxsize)(func)
        func.lifetime = timedelta(seconds=seconds)
        func.expiration = datetime.utcnow() + func.lifetime
        @wraps(func)
        def wrapped_func(*args, **kwargs):
            if datetime.utcnow() >= func.expiration:
                func.cache_clear()
                func.expiration = datetime.utcnow() + func.lifetime
            return func(*args, **kwargs)
        return wrapped_func
    return wrapper_cache

# ==== CORS HANDLING ====
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = make_response()
        origin = request.headers.get('Origin')
        if origin and any(allowed in origin for allowed in ALLOWED_ORIGINS):
            response.headers['Access-Control-Allow-Origin'] = origin
        else:
            response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS, PUT, DELETE'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With, Accept, Origin, X-API-Key, Cache-Control'
        response.headers['Access-Control-Allow-Credentials'] = 'true'
        response.headers['Access-Control-Max-Age'] = '86400'
        response.status_code = 200
        return response

@app.after_request
def after_request(response):
    origin = request.headers.get('Origin')
    if origin and any(allowed in origin for allowed in ALLOWED_ORIGINS):
        response.headers['Access-Control-Allow-Origin'] = origin
    else:
        response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    if request.endpoint in ['static', 'serve_sop']:
        response.headers['Cache-Control'] = 'public, max-age=300'
    return response

# ==== UTILITY FUNCTIONS ====
@timed_lru_cache(seconds=300, maxsize=1000)
def clean_text_cached(txt: str) -> str:
    if not txt:
        return ""
    txt = txt.replace('\u2022', '-').replace('\t', ' ')
    txt = re.sub(r'\s+', ' ', txt)
    txt = re.sub(r'[*#]+', '', txt)
    txt = re.sub(r'\[.*?\]', '', txt)
    return txt.strip()

@timed_lru_cache(seconds=600, maxsize=500)
def get_query_complexity_cached(query: str) -> str:
    words = query.lower().split()
    word_count = len(words)
    if word_count <= 6 and any(word in query.lower() for word in ['what', 'when', 'where', 'who']):
        return "simple"
    elif word_count > 20 or any(word in query.lower() for word in ['analyze', 'compare', 'comprehensive']):
        return "complex"
    else:
        return "medium"

def get_optimal_llm_fast(complexity: str) -> ChatOpenAI:
    model_configs = {
        "simple": {
            "model": "gpt-3.5-turbo",
            "temperature": 0,
            "request_timeout": 20,
            "max_retries": 1,
            "max_tokens": 300
        },
        "medium": {
            "model": "gpt-4",
            "temperature": 0,
            "request_timeout": 30,
            "max_retries": 2,
            "max_tokens": 500
        },
        "complex": {
            "model": "gpt-4",
            "temperature": 0,
            "request_timeout": 45,
            "max_retries": 2,
            "max_tokens": 800
        }
    }
    config = model_configs.get(complexity, model_configs["medium"])
    performance_metrics["model_usage"][config["model"]] += 1
    return ChatOpenAI(**config)

def get_cache_key_fast(query: str, company_id: str) -> str:
    combined = f"{company_id}:{query.lower().strip()}"
    return hashlib.md5(combined.encode()).hexdigest()

def get_cached_response_optimized(query: str, company_id: str) -> dict:
    cache_key = get_cache_key_fast(query, company_id)
    with cache_lock:
        cached = query_cache.get(cache_key)
        if cached and time.time() - cached['timestamp'] < QUERY_CACHE_TTL:
            query_cache.move_to_end(cache_key)
            performance_metrics["cache_hits"] += 1
            performance_metrics["response_sources"]["cache"] += 1
            return cached['response']
        if cached:
            del query_cache[cache_key]
    return None

def cache_response_optimized(query: str, company_id: str, response: dict):
    cache_key = get_cache_key_fast(query, company_id)
    with cache_lock:
        query_cache[cache_key] = {
            'response': response,
            'timestamp': time.time()
        }
        if len(query_cache) > MAX_CACHE_SIZE:
            remove_count = MAX_CACHE_SIZE // 5
            for _ in range(remove_count):
                query_cache.popitem(last=False)

# ==== PRECOMPUTED RESPONSES ====
def initialize_precomputed_responses():
    global precomputed_responses
    precomputed_responses = {
        "hello": "Hello! I'm here to help you with your company procedures, policies, and general business questions. What would you like to know?",
        "hi": "Hi there! How can I assist you with your business operations today?",
        "help": "I can help you with company procedures, policies, customer service guidelines, and general business questions. What specific topic would you like assistance with?",
        "customer service": """Here are proven customer service best practices:

**Key Principles:**
• Listen actively and remain calm
• Acknowledge concerns sincerely  
• Focus on solutions, not problems
• Follow up to ensure satisfaction
• Document interactions for improvement

**Handling Difficult Customers:**
1. Stay professional and empathetic
2. Ask clarifying questions
3. Offer multiple solution options
4. Escalate when appropriate
5. Follow your company's specific guidelines

Would you like more details on any specific aspect?""",
        "emergency procedures": """**General Emergency Response Steps:**

**Immediate Actions:**
1. Ensure personal safety first
2. Call 911 for life-threatening situations
3. Follow evacuation procedures
4. Use nearest safe exit route
5. Report to designated assembly point

**Important:** Every workplace should have documented emergency procedures. Please check with your manager for your company's specific emergency plans, including fire safety, medical emergencies, and evacuation procedures.

Is there a specific type of emergency procedure you need help with?""",
        "refund policy": """**Standard Refund Policy Guidelines:**

**Best Practices:**
• Clear time limits (typically 30 days)
• Proof of purchase required
• Original payment method preferred
• Manager approval for large amounts
• Document all refund transactions

**Process Steps:**
1. Verify purchase details
2. Check policy compliance
3. Get appropriate approval
4. Process refund promptly
5. Update customer records

For your company's specific refund policy, please check your employee handbook or ask your supervisor."""
    }

def get_precomputed_response(query: str) -> str:
    query_lower = query.lower().strip()
    if query_lower in precomputed_responses:
        performance_metrics["precompute_hits"] += 1
        performance_metrics["response_sources"]["precomputed"] += 1
        return precomputed_responses[query_lower]
    for key, response in precomputed_responses.items():
        if key in query_lower or any(word in query_lower for word in key.split()):
            performance_metrics["precompute_hits"] += 1
            performance_metrics["response_sources"]["precomputed"] += 1
            return response
    return None

# ==== BUSINESS INTELLIGENCE ====
@timed_lru_cache(seconds=3600, maxsize=100)
def generate_business_fallback_cached(query_type: str, company_id: str) -> str:
    company_name = company_id.replace('-', ' ').title()
    business_templates = {
        "customer_service": f"""**Customer Service Best Practices** (for {company_name}):

**De-escalation Techniques:**
• Listen without interrupting
• Use empathetic language: "I understand your frustration"
• Stay calm and professional
• Focus on solutions: "Here's what I can do for you"

**Resolution Process:**
1. **Acknowledge** - Validate their concern
2. **Apologize** - Take ownership appropriately  
3. **Act** - Provide clear next steps
4. **Follow-up** - Ensure satisfaction

**When to Escalate:**
- Customer requests manager
- Issue exceeds your authority
- Complex technical problems
- Legal or compliance concerns

For company-specific procedures, please check your employee handbook or consult your supervisor.""",
        "financial": f"""**Financial Transaction Guidelines** (for {company_name}):

**Authorization Levels:**
• Under $50: Staff level
• $50-$500: Supervisor approval
• Over $500: Manager approval
• Refunds: Follow specific policy

**Documentation Requirements:**
- Receipt for all transactions
- Customer information
- Authorization signatures
- Transaction logs

**Security Protocols:**
• Verify customer identity
• Check payment method validity
• Follow cash handling procedures
• Report discrepancies immediately

Please check with your accounting department for {company_name}'s specific financial procedures.""",
        "procedures": f"""**Standard Operating Procedure Framework**:

**Creating Effective SOPs:**
1. **Clear Objective** - What needs to be accomplished?
2. **Step-by-Step Process** - Detailed instructions
3. **Required Resources** - Tools, materials, access needed
4. **Quality Standards** - Expected outcomes
5. **Troubleshooting** - Common issues and solutions

**Implementation Tips:**
• Use simple, clear language
• Include visual aids when helpful
• Test procedures with new team members
• Update regularly based on feedback
• Make easily accessible to all staff

**For {company_name}:** Consider documenting your most common procedures first, then expanding to cover all critical processes.""",
        "training": f"""**Employee Training Best Practices**:

**Onboarding Essentials:**
• Company culture and values
• Role-specific responsibilities  
• System access and tools
• Safety procedures
• Key contacts and resources

**Training Methods:**
- Job shadowing with experienced staff
- Interactive workshops
- Online modules for flexibility
- Hands-on practice with feedback
- Regular check-ins and assessments

**Ongoing Development:**
• Monthly skill-building sessions
• Cross-training opportunities  
• Performance feedback
• Career development planning

For {company_name}, consider creating a structured 30-60-90 day onboarding plan.""",
        "default": f"""**Business Operations Guidance**:

**General Best Practices:**
• Document important procedures
• Communicate clearly with team members
• Follow company policies consistently
• Seek clarification when uncertain
• Focus on customer satisfaction

**When You Need Help:**
1. Check existing documentation
2. Ask experienced colleagues
3. Consult your supervisor
4. Review company handbook
5. Contact relevant departments

**Next Steps for {company_name}:**
Consider uploading your specific procedures to this system for easy team access."""
    }
    return business_templates.get(query_type, business_templates["default"])

def get_smart_business_response(query: str, company_id: str) -> str:
    query_lower = query.lower()
    if any(word in query_lower for word in ['angry', 'upset', 'difficult', 'complaint', 'customer']):
        return generate_business_fallback_cached("customer_service", company_id)
    elif any(word in query_lower for word in ['cash', 'money', 'refund', 'payment', 'financial']):
        return generate_business_fallback_cached("financial", company_id)
    elif any(word in query_lower for word in ['training', 'onboard', 'first day', 'new employee']):
        return generate_business_fallback_cached("training", company_id)
    elif any(word in query_lower for word in ['process', 'procedure', 'how to', 'steps']):
        return generate_business_fallback_cached("procedures", company_id)
    else:
        return generate_business_fallback_cached("default", company_id)

# ==== SESSION MANAGEMENT ====
def get_session_memory_optimized(session_id: str) -> ConversationBufferMemory:
    with session_lock:
        if session_id not in conversation_sessions:
            conversation_sessions[session_id] = {
                'memory': ConversationBufferMemory(
                    memory_key="chat_history",
                    return_messages=True,
                    output_key="answer",
                    max_token_limit=2000
                ),
                'created_at': time.time(),
                'last_accessed': time.time(),
                'query_count': 0
            }
        session = conversation_sessions[session_id]
        session['last_accessed'] = time.time()
        session['query_count'] += 1
        return session['memory']

# ==== METRICS ====
def update_metrics_optimized(response_time: float, source: str, company_id: str = None):
    performance_metrics["total_queries"] += 1
    performance_metrics["response_sources"][source] = performance_metrics["response_sources"].get(source, 0) + 1
    performance_metrics["response_times"].append(response_time)
    if len(performance_metrics["response_times"]) > 1000:
        performance_metrics["response_times"] = performance_metrics["response_times"][-500:]
    performance_metrics["avg_response_time"] = round(
        sum(performance_metrics["response_times"][-100:]) / min(100, len(performance_metrics["response_times"])), 3
    )
    if performance_metrics["total_queries"] % 50 == 0:
        performance_metrics["system_health"].update({
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "active_threads": len([t for t in Thread._instances if t.is_alive()]) if hasattr(Thread, '_instances') else 0
        })
    if company_id and performance_metrics["total_queries"] % 10 == 0:
        if company_id not in performance_metrics["companies"]:
            performance_metrics["companies"][company_id] = {"queries": 0, "avg_response_time": 0}
        company_metrics = performance_metrics["companies"][company_id]
        company_metrics["queries"] += 1
        if company_metrics["queries"] <= 10:
            company_metrics["avg_response_time"] = round(
                ((company_metrics["avg_response_time"] * (company_metrics["queries"] - 1)) + response_time) / company_metrics["queries"], 3
            )

# ==== QUERY PROCESSING ====
def process_query_optimized(query: str, company_id: str, session_id: str) -> dict:
    start_time = time.time()
    precomputed = get_precomputed_response(query)
    if precomputed:
        return {
            "answer": precomputed,
            "source": "precomputed",
            "response_time": time.time() - start_time,
            "session_id": session_id
        }
    cached_response = get_cached_response_optimized(query, company_id)
    if cached_response:
        cached_response.update({
            "cache_hit": True,
            "response_time": time.time() - start_time,
            "session_id": session_id
        })
        return cached_response
    try:
        if not vectorstore:
            return {
                "answer": get_smart_business_response(query, company_id),
                "source": "business_fallback",
                "fallback_used": True,
                "session_id": session_id,
                "response_time": time.time() - start_time
            }
        complexity = get_query_complexity_cached(query)
        optimal_llm = get_optimal_llm_fast(complexity)
        retriever = vectorstore.as_retriever(
            search_kwargs={
                "k": 5,
                "filter": {"company_id_slug": company_id}
            }
        )
        memory = get_session_memory_optimized(session_id)
        qa = ConversationalRetrievalChain.from_llm(
            optimal_llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            verbose=False
        )
        result = qa.invoke({"question": query})
        answer = clean_text_cached(result.get("answer", ""))
        if len(answer) < 20 or any(phrase in answer.lower() for phrase in ["don't know", "no information"]):
            answer = get_smart_business_response(query, company_id)
            source = "business_intelligence"
            fallback_used = True
        else:
            source = "sop"
            fallback_used = False
        if len(answer.split()) > 100:
            answer = " ".join(answer.split()[:85]) + "... Would you like me to continue with more details?"
        response = {
            "answer": answer,
            "source": source,
            "fallback_used": fallback_used,
            "model_used": optimal_llm.model_name,
            "complexity": complexity,
            "session_id": session_id,
            "response_time": time.time() - start_time,
            "source_documents": len(result.get("source_documents", []))
        }
        cache_response_optimized(query, company_id, response)
        return response
    except Exception as e:
        logger.error(f"Query processing error: {e}")
        return {
            "answer": get_smart_business_response(query, company_id),
            "source": "error_fallback",
            "error": True,
            "session_id": session_id,
            "response_time": time.time() - start_time
        }

# ==== FLASK ROUTES ====
@app.route('/')
def home():
    return jsonify({
        'status': 'optimal',
        'service': 'OpsVoice RAG API - Performance Optimized',
        'version': '4.0.0-speed-optimized',
        'performance': {
            'avg_response_time': performance_metrics.get("avg_response_time", 0),
            'cache_hit_rate': round((performance_metrics.get("cache_hits", 0) / max(1, performance_metrics.get("total_queries", 1))) * 100, 2),
            'total_queries': performance_metrics.get("total_queries", 0)
        },
        'features': [
            'instant_precomputed_responses',
            'optimized_caching',
            'smart_business_intelligence', 
            'enhanced_conversation_continuity',
            'fast_model_selection',
            'thread_safe_operations',
            'system_health_monitoring'
        ]
    })

@app.route('/healthz', methods=['GET', 'OPTIONS'])
def healthz():
    if request.method == "OPTIONS":
        return "", 204
    return jsonify({
        "status": "optimal",
        "timestamp": time.time(),
        "performance": {
            "avg_response_time": performance_metrics.get("avg_response_time", 0),
            "cache_size": len(query_cache),
            "active_sessions": len(conversation_sessions),
            "system_health": performance_metrics["system_health"]
        },
        "optimization": {
            "precomputed_responses": len(precomputed_responses),
            "vectorstore_loaded": vectorstore is not None,
            "thread_pool_active": getattr(executor, "_threads", None) is not None
        }
    })

@app.route('/query', methods=['POST', 'OPTIONS'])
def query_sop():
    if request.method == "OPTIONS":
        return "", 204
    start_time = time.time()
    try:
        if not request.is_json:
            return jsonify({"error": "JSON required"}), 400
        payload = request.get_json() or {}
        query = payload.get("query", "").strip()
        company_id = payload.get("company_id_slug", "").strip()
        session_id = payload.get("session_id", f"{company_id}_{int(time.time())}")
        if not query or len(query) < 2:
            return jsonify({"error": "Query too short"}), 400
        if not company_id or len(company_id) < 3:
            return jsonify({"error": "Invalid company ID"}), 400
        response = process_query_optimized(query, company_id, session_id)
        if response["source"] not in ["precomputed", "cache"]:
            response["followups"] = generate_contextual_followups(query, response["answer"])
        update_metrics_optimized(response["response_time"], response["source"], company_id)
        return jsonify(response)
    except Exception as e:
        logger.error(f"Query endpoint error: {e}")
        response = {
            "answer": "I'm experiencing high load right now. Please try again in a moment.",
            "source": "system_busy",
            "error": True,
            "response_time": time.time() - start_time
        }
        update_metrics_optimized(response["response_time"], "error")
        return jsonify(response), 503

def generate_contextual_followups(query: str, answer: str) -> list:
    followups = []
    q_lower = query.lower()
    a_lower = answer.lower()
    if "procedure" in a_lower or "step" in a_lower:
        followups.append("Would you like more detailed steps?")
    if "policy" in a_lower:
        followups.append("Are there exceptions to this policy?")
    if "customer" in q_lower:
        followups.append("Need help with difficult customer situations?")
    if not followups:
        followups = [
            "Need more specific details?",
            "Want help with a related topic?",
            "Any other questions?"
        ]
    return followups[:3]

@app.route('/continue', methods=['POST', 'OPTIONS'])
def continue_conversation():
    if request.method == "OPTIONS":
        return "", 204
    try:
        payload = request.get_json() or {}
        session_id = payload.get("session_id", "").strip()
        company_id = payload.get("company_id_slug", "").strip()
        if not session_id or not company_id:
            return jsonify({"error": "Session ID and company ID required"}), 400
        with session_lock:
            if session_id not in conversation_sessions:
                return jsonify({
                    "answer": "I don't have a previous conversation to continue from. What would you like to know?",
                    "source": "no_session",
                    "session_id": session_id
                })
            memory = conversation_sessions[session_id]['memory']
        if hasattr(memory, 'chat_memory') and memory.chat_memory.messages:
            last_ai_message = None
            for msg in reversed(memory.chat_memory.messages):
                if hasattr(msg, 'content') and len(msg.content) > 30:
                    last_ai_message = msg.content
                    break
            if last_ai_message and any(trigger in last_ai_message for trigger in [
                "Would you like me to continue", "Should I continue", "For complete details", "more details"
            ]):
                continue_query = "Please continue from where you left off with the complete information."
                response = process_query_optimized(continue_query, company_id, session_id)
                response["continued"] = True
                return jsonify(response)
        return jsonify({
            "answer": "I don't see a previous response that was truncated. What specific aspect would you like me to elaborate on?",
            "source": "no_continuation_needed",
            "session_id": session_id,
            "followups": ["Ask for more details on a topic", "Request a specific procedure", "Ask a new question"]
        })
    except Exception as e:
        logger.error(f"Continue conversation error: {e}")
        return jsonify({"error": "Continuation failed"}), 500

@app.route('/voice-reply', methods=['POST', 'OPTIONS'])
def voice_reply():
    if request.method == "OPTIONS":
        return "", 204
    try:
        data = request.get_json() or {}
        text = clean_text_cached(data.get("query", ""))
        company_id = data.get("company_id_slug", "default")
        if not text:
            return jsonify({"error": "Text required"}), 400
        tts_text = text[:400] if len(text) > 400 else text
        content_hash = hashlib.md5(tts_text.encode()).hexdigest()[:12]
        cache_key = f"tts_{content_hash}.mp3"
        cache_path = os.path.join(AUDIO_CACHE_DIR, cache_key)
        if os.path.exists(cache_path):
            cache_age = time.time() - os.path.getmtime(cache_path)
            if cache_age < AUDIO_CACHE_TTL:
                return send_file(cache_path, mimetype="audio/mp3", as_attachment=False)
        def generate_audio():
            try:
                response = requests.post(
                    "https://api.elevenlabs.io/v1/text-to-speech/tnSpp4vdxKPjI9w0GnoV/stream",
                    headers={
                        "xi-api-key": os.getenv("ELEVENLABS_API_KEY"),
                        "Content-Type": "application/json"
                    },
                    json={
                        "text": tts_text,
                        "voice_settings": {
                            "stability": 0.6,
                            "similarity_boost": 0.8,
                            "style": 0.1,
                            "use_speaker_boost": True
                        },
                        "model_id": "eleven_multilingual_v2"
                    },
                    timeout=25,
                    stream=True
                )
                return response.content if response.status_code == 200 else None
            except:
                return None
        audio_data = generate_audio()
        if not audio_data:
            return jsonify({"error": "TTS temporarily unavailable"}), 503
        try:
            with open(cache_path, "wb") as f:
                f.write(audio_data)
        except:
            pass
        return send_file(io.BytesIO(audio_data), mimetype="audio/mp3", as_attachment=False)
    except Exception as e:
        logger.error(f"TTS error: {e}")
        return jsonify({"error": "TTS failed"}), 500

@app.route('/upload-sop', methods=['POST', 'OPTIONS'])
def upload_sop():
    if request.method == "OPTIONS":
        return "", 204
    try:
        company_id = request.form.get('company_id_slug', '').strip()
        if not company_id or len(company_id) < 3:
            return jsonify({"error": "Invalid company ID"}), 400
        file = request.files.get("file")
        if not file or not file.filename:
            return jsonify({"error": "No file uploaded"}), 400
        filename = secure_filename(file.filename)
        if not filename or '.' not in filename:
            return jsonify({"error": "Invalid filename"}), 400
        ext = filename.rsplit('.', 1)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            return jsonify({"error": f"Only {', '.join(ALLOWED_EXTENSIONS)} files allowed"}), 400
        file.seek(0, 2)
        size = file.tell()
        file.seek(0)
        if size > MAX_FILE_SIZE:
            return jsonify({"error": f"File too large (max {MAX_FILE_SIZE // 1024 // 1024}MB)"}), 400
        timestamp = int(time.time())
        safe_filename = f"{company_id}_{timestamp}_{filename}"
        save_path = os.path.join(SOP_FOLDER, safe_filename)
        file.save(save_path)
        metadata = {
            "title": request.form.get("doc_title", filename)[:200],
            "company_id_slug": company_id,
            "filename": safe_filename,
            "uploaded_at": timestamp,
            "file_size": size,
            "file_extension": ext
        }
        update_status_fast(safe_filename, {"status": "processing", **metadata})
        executor.submit(embed_sop_worker_optimized, save_path, metadata)
        clear_company_cache(company_id)
        return jsonify({
            "message": "Document uploaded and processing started",
            "doc_id": safe_filename,
            "company_id_slug": company_id,
            "status": "processing",
            "estimated_time": "30-90 seconds",
            **metadata
        }), 201
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({"error": "Upload failed"}), 500

# ==== BACKGROUND PROCESSING ====
def update_status_fast(filename: str, status: dict):
    try:
        if os.path.exists(STATUS_FILE):
            with open(STATUS_FILE, 'r') as f:
                data = json.load(f)
        else:
            data = {}
        data[filename] = {**status, "updated_at": time.time()}
        temp_file = STATUS_FILE + '.tmp'
        with open(temp_file, 'w') as f:
            json.dump(data, f, indent=2)
        os.rename(temp_file, STATUS_FILE)
    except Exception as e:
        logger.error(f"Status update error: {e}")

def embed_sop_worker_optimized(file_path: str, metadata: dict):
    filename = os.path.basename(file_path)
    try:
        logger.info(f"Starting embedding for {filename}")
        ext = metadata.get("file_extension", "")
        if ext == "docx":
            docs = UnstructuredWordDocumentLoader(file_path).load()
        elif ext == "pdf":
            docs = PyPDFLoader(file_path).load()
        elif ext == "txt":
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            from langchain.schema import Document
            docs = [Document(page_content=content, metadata={"source": file_path})]
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", "? ", "! ", "; ", " "]
        )
        chunks = text_splitter.split_documents(docs)
        company_id_slug = metadata.get("company_id_slug")
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "company_id_slug": company_id_slug,
                "filename": filename,
                "chunk_id": f"{filename}_{i}",
                "chunk_index": i,
                "total_chunks": len(chunks),
                "source": file_path,
                "uploaded_at": metadata.get("uploaded_at"),
                "title": metadata.get("title", filename)
            })
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if not vectorstore:
                    load_vectorstore_fast()
                db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding)
                db.add_documents(chunks)
                db.persist()
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(2 ** attempt)
        update_status_fast(filename, {
            "status": "embedded",
            "chunk_count": len(chunks),
            "processing_time": time.time() - metadata.get("uploaded_at", time.time()),
            **metadata
        })
        logger.info(f"Successfully embedded {len(chunks)} chunks from {filename}")
    except Exception as e:
        logger.error(f"Embedding error for {filename}: {e}")
        update_status_fast(filename, {
            "status": f"error: {str(e)}",
            "error_time": time.time(),
            **metadata
        })

def clear_company_cache(company_id: str):
    with cache_lock:
        keys_to_remove = [key for key in query_cache.keys() if company_id in str(key)]
        for key in keys_to_remove:
            del query_cache[key]

def load_vectorstore_fast():
    global vectorstore
    try:
        logger.info(f"🔍 Checking vectorstore directory: {CHROMA_DIR}")
        logger.info(f"🔍 Directory exists: {os.path.exists(CHROMA_DIR)}")
        if os.path.exists(CHROMA_DIR):
            dir_contents = os.listdir(CHROMA_DIR)
            logger.info(f"🔍 Vectorstore directory contents: {dir_contents}")
            if dir_contents:
                vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding)
                logger.info("✅ Vectorstore loaded successfully")
            else:
                logger.warning("⚠️ Vectorstore directory is empty")
                vectorstore = None
        else:
            logger.warning(f"⚠️ Vectorstore directory not found: {CHROMA_DIR}")
            vectorstore = None
    except Exception as e:
        logger.error(f"❌ Vectorstore load error: {e}")
        vectorstore = None

# ==== COMPANY DOCS ROUTE ====
@app.route('/company-docs/<company_id_slug>')
def company_docs(company_id_slug):
    try:
        if not company_id_slug or len(company_id_slug) < 3:
            return jsonify({"error": "Invalid company ID"}), 400
        debug_info = {
            "company_id_slug": company_id_slug,
            "status_file_path": STATUS_FILE,
            "status_file_exists": os.path.exists(STATUS_FILE),
            "data_path": DATA_PATH,
            "sop_folder": SOP_FOLDER
        }
        docs = get_company_documents_fast(company_id_slug)
        return jsonify({
            "documents": docs,
            "count": len(docs),
            "company_id": company_id_slug,
            "debug": debug_info
        })
    except Exception as e:
        logger.error(f"Company docs error: {e}")
        return jsonify({"error": "Failed to fetch documents", "debug": str(e)}), 500

def get_company_documents_fast(company_id_slug: str) -> list:
    try:
        logger.info(f"🔍 Getting documents for company: {company_id_slug}")
        logger.info(f"📄 Status file path: {STATUS_FILE}")
        logger.info(f"📄 Status file exists: {os.path.exists(STATUS_FILE)}")
    except:
        pass
    if not os.path.exists(STATUS_FILE):
        logger.warning(f"❌ Status file not found: {STATUS_FILE}")
        return []
    try:
        with open(STATUS_FILE, 'r') as f:
            data = json.load(f)
        logger.info(f"📄 Loaded {len(data)} documents from status file")
        docs = []
        for filename, metadata in data.items():
            if metadata.get("company_id_slug") == company_id_slug:
                doc_info = {
                    "filename": filename,
                    "title": metadata.get("title", filename),
                    "status": metadata.get("status", "unknown"),
                    "uploaded_at": metadata.get("uploaded_at"),
                    "file_size": metadata.get("file_size"),
                    "chunk_count": metadata.get("chunk_count", 0)
                }
                docs.append(doc_info)
        docs.sort(key=lambda x: x.get('uploaded_at', 0), reverse=True)
        logger.info(f"✅ Found {len(docs)} documents for company {company_id_slug}")
        return docs
    except Exception as e:
        logger.error(f"❌ Error in get_company_documents_fast: {e}")
        return []

@app.route('/metrics')
def get_metrics():
    try:
        total_queries = performance_metrics.get("total_queries", 0)
        cache_hits = performance_metrics.get("cache_hits", 0) + performance_metrics.get("precompute_hits", 0)
        return jsonify({
            "performance": {
                "total_queries": total_queries,
                "cache_hit_rate": round((cache_hits / max(1, total_queries)) * 100, 2),
                "avg_response_time": performance_metrics.get("avg_response_time", 0),
                "precompute_hits": performance_metrics.get("precompute_hits", 0)
            },
            "system": {
                "cache_size": len(query_cache),
                "active_sessions": len(conversation_sessions),
                "system_health": performance_metrics["system_health"],
                "vectorstore_loaded": vectorstore is not None
            },
            "optimization": {
                "model_usage": performance_metrics["model_usage"],
                "response_sources": performance_metrics["response_sources"]
            }
        })
    except Exception as e:
        logger.error(f"Metrics error: {e}")
        return jsonify({"error": "Metrics unavailable"}), 500

# ==== CLEANUP & MAINTENANCE ====
def optimized_cleanup():
    try:
        current_time = time.time()
        with session_lock:
            expired_sessions = [
                sid for sid, data in conversation_sessions.items()
                if current_time - data['last_accessed'] > SESSION_TTL
            ]
            for sid in expired_sessions:
                del conversation_sessions[sid]
        with cache_lock:
            expired_keys = [
                key for key, data in query_cache.items()
                if current_time - data['timestamp'] > QUERY_CACHE_TTL
            ]
            for key in expired_keys:
                del query_cache[key]
        if os.path.exists(AUDIO_CACHE_DIR):
            for filename in os.listdir(AUDIO_CACHE_DIR):
                filepath = os.path.join(AUDIO_CACHE_DIR, filename)
                try:
                    if current_time - os.path.getmtime(filepath) > AUDIO_CACHE_TTL:
                        os.remove(filepath)
                except:
                    pass
        if performance_metrics["total_queries"] % 100 == 0:
            gc.collect()
        logger.debug(f"Cleanup: removed {len(expired_sessions)} sessions, {len(expired_keys)} cache entries")
    except Exception as e:
        logger.error(f"Cleanup error: {e}")

def start_cleanup_timer():
    optimized_cleanup()
    Timer(1800, start_cleanup_timer).start()

# ==== ERROR HANDLERS ====
@app.errorhandler(400)
def bad_request(error):
    return jsonify({"error": "Invalid request", "code": 400}), 400

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not found", "code": 404}), 404

@app.errorhandler(429)
def rate_limited(error):
    return jsonify({"error": "Rate limited", "retry_after": 60}), 429

@app.errorhandler(500)
def server_error(error):
    logger.error(f"Server error: {error}")
    return jsonify({"error": "Service temporarily unavailable"}), 500

# ==== INITIALIZATION ====
def initialize_optimized_app():
    logger.info("🚀 Initializing OpsVoice API v4.0 - Performance Optimized")
    load_vectorstore_fast()
    initialize_precomputed_responses()
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'rb') as f:
                global query_cache
                query_cache = pickle.load(f)
                logger.info(f"Loaded {len(query_cache)} cached responses")
    except:
        pass
    try:
        if os.path.exists(METRICS_FILE):
            with open(METRICS_FILE, 'r') as f:
                saved_metrics = json.load(f)
                performance_metrics.update(saved_metrics)
    except:
        pass
    start_cleanup_timer()
    logger.info(f"✅ Data path: {DATA_PATH}")
    logger.info(f"✅ SOP folder: {SOP_FOLDER}")
    logger.info(f"✅ Chroma directory: {CHROMA_DIR}")
    logger.info(f"✅ Current working directory: {os.getcwd()}")
    logger.info(f"✅ Cache size: {len(query_cache)}")
    logger.info(f"✅ Precomputed responses: {len(precomputed_responses)}")
    logger.info(f"✅ Vectorstore: {'loaded' if vectorstore else 'not loaded'}")
    logger.info(f"✅ Thread pool: {executor._max_workers} workers")
    logger.info("🎯 Optimization features active:")
    logger.info("   • Instant precomputed responses")
    logger.info("   • Multi-level caching system")
    logger.info("   • Smart business intelligence")
    logger.info("   • Enhanced conversation memory")
    logger.info("   • Optimized model selection")
    logger.info("   • Background processing")
    logger.info("   • System health monitoring")
    logger.info("⚡ OpsVoice API ready for high-performance operation!")

# ==== MAIN ENTRYPOINT ====
if __name__ == '__main__' or os.environ.get("RENDER", "").lower() == "true":
    # On Render, __name__ may not be '__main__', so also check RENDER env
    initialize_optimized_app()
    port = int(os.environ.get('PORT', 10000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    logger.info(f"🚀 Starting OpsVoice API v4.0 on port {port}")
    logger.info(f"🎯 Target response time: <2 seconds (80% faster)")
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug,
        threaded=True,
        use_reloader=False
    )