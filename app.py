from flask import Flask, request, jsonify, send_from_directory, send_file, make_response
import os, glob, json, re, time, io, shutil, requests, hashlib
from dotenv import load_dotenv
from threading import Thread
from functools import lru_cache
from collections import OrderedDict
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

# ---- FIXED: Persistent Data Path (NO MORE DATA WIPING) ----
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
CHROMA_DIR = "/data/chroma_db"
AUDIO_CACHE_DIR = os.path.join(DATA_PATH, "audio_cache")
STATUS_FILE = os.path.join(SOP_FOLDER, "status.json")

# ---- CRITICAL FIX: Ensure directories exist WITHOUT deleting data ----
os.makedirs(SOP_FOLDER, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)
os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)

# Performance tracking
performance_metrics = {
    "total_queries": 0,
    "cache_hits": 0,
    "avg_response_time": 0,
    "model_usage": {"gpt-3.5-turbo": 0, "gpt-4": 0},
    "response_sources": {"sop": 0, "fallback": 0, "cache": 0, "error": 0, "general_business": 0}
}

# Enhanced caching with LRU
class LRUCache(OrderedDict):
    def __init__(self, maxsize=500):
        self.maxsize = maxsize
        super().__init__()
    
    def __setitem__(self, key, value):
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        if len(self) > self.maxsize:
            oldest = next(iter(self))
            del self[oldest]
    
    def get(self, key, default=None):
        if key in self:
            self.move_to_end(key)
            return self[key]
        return default

query_cache = LRUCache(MAX_CACHE_SIZE)

# Initialize embeddings and persistent vectorstore
embedding = OpenAIEmbeddings()
vectorstore = None
vectorstore_last_health_check = 0

# ---- Flask Setup ----
app = Flask(__name__)

# Enhanced CORS configuration
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
    word_count = len(words)
    
    # Simple queries: short, basic questions
    simple_indicators = [
        word_count <= 8,
        any(word in query.lower() for word in ['what', 'when', 'where', 'who', 'how many', 'is there']),
        query.endswith('?') and word_count <= 6,
        bool(re.match(r'^(what|where|when|who|how much|how many|is there|do we have)', query.lower()))
    ]
    
    # Complex queries: multi-part, analytical, or detailed
    complex_indicators = [
        word_count > 15,
        any(word in query.lower() for word in ['analyze', 'compare', 'explain why', 'walk me through', 'detailed', 'comprehensive']),
        query.count('?') > 1,
        query.count(',') > 2,
        any(word in query.lower() for word in ['and also', 'in addition', 'furthermore', 'moreover'])
    ]
    
    if sum(complex_indicators) >= 2:
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
        return ChatOpenAI(temperature=0, model="gpt-3.5-turbo", request_timeout=30)
    else:
        performance_metrics["model_usage"]["gpt-4"] += 1
        return ChatOpenAI(temperature=0, model="gpt-4", request_timeout=45)

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
    """Cache response with LRU eviction"""
    cache_key = get_cache_key(query, company_id)
    query_cache[cache_key] = {
        'response': response,
        'timestamp': time.time()
    }

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

def analyze_query_intent_with_llm(query: str, company_name: str) -> str:
    """Use LLM to analyze query intent and categorize it"""
    try:
        intent_prompt = f"""
        Analyze this business query and categorize it: "{query}"
        
        Categories: customer_service, financial_procedures, hr_onboarding, daily_operations, 
        safety_emergency, sales_process, inventory_management, quality_compliance, 
        marketing_communication, technology_it, general_business
        
        Respond with ONLY the category name.
        """
        
        llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
        response = llm.invoke(intent_prompt)
        
        category = response.content.strip().lower()
        valid_categories = ["customer_service", "financial_procedures", "hr_onboarding", 
                          "daily_operations", "safety_emergency", "sales_process", 
                          "inventory_management", "quality_compliance", "marketing_communication", 
                          "technology_it", "general_business"]
        
        if category in valid_categories:
            return category
        else:
            return "general_business"
            
    except Exception as e:
        logger.error(f"Intent analysis error: {e}")
        return "general_business"

def generate_fallback_response(query: str, company_name: str, query_category: str) -> str:
    """Generate intelligent fallback response based on query category"""
    category_prompts = {
        "customer_service": """
        Provide best practices for handling customer service situations, including:
        - De-escalation techniques
        - Active listening strategies
        - Problem resolution steps
        - When to involve management
        """,
        "financial_procedures": """
        Explain standard financial procedures and controls:
        - Cash handling best practices
        - Transaction documentation
        - Approval hierarchies
        - Audit trail maintenance
        """,
        "hr_onboarding": """
        Outline effective employee onboarding practices:
        - First day/week checklist
        - Essential paperwork
        - Training schedules
        - Mentorship programs
        """,
        "daily_operations": """
        Describe operational best practices:
        - Opening/closing procedures
        - Task prioritization
        - Team communication
        - Efficiency optimization
        """,
        "safety_emergency": """
        Detail emergency response protocols:
        - Emergency contact procedures
        - Evacuation protocols
        - First aid basics
        - Incident reporting
        """,
        "sales_process": """
        Explain effective sales techniques:
        - Customer engagement
        - Needs assessment
        - Solution presentation
        - Closing strategies
        """,
        "inventory_management": """
        Describe inventory control best practices:
        - Stock tracking methods
        - Reorder points
        - Loss prevention
        - Organization systems
        """,
        "general_business": """
        Provide general business guidance and best practices.
        """
    }
    
    try:
        base_prompt = category_prompts.get(query_category, category_prompts["general_business"])
        
        fallback_prompt = f"""
        A user from {company_name} asked: "{query}"
        
        Since I don't have their specific company procedures, provide general business best practices.
        
        {base_prompt}
        
        Start with: "I don't see specific procedures for {company_name}, but here's general business best practice:"
        
        Make it practical, actionable, and conversational. Limit to 2-3 paragraphs.
        """
        
        llm = ChatOpenAI(temperature=0.3, model="gpt-3.5-turbo")
        response = llm.invoke(fallback_prompt)
        
        return clean_text(response.content)
        
    except Exception as e:
        logger.error(f"Fallback response error: {e}")
        return f"""I don't see specific procedures for {company_name}, but here's general business best practice:

For most business situations, focus on clear communication, documentation, and following established hierarchies. 
Listen carefully to understand the full situation, document important details, and escalate to management when needed.

Would you like me to help you find your company's specific procedures?"""

# ---- ENHANCED: General Business Intelligence System ----
def get_general_business_response(query: str, company_id: str = None) -> dict:
    """
    Intelligent general business response system using ChatGPT
    Works even when no company documents are uploaded
    """
    try:
        # Analyze query intent
        company_name = company_id.replace('-', ' ').replace('_', ' ').title() if company_id else "your company"
        query_category = analyze_query_intent_with_llm(query, company_name)
        
        # Create a comprehensive business prompt
        business_prompt = f"""
        You are an expert business consultant and advisor. A user asked: "{query}"
        
        This appears to be a {query_category.replace('_', ' ')} question.
        
        Provide a helpful, professional business response that includes:
        1. Direct answer to their question with practical advice
        2. Best practices and industry standards
        3. Step-by-step guidance when applicable
        4. Common pitfalls to avoid
        5. Resources or next steps they should consider
        
        Make it actionable and specific. Keep it conversational but professional.
        Limit to 2-3 paragraphs for clarity.
        """
        
        llm = ChatOpenAI(temperature=0.3, model="gpt-4")
        response = llm.invoke(business_prompt)
        
        business_answer = clean_text(response.content)
        
        # Generate contextual follow-ups for business topics
        followup_prompt = f"""
        Based on this {query_category.replace('_', ' ')} question: "{query}"
        
        Generate 3 relevant follow-up questions someone might ask related to this topic.
        Make them practical and actionable. Return as a simple list.
        """
        
        followup_response = ChatOpenAI(temperature=0.4, model="gpt-3.5-turbo").invoke(followup_prompt)
        followups = [line.strip("- ").strip() for line in followup_response.content.split('\n') if line.strip()][:3]
        
        return {
            "answer": business_answer,
            "followups": followups if followups else [
                "What are the key metrics I should track?",
                "How do I implement this in my business?",
                "What are common mistakes to avoid?"
            ],
            "source": "general_business_intelligence",
            "category": query_category
        }
        
    except Exception as e:
        logger.error(f"General business response error: {e}")
        
        # Fallback for AI errors
        return {
            "answer": """I'd be happy to help with general business questions! However, I'm having trouble generating a response right now.
            
For the best experience, try:
1. Upload your company documents so I can give specific guidance
2. Rephrase your question to be more specific
3. Ask about common business topics like customer service, operations, or policies""",
            "followups": [
                "How do I upload company documents?",
                "What business topics can you help with?",
                "Can you give general business advice?"
            ],
            "source": "fallback"
        }

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

# ---- Enhanced Demo Document Creation ----
def create_comprehensive_demo_document():
    """Create comprehensive demo document for demo-business-123"""
    demo_file = os.path.join(SOP_FOLDER, f"demo-business-123_{int(time.time())}_comprehensive_handbook.txt")
    
    demo_content = """DEMO COMPANY COMPREHENSIVE BUSINESS HANDBOOK

CUSTOMER SERVICE EXCELLENCE PROCEDURES:

When dealing with customers, our standard approach includes:
1. Greet customers warmly with genuine enthusiasm
2. Listen actively to understand their specific needs and concerns
3. If a customer is upset, angry, or frustrated, remain calm and empathetic
4. Ask clarifying questions to fully understand the situation
5. Offer multiple practical solutions and alternatives when possible
6. Follow up to ensure complete customer satisfaction
7. Escalate to management when issues exceed your authority
8. Document all interactions for continuous improvement

For difficult customer situations:
- Never take complaints personally
- Use phrases like "I understand your frustration" and "Let me help you solve this"
- Offer genuine apologies when appropriate
- Focus on solutions, not problems
- Know when to involve a supervisor

REFUND AND RETURN POLICY:

Our customer-friendly return policy includes:
- Full refunds available within 30 days with original receipt
- Manager approval required for refunds over $100
- Cash refunds provided for cash purchases only
- Credit card refunds processed to original payment method
- Store credit offered for returns after 30 days
- Exchanges allowed within 14 days for different sizes/colors
- Items must be in original condition with tags attached
- Defective products can be returned at any time with receipt

Special circumstances:
- Holiday purchases can be returned until January 31st
- Gift receipts extend return period to 60 days
- Damaged items receive immediate replacement or refund

EMPLOYEE ONBOARDING AND TRAINING:

First Week Schedule:
Day 1: Welcome, workspace tour, paperwork completion, handbook review
Day 2: Job-specific training begins, computer/system access setup
Day 3: Shadow experienced team member, begin hands-on learning
Day 4: Continue job training, first supervised customer interactions
Day 5: Week review, feedback session, goal setting for week 2

Onboarding Checklist:
✓ Complete all HR paperwork and tax forms
✓ Receive employee handbook and company policies
✓ Set up email account and system access
✓ Introduction to all team members and key personnel
✓ Assign buddy/mentor for first month
✓ Schedule initial training sessions based on role
✓ Provide workspace essentials and equipment
✓ Review emergency procedures and safety protocols

SAFETY AND EMERGENCY PROCEDURES:

Emergency Response Protocol:
- Call 911 immediately for medical emergencies, fires, or security threats
- Notify management as soon as it's safe to do so
- All emergency exits are clearly marked throughout the building
- Assembly point for evacuations is the parking lot across the street
- Fire drills conducted monthly on rotating schedules
- First aid kits located near each entrance and in break room
- Automated External Defibrillator (AED) located at front desk

Workplace Safety Guidelines:
- Report all accidents, injuries, and near-misses immediately
- Keep walkways and exits clear of obstructions
- Proper lifting techniques: bend knees, not back
- Report spills immediately and clean up promptly
- Wear appropriate footwear (no open-toed shoes)
- Follow lockout/tagout procedures for equipment maintenance

FINANCIAL PROCEDURES AND CASH HANDLING:

Daily Cash Management:
- Count cash drawer at beginning and end of each shift
- All transactions must be properly documented in POS system
- Manager override required for discounts over 20%
- Daily bank deposits made before 3 PM
- Cash drops to safe when drawer exceeds $200
- Never leave cash drawer unattended or open
- Two-person verification for large cash transactions

Expense and Purchasing:
- Expense reports due by 5th of following month
- Manager approval required for purchases over $50
- Petty cash fund maintained at $200 maximum
- All receipts must be submitted within 30 days
- Corporate credit card usage requires pre-approval
- Purchase orders needed for vendor payments over $500

COMMUNICATION AND TEAMWORK:

Internal Communication Standards:
- Check email and company messaging system at start of each shift
- Weekly team meetings every Monday at 9 AM
- Monthly all-hands meetings first Friday of each month
- Use professional language in all written communications
- Respond to internal messages within 4 hours during business hours
- Escalate urgent matters immediately to management

Customer Communication:
- Answer phones within 3 rings with standard greeting
- Return customer calls within 24 hours
- Email responses within 4 hours during business hours
- Use customers' names when possible during interactions
- Always end conversations by asking "Is there anything else I can help you with?"

TECHNOLOGY AND EQUIPMENT USAGE:

Computer and System Guidelines:
- Password changes required every 90 days
- Log out of all systems when leaving workstation
- No personal use of company computers during work hours
- Software installation requires IT approval
- Regular data backups performed automatically
- Report technical issues to IT helpdesk immediately

Mobile Device and Social Media:
- Personal phone use limited to breaks and emergencies
- Company information cannot be shared on social media
- Professional online presence expected for customer-facing roles
- No photography of customers or sensitive areas without permission

PERFORMANCE STANDARDS AND DEVELOPMENT:

Quality Expectations:
- Maintain professional appearance and demeanor at all times
- Arrive on time and ready to work
- Meet or exceed individual and team performance goals
- Participate actively in training and development opportunities
- Provide constructive feedback and suggestions for improvement
- Support teammates and contribute to positive work environment

Career Development:
- Annual performance reviews with goal setting
- Skills training opportunities available quarterly
- Tuition reimbursement program for job-related education
- Internal promotion preferred when positions become available
- Cross-training encouraged to develop versatile skill sets
- Mentoring program pairs new employees with experienced team members

This handbook serves as your guide to success at Demo Company. We're committed to providing excellent customer service while maintaining a positive, productive work environment for all team members."""

    with open(demo_file, 'w') as f:
        f.write(demo_content)
    
    # Update status
    metadata = {
        "title": "Demo Company Comprehensive Business Handbook",
        "company_id_slug": "demo-business-123",
        "filename": os.path.basename(demo_file),
        "uploaded_at": time.time(),
        "status": "embedding...",
        "is_demo": True,
        "file_size": len(demo_content)
    }
    
    update_status(os.path.basename(demo_file), metadata)
    
    # Start embedding
    Thread(target=embed_sop_worker, args=(demo_file, metadata), daemon=True).start()
    
    logger.info(f"Created comprehensive demo document: {demo_file}")
    return demo_file

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
    """Ensure vectorstore is available with health checks"""
    global vectorstore, vectorstore_last_health_check
    
    # Health check every 5 minutes
    current_time = time.time()
    if current_time - vectorstore_last_health_check > 300:  # 5 minutes
        try:
            if vectorstore:
                # Test the connection
                test_results = vectorstore.similarity_search("health_check", k=1)
                vectorstore_last_health_check = current_time
                return True
            else:
                load_vectorstore()
                vectorstore_last_health_check = current_time
                return vectorstore is not None
        except Exception as e:
            logger.error(f"Vectorstore health check failed: {e}")
            load_vectorstore()
            vectorstore_last_health_check = current_time
            return vectorstore is not None
    
    return vectorstore is not None

def get_company_documents_internal(company_id_slug):
    """Get documents for a company"""
    if not os.path.exists(STATUS_FILE): 
        return []
    
    try:
        with open(STATUS_FILE, 'r') as f:
            data = json.load(f)
        
        company_docs = []
        for filename, metadata in data.items():
            if metadata.get("company_id_slug") == company_id_slug:
                safe_filename = secure_filename(filename)
                if safe_filename == filename:  # Only include safe filenames
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

# ---- Routes ----
@app.route("/")
def home(): 
    return safe_json_response({
        "status": "ok", 
        "message": "🚀 OpsVoice RAG API is live!",
        "version": "3.0.0-performance-optimized",
        "features": [
            "smart_model_selection",
            "intelligent_caching", 
            "persistent_vectorstore",
            "smart_truncation",
            "llm_based_fallback",
            "general_business_intelligence",
            "session_memory",
            "performance_tracking"
        ],
        "data_path": DATA_PATH,
        "persistent_storage": os.path.exists("/data"),
        "performance": {
            "cache_hit_rate": f"{(performance_metrics['cache_hits'] / max(1, performance_metrics['total_queries']) * 100):.1f}%",
            "avg_response_time": f"{performance_metrics['avg_response_time']:.2f}s"
        }
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

@app.route("/list-sops")
def list_sops():
    """List all uploaded SOP files"""
    docs = glob.glob(os.path.join(SOP_FOLDER, "*.docx")) + glob.glob(os.path.join(SOP_FOLDER, "*.pdf")) + glob.glob(os.path.join(SOP_FOLDER, "*.txt"))
    return safe_json_response({"files": [os.path.basename(f) for f in docs], "count": len(docs)})

@app.route("/static/sop-files/<path:filename>")
def serve_sop(filename): 
    """Serve SOP files with enhanced security"""
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
    """ENHANCED query processing with performance optimizations"""
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

        # PERFORMANCE OPTIMIZATION: Check cache first
        cached_response = get_cached_response(qtext, tenant)
        if cached_response:
            cached_response["session_id"] = session_id
            cached_response["cache_hit"] = True
            cached_response["response_time"] = round(time.time() - start_time, 2)
            update_metrics(time.time() - start_time, "cache")
            return safe_json_response(cached_response)

        # Ensure vectorstore is available
        if not ensure_vectorstore():
            # If vectorstore fails, use general business intelligence
            logger.warning("Vectorstore unavailable, using general business intelligence")
            general_response = get_general_business_response(qtext, tenant)
            general_response["session_id"] = session_id
            general_response["response_time"] = round(time.time() - start_time, 2)
            performance_metrics["response_sources"]["general_business"] += 1
            update_metrics(time.time() - start_time, "general_business")
            return safe_json_response(general_response)

        # Handle vague queries
        if is_vague(qtext):
            response = {
                "answer": "Could you please be more specific? What procedure or policy are you looking for?",
                "source": "clarify",
                "followups": ["How do I handle customer complaints?", "What's the refund procedure?", "Where are the training documents?"],
                "session_id": session_id,
                "response_time": round(time.time() - start_time, 2)
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
                "session_id": session_id,
                "response_time": round(time.time() - start_time, 2)
            }
            update_metrics(time.time() - start_time, "off_topic")
            return safe_json_response(response)

        # Check if company has documents
        company_docs = get_company_documents_internal(tenant)
        has_company_docs = len(company_docs) > 0

        # PERFORMANCE OPTIMIZATION: Get query complexity and optimal model
        complexity = get_query_complexity(qtext)
        optimal_llm = get_optimal_llm(complexity)
        expanded_query = expand_query_with_synonyms(qtext)
        
        logger.info(f"Processing query for {tenant}: {qtext[:50]}... (complexity: {complexity}, model: {optimal_llm.model_name}, has_docs: {has_company_docs})")
        
        company_answer = None
        
        if has_company_docs:
            # Try company-specific documents first
            try:
                retriever = vectorstore.as_retriever(
                    search_kwargs={
                        "k": 5,
                        "filter": {"company_id_slug": tenant}
                    }
                )
                
                # Test retrieval
                test_docs = retriever.get_relevant_documents(expanded_query)
                logger.info(f"Found {len(test_docs)} relevant company documents")
                
                if test_docs:
                    # Get session memory
                    memory = get_session_memory(session_id)
                    
                    # Create conversational chain
                    qa = ConversationalRetrievalChain.from_llm(
                        optimal_llm,
                        retriever=retriever,
                        memory=memory,
                        return_source_documents=True
                    )
                    
                    # Query the chain
                    result = qa.invoke({"question": expanded_query})
                    answer = clean_text(result.get("answer", ""))
                    source_docs = result.get("source_documents", [])
                    
                    logger.info(f"Company RAG answer length: {len(answer)} chars, source docs: {len(source_docs)}")

                    # Check if answer is helpful
                    if answer and not is_unhelpful_answer(answer) and len(answer.strip()) > 10:
                        # PERFORMANCE OPTIMIZATION: Smart truncation
                        if len(answer.split()) > 150:
                            answer = smart_truncate(answer, 150)
                            
                        company_answer = {
                            "answer": answer,
                            "fallback_used": False,
                            "followups": generate_contextual_followups(qtext, answer),
                            "source": "company_docs",
                            "source_documents": len(source_docs),
                            "session_id": session_id,
                            "model_used": optimal_llm.model_name,
                            "complexity": complexity,
                            "response_time": round(time.time() - start_time, 2)
                        }
                        
                        # Cache successful responses
                        cache_response(qtext, tenant, company_answer)
                        update_metrics(time.time() - start_time, "sop")
                        return safe_json_response(company_answer)
                        
            except Exception as company_rag_error:
                logger.error(f"Company RAG processing error: {company_rag_error}")
                # Continue to intelligent fallback
        
        # ENHANCED: Use LLM-based fallback for ALL companies
        logger.info(f"Using intelligent fallback system for query: {qtext}")
        
        # Analyze query intent and generate fallback
        company_name = tenant.replace('-', ' ').replace('_', ' ').title()
        query_category = analyze_query_intent_with_llm(qtext, company_name)
        fallback_answer = generate_fallback_response(qtext, company_name, query_category)
        
        # If fallback is good, use it
        if fallback_answer and len(fallback_answer) > 50:
            if len(fallback_answer.split()) > 150:
                fallback_answer = smart_truncate(fallback_answer, 150)
                
            fallback_response = {
                "answer": fallback_answer,
                "followups": generate_contextual_followups(qtext, fallback_answer),
                "source": "intelligent_fallback",
                "category": query_category,
                "session_id": session_id,
                "model_used": "gpt-3.5-turbo",
                "complexity": complexity,
                "response_time": round(time.time() - start_time, 2)
            }
            
            # Cache fallback responses too
            cache_response(qtext, tenant, fallback_response)
            update_metrics(time.time() - start_time, "general_business")
            performance_metrics["response_sources"]["general_business"] += 1
            return safe_json_response(fallback_response)
        
        # Final fallback: General business intelligence
        general_response = get_general_business_response(qtext, tenant)
        general_response["session_id"] = session_id
        general_response["response_time"] = round(time.time() - start_time, 2)
        
        # Add context about company documents
        if not has_company_docs:
            general_response["answer"] += f"\n\nTo get answers specific to your company's procedures, consider uploading your business documents, policies, and handbooks."
            general_response["followups"].append("How do I upload company documents?")
        else:
            general_response["answer"] += f"\n\nI also searched your {len(company_docs)} uploaded company documents but didn't find specific information about this topic."
        
        # Cache general business responses too
        cache_response(qtext, tenant, general_response)
        update_metrics(time.time() - start_time, "general_business")
        performance_metrics["response_sources"]["general_business"] += 1
        return safe_json_response(general_response)

    except Exception as e:
        logger.error(f"Query error: {traceback.format_exc()}")
        update_metrics(time.time() - start_time, "error")
        
        # Return user-friendly error with session preservation
        return safe_json_response({
            "answer": "I encountered an error processing your request. Please try asking your question differently.",
            "source": "error",
            "followups": ["Ask a different question", "Try a simpler query", "Check if your documents are uploaded"],
            "session_id": session_id,
            "response_time": round(time.time() - start_time, 2),
            "error_details": str(e) if app.debug else None
        }, 500)

@app.route("/voice-reply", methods=["POST", "OPTIONS"])
def voice_reply():
    """Convert text to speech with enhanced error handling"""
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
    """Get documents with enhanced security and error handling"""
    # Validate company ID
    if not validate_company_id(company_id_slug):
        return safe_json_response({"error": "Invalid company identifier"}, 400)
    
    company_docs = get_company_documents_internal(company_id_slug)
    return safe_json_response(company_docs)

@app.route("/continue", methods=["POST", "OPTIONS"])
def continue_conversation():
    """Continue conversation with proper error handling"""
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

# ---- Admin/Debug Routes (Restored) ----
@app.route("/debug/status")
def debug_status():
    """Debug endpoint for checking system status"""
    cache_hit_rate = (performance_metrics['cache_hits'] / max(1, performance_metrics['total_queries'])) * 100
    
    return safe_json_response({
        "data_path": DATA_PATH,
        "persistent_storage": os.path.exists("/data"),
        "vectorstore_loaded": vectorstore is not None,
        "total_files": len(glob.glob(os.path.join(SOP_FOLDER, "*.*"))),
        "demo_files": len([f for f in os.listdir(SOP_FOLDER) if f.startswith("demo-business-123_")]) if os.path.exists(SOP_FOLDER) else 0,
        "cache_size": len(query_cache),
        "cache_hit_rate": f"{cache_hit_rate:.1f}%",
        "active_sessions": len(conversation_sessions),
        "metrics": performance_metrics
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
def startup_initialization():
    """Initialize system on startup - NO DATA WIPING"""
    logger.info("=== OpsVoice API Startup (Performance Optimized) ===")
    logger.info(f"Data path: {DATA_PATH}")
    logger.info(f"Persistent storage: {os.path.exists('/data')}")
    
    # Load existing vectorstore (don't wipe it!)
    logger.info("Loading existing vector store...")
    load_vectorstore()
    
    # Check existing files
    existing_files = glob.glob(os.path.join(SOP_FOLDER, "*.*"))
    logger.info(f"Found {len(existing_files)} existing files")
    
    # Create demo document only if none exist for demo business
    demo_files = [f for f in existing_files if "demo-business-123" in f]
    if len(demo_files) == 0:
        logger.info("No demo files found, creating comprehensive demo document...")
        create_comprehensive_demo_document()
    else:
        logger.info(f"Found {len(demo_files)} existing demo files")
    
    logger.info("=== Startup Complete ===")
    logger.info("Performance features enabled:")
    logger.info("- Smart model selection (GPT-3.5 for simple, GPT-4 for complex)")
    logger.info("- Intelligent caching with LRU eviction")
    logger.info("- Persistent vectorstore connection")
    logger.info("- Smart response truncation")
    logger.info("- LLM-based fallback for all companies")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    
    # Initialize system without wiping data
    startup_initialization()
    
    logger.info(f"Starting Performance-Optimized OpsVoice API v3.0.0 on port {port}")
    logger.info("Target response time: <8 seconds")
    logger.info("🎯 Ready for demo and production use - works with or without uploaded documents!")
    app.run(host="0.0.0.0", port=port, debug=False)