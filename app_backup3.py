from flask import Flask, request, jsonify, send_from_directory, send_file, make_response
import os, glob, json, re, time, io, shutil, requests, hashlib, traceback, secrets
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
from datetime import datetime, timedelta

load_dotenv()

# ==== CONFIG/CONSTANTS ====
if os.path.exists("/data"):
    DATA_PATH = "/data"
else:
    DATA_PATH = os.path.join(os.getcwd(), "data")

SOP_FOLDER = os.path.join(DATA_PATH, "sop-files")
CHROMA_DIR = os.path.join(DATA_PATH, "chroma_db")
AUDIO_CACHE_DIR = os.path.join(DATA_PATH, "audio_cache")
STATUS_FILE = os.path.join(SOP_FOLDER, "status.json")

os.makedirs(SOP_FOLDER, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)
os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}
MAX_FILE_SIZE = 10 * 1024 * 1024

query_cache = OrderedDict()
MAX_CACHE_SIZE = 500
conversation_sessions = {}

performance_metrics = {
    "total_queries": 0,
    "cache_hits": 0,
    "avg_response_time": 0,
    "model_usage": {"gpt-3.5-turbo": 0, "gpt-4": 0},
    "response_sources": {"sop": 0, "fallback": 0, "cache": 0, "error": 0, "general_business": 0}
}

embedding = OpenAIEmbeddings()
vectorstore = None

# ==== FLASK APP ====
app = Flask(__name__)
ALLOWED_ORIGINS = [
    "https://opsvoice-widget.vercel.app",
    "http://localhost:3000",
    "http://127.0.0.1:3000"
]
CORS(app, resources={
    r"/*": {
        "origins": ALLOWED_ORIGINS,
        "supports_credentials": True
    }
})

@app.after_request
def after_request(response):
    origin = request.headers.get('Origin')
    if origin in ALLOWED_ORIGINS or "*" in ALLOWED_ORIGINS:
        response.headers['Access-Control-Allow-Origin'] = origin
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    if request.method == 'OPTIONS':
        response.headers['Access-Control-Max-Age'] = '3600'
        response.status_code = 200
    return response

# ==== UTILITY ====
def clean_text(txt: str) -> str:
    if not txt: return ""
    txt = re.sub(r"\s+", " ", txt)
    txt = txt.replace("\u2022", "-").replace("\t", " ")
    txt = re.sub(r"[*#]+", "", txt)
    txt = re.sub(r"\[.*?\]", "", txt)
    txt = re.sub(r"[<>\"'\x00\r\n]", "", txt)
    txt = txt.strip()
    return txt

def get_query_complexity(query: str) -> str:
    words = query.lower().split()
    simple = [
        len(words) <= 10,
        any(word in query.lower() for word in ['what', 'when', 'where', 'who', 'how many']),
        query.endswith('?') and len(words) <= 8
    ]
    complex_ = [
        len(words) > 15,
        any(word in query.lower() for word in ['analyze', 'compare', 'explain why', 'walk me through', 'break down']),
        query.count('?') > 1,
        any(word in query.lower() for word in ['because', 'therefore', 'however', 'although'])
    ]
    if sum(complex_) > 0: return "complex"
    elif sum(simple) >= 2: return "simple"
    else: return "medium"

def get_optimal_llm(complexity: str) -> ChatOpenAI:
    if complexity == "simple":
        performance_metrics["model_usage"]["gpt-3.5-turbo"] += 1
        return ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    else:
        performance_metrics["model_usage"]["gpt-4"] += 1
        return ChatOpenAI(temperature=0, model="gpt-4")

def get_cache_key(query: str, company_id: str) -> str:
    combined = f"{company_id}:{query.lower().strip()}"
    return hashlib.md5(combined.encode()).hexdigest()

def get_cached_response(query: str, company_id: str) -> dict:
    cache_key = get_cache_key(query, company_id)
    cached = query_cache.get(cache_key)
    if cached and time.time() - cached['timestamp'] < 3600:
        performance_metrics["cache_hits"] += 1
        performance_metrics["response_sources"]["cache"] += 1
        return cached['response']
    return None

def cache_response(query: str, company_id: str, response: dict):
    cache_key = get_cache_key(query, company_id)
    query_cache[cache_key] = {'response': response, 'timestamp': time.time()}
    while len(query_cache) > MAX_CACHE_SIZE:
        query_cache.popitem(last=False)

def update_metrics(response_time: float, source: str):
    performance_metrics["total_queries"] += 1
    performance_metrics["response_sources"][source] = performance_metrics["response_sources"].get(source, 0) + 1
    current_avg = performance_metrics["avg_response_time"]
    total_queries = performance_metrics["total_queries"]
    new_avg = ((current_avg * (total_queries - 1)) + response_time) / total_queries
    performance_metrics["avg_response_time"] = round(new_avg, 2)

def is_unhelpful_answer(text):
    if not text or not text.strip(): return True
    low = text.lower()
    triggers = [
        "don't know", "no information", "i'm not sure", "sorry",
        "unavailable", "not covered", "cannot find", "no specific information",
        "not mentioned", "doesn't provide", "no details", "not included",
        "context provided does not include", "text does not provide"
    ]
    has_trigger = any(t in low for t in triggers)
    is_too_short = len(low.split()) < 8
    return has_trigger or is_too_short

def generate_contextual_followups(query: str, answer: str) -> list:
    q = query.lower()
    a = answer.lower()
    base_followups = []
    if any(word in a for word in ["step", "procedure", "process"]):
        base_followups.append("Would you like the complete step-by-step procedure?")
    if any(word in a for word in ["policy", "rule", "requirement"]):
        base_followups.append("Do you need to know about related policies?")
    if any(word in a for word in ["form", "document", "paperwork"]):
        base_followups.append("Do you need help finding the actual forms?")
    if any(word in q for word in ["employee", "staff", "worker"]):
        base_followups.append("Do you need information about employee procedures?")
    elif any(word in q for word in ["time", "schedule", "hours"]):
        base_followups.append("Would you like details about scheduling policies?")
    elif any(word in q for word in ["customer", "client"]):
        base_followups.append("Do you need customer service procedures?")
    if not base_followups:
        base_followups.extend([
            "Do you want to know more details?",
            "Would you like steps for a related task?",
            "Need help finding a specific document?"
        ])
    return base_followups[:3]

def is_vague(query): 
    if not query or len(query.strip()) < 3: return True
    if len(query.split()) < 2: return True
    greetings = ['hello', 'hi', 'hey', 'good morning', 'good afternoon']
    if any(greeting in query.lower() for greeting in greetings) and '?' not in query:
        return True
    return False

def validate_file_upload(file):
    if not file or not file.filename: return False, "No file uploaded"
    filename = secure_filename(file.filename)
    if not filename: return False, "Invalid filename"
    if '.' not in filename: return False, "File must have extension"
    ext = filename.rsplit('.', 1)[1].lower()
    if ext not in ALLOWED_EXTENSIONS: return False, "Only PDF, DOCX, and TXT files allowed"
    file.seek(0, 2)
    size = file.tell()
    file.seek(0)
    if size > MAX_FILE_SIZE: return False, "File too large (max 10MB)"
    if size < 100: return False, "File appears to be empty or corrupted"
    return True, filename

def validate_company_id(company_id: str) -> bool:
    if not company_id or len(company_id) < 3 or len(company_id) > 50: return False
    if not re.match(r'^[a-zA-Z0-9_-]+$', company_id): return False
    dangerous_patterns = ['..', '/', '\\', '<', '>', '"', "'", '&', '|', ';']
    if any(pattern in company_id for pattern in dangerous_patterns): return False
    return True

def update_status(filename, status):
    try:
        data = json.load(open(STATUS_FILE)) if os.path.exists(STATUS_FILE) else {}
        data[filename] = {**status, "updated_at": time.time()}
        with open(STATUS_FILE, "w") as f: json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Error updating status for {filename}: {e}")

def load_vectorstore():
    global vectorstore
    try:
        vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding)
        print("[DB] Vector store loaded successfully")
    except Exception as e:
        print(f"[DB] Error loading vector store: {e}")
        vectorstore = None

def ensure_vectorstore():
    global vectorstore
    try:
        if not vectorstore:
            load_vectorstore()
        if vectorstore and hasattr(vectorstore, '_collection'):
            test_results = vectorstore.similarity_search("test", k=1)
            print(f"[DB] Vectorstore healthy, {len(test_results)} test results")
        return vectorstore is not None
    except Exception as e:
        print(f"[DB] Vectorstore health check failed: {e}")
        load_vectorstore()
        return vectorstore is not None

def get_company_documents_internal(company_id_slug):
    if not os.path.exists(STATUS_FILE): return []
    try:
        with open(STATUS_FILE, 'r') as f:
            data = json.load(f)
        company_docs = []
        for filename, metadata in data.items():
            if metadata.get("company_id_slug") == company_id_slug:
                safe_filename = secure_filename(filename)
                if safe_filename == filename:
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
        print(f"Error fetching docs for {company_id_slug}: {e}")
        return []

# ==== EMBEDDING WORKER ====
def embed_sop_worker(fpath, metadata=None):
    fname = os.path.basename(fpath)
    try:
        ext = fname.rsplit(".", 1)[-1].lower()
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
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200,
            separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]
        ).split_documents(docs)
        company_id_slug = metadata.get("company_id_slug") if metadata else None
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "company_id_slug": company_id_slug,
                "filename": fname,
                "chunk_id": f"{fname}_{i}",
                "source": fpath,
                "uploaded_at": metadata.get("uploaded_at", time.time()),
                "title": metadata.get("title", fname)
            })
        db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding)
        db.add_documents(chunks)
        db.persist()
        print(f"[EMBED] Successfully embedded {len(chunks)} chunks from {fname} for company {company_id_slug}")
        update_status(fname, {"status": "embedded", "chunk_count": len(chunks), **(metadata or {})})
    except Exception as e:
        print(f"[EMBED] Error embedding {fname}: {traceback.format_exc()}")
        update_status(fname, {"status": f"error: {str(e)}", **(metadata or {})})

# ==== ROUTES ====
@app.route("/")
def home(): 
    return jsonify({
        "status": "ok",
        "message": "🚀 OpsVoice RAG API is live!",
        "version": "3.0.0-performance-optimized-fixed",
        "features": [
            "smart_model_selection", "intelligent_caching",
            "persistent_vectorstore", "smart_truncation",
            "llm_based_fallback", "general_business_intelligence",
            "session_memory", "performance_tracking",
            "extensive_debugging"
        ]
    })

@app.route("/healthz")
def healthz():
    return jsonify({
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

@app.route("/list-sops")
def list_sops():
    docs = glob.glob(os.path.join(SOP_FOLDER, "*.docx")) + glob.glob(os.path.join(SOP_FOLDER, "*.pdf")) + glob.glob(os.path.join(SOP_FOLDER, "*.txt"))
    return jsonify({"files": [os.path.basename(f) for f in docs], "count": len(docs)})

@app.route("/static/sop-files/<path:filename>")
def serve_sop(filename):
    safe_filename = secure_filename(filename)
    if safe_filename != filename: return jsonify({"error": "Invalid filename"}), 400
    file_path = os.path.join(SOP_FOLDER, safe_filename)
    if not os.path.exists(file_path): return jsonify({"error": "File not found"}), 404
    try:
        return send_from_directory(SOP_FOLDER, safe_filename)
    except Exception as e:
        print(f"File serve error: {e}")
        return jsonify({"error": "File serve failed"}), 500

@app.route("/upload-sop", methods=["POST", "OPTIONS"])
def upload_sop():
    if request.method == "OPTIONS":
        return "", 204
    try:
        file = request.files.get("file")
        is_valid, result = validate_file_upload(file)
        if not is_valid: return jsonify({"error": result}), 400
        filename = result
        tenant = request.form.get("company_id_slug", "").strip()
        if not validate_company_id(tenant): return jsonify({"error": "Invalid company identifier"}), 400
        title = request.form.get("doc_title", filename)[:100]
        title = re.sub(r'[<>"\']', '', title)
        timestamp = int(time.time())
        safe_filename = f"{tenant}_{timestamp}_{filename}"
        save_path = os.path.join(SOP_FOLDER, safe_filename)
        file.save(save_path)
        if not os.path.exists(save_path) or os.path.getsize(save_path) == 0:
            return jsonify({"error": "File save verification failed"}), 500
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
            "upload_id": secrets.token_hex(6)
        })
    except Exception as e:
        print(f"[UPLOAD] Error: {e}")
        return jsonify({"error": "Failed to process upload", "details": str(e)}), 500

@app.route("/query", methods=["POST"])
def query_sop():
    start_time = time.time()
    if not ensure_vectorstore():
        update_metrics(time.time() - start_time, "error")
        return jsonify({"error": "Vectorstore unavailable"}), 503
    try:
        payload = request.get_json() or {}
        qtext = clean_text(payload.get("query", ""))
        tenant = payload.get("company_id_slug", "").strip()
        if not validate_company_id(tenant): return jsonify({"error": "Invalid company_id_slug"}), 400
        session_id = payload.get("session_id", f"{tenant}_{int(time.time())}")
        if not qtext: return jsonify({"error": "Missing query"}), 400

        cached_response = get_cached_response(qtext, tenant)
        if cached_response:
            update_metrics(time.time() - start_time, "cache")
            return jsonify(cached_response)

        # List documents if asked
        doc_keywords = ['what documents', 'what files', 'what sops', 'uploaded documents', 'what do you have', 'what can you help']
        if any(keyword in qtext.lower() for keyword in doc_keywords):
            docs = get_company_documents_internal(tenant)
            if docs:
                doc_titles = []
                for doc in docs:
                    title = doc.get('title', doc.get('filename', 'Unknown Document'))
                    if title.endswith('.docx') or title.endswith('.pdf'):
                        title = title.rsplit('.', 1)[0]
                    doc_titles.append(title)
                response = {
                    "answer": f"I have access to {len(doc_titles)} documents: {', '.join(doc_titles)}. I can answer questions about any of these procedures and policies.",
                    "source": "document_list",
                    "followups": [
                        "Would you like details about any specific procedure?",
                        "Do you need help with a particular process?",
                        "What specific information are you looking for?"
                    ]
                }
                cache_response(qtext, tenant, response)
                update_metrics(time.time() - start_time, "sop")
                return jsonify(response)

        if is_vague(qtext):
            response = {
                "answer": "Can you give me more detail—like the specific procedure or process you're referring to?",
                "source": "clarify",
                "followups": generate_contextual_followups(qtext, "")
            }
            update_metrics(time.time() - start_time, "clarify")
            return jsonify(response)

        off_topic_keywords = ["weather", "news", "stock price", "sports", "celebrity", "movie"]
        if any(keyword in qtext.lower() for keyword in off_topic_keywords):
            response = {
                "answer": "I'm focused on helping with your business procedures and operations. Please ask about your company's SOPs, policies, or general business questions.",
                "source": "off_topic"
            }
            update_metrics(time.time() - start_time, "off_topic")
            return jsonify(response)

        complexity = get_query_complexity(qtext)
        optimal_llm = get_optimal_llm(complexity)
        print(f"[QUERY] Using {optimal_llm.model_name} for {complexity} query: {qtext[:50]}...")

        retriever = vectorstore.as_retriever(
            search_kwargs={
                "k": 7,
                "filter": {"company_id_slug": tenant}
            }
        )
        memory = conversation_sessions.get(session_id)
        if not memory:
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
            conversation_sessions[session_id] = memory

        qa = ConversationalRetrievalChain.from_llm(
            optimal_llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True
        )
        result = qa.invoke({"question": qtext})
        answer = clean_text(result.get("answer", ""))

        if is_unhelpful_answer(answer):
            fallback_prompt = f"""
            The user asked: "{qtext}"

            This is a business assistant for a company. Their company SOPs don't cover this specific topic, 
            but provide a helpful, professional business response with general best practices, industry standards, 
            or actionable guidance. Keep it concise but valuable. Focus on:
            - Industry best practices
            - Professional recommendations  
            - Actionable steps
            - Business-focused advice

            If it's about procedures they don't have documented, suggest they create documentation for it.
            """
            fallback_llm = get_optimal_llm("medium")
            fallback = fallback_llm.invoke(fallback_prompt)
            fallback_text = clean_text(getattr(fallback, "content", str(fallback)))
            response = {
                "answer": fallback_text,
                "fallback_used": True,
                "followups": [
                    "Would you like me to help you create documentation for this?",
                    "Do you want to know about related procedures?",
                    "Need help with anything else?"
                ],
                "source": "business_fallback",
                "model_used": fallback_llm.model_name
            }
            cache_response(qtext, tenant, response)
            update_metrics(time.time() - start_time, "fallback")
            return jsonify(response)

        # Truncate for voice
        if len(answer.split()) > 80:
            answer = "Here's a summary: " + " ".join(answer.split()[:70]) + "... For complete details, you can ask for more specifics."
        response = {
            "answer": answer,
            "fallback_used": False,
            "followups": generate_contextual_followups(qtext, answer),
            "source": "sop",
            "source_documents": len(result.get("source_documents", [])),
            "model_used": optimal_llm.model_name,
            "session_id": session_id
        }
        cache_response(qtext, tenant, response)
        update_metrics(time.time() - start_time, "sop")
        return jsonify(response)
    except Exception as e:
        print(f"[QUERY] Error: {traceback.format_exc()}")
        response = {
            "answer": "I'm having trouble accessing the information right now. Please try rephrasing your question or ask about a different topic.",
            "error": "Query processing failed",
            "source": "error",
            "followups": [
                "Try asking about a specific procedure", "Rephrase your question", "Ask about available documents"
            ]
        }
        update_metrics(time.time() - start_time, "error")
        return jsonify(response)

@app.route("/voice-reply", methods=["POST", "OPTIONS"])
def voice_reply():
    if request.method == "OPTIONS":
        return "", 204
    data = request.get_json() or {}
    text = clean_text(data.get("query", ""))
    if not text: return jsonify({"error": "Empty text"}), 400
    content_hash = hashlib.md5(text.encode()).hexdigest()
    tenant = re.sub(r"[^\w\-]", "", data.get("company_id_slug", ""))
    cache_key = f"{tenant}_{content_hash}.mp3"
    cache_path = os.path.join(AUDIO_CACHE_DIR, cache_key)
    if os.path.exists(cache_path):
        print(f"[TTS] Serving cached audio: {cache_key}")
        return send_file(cache_path, mimetype="audio/mp3", as_attachment=False)
    try:
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
        audio_data = tts_resp.content
        with open(cache_path, "wb") as f: f.write(audio_data)
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
    if os.path.exists(STATUS_FILE):
        return send_file(STATUS_FILE)
    return jsonify({})

@app.route("/company-docs/<company_id_slug>")
def company_docs(company_id_slug):
    return jsonify(get_company_documents_internal(company_id_slug))

@app.route("/reload-db", methods=["POST"])
def reload_db():
    load_vectorstore()
    return jsonify({"message": "Vectorstore reloaded successfully."})

@app.route("/clear-cache", methods=["POST"])
def clear_cache():
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

@app.route("/clear-sessions", methods=["POST"])
def clear_sessions():
    session_count = len(conversation_sessions)
    conversation_sessions.clear()
    return jsonify({
        "message": f"Cleared {session_count} conversation sessions"
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print("[STARTUP] Loading vector store...")
    load_vectorstore()
    print(f"[STARTUP] Starting Optimized OpsVoice API v3.0.0 on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
