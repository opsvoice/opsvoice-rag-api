from flask import Flask, request, jsonify, send_from_directory, send_file, make_response
import os, glob, json, re, time, io, shutil, requests
from dotenv import load_dotenv
from threading import Thread
from langchain_community.document_loaders import UnstructuredWordDocumentLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from flask_cors import CORS

load_dotenv()

# ---- Paths & Setup ----
DATA_PATH = "/data"
SOP_FOLDER = os.path.join(DATA_PATH, "sop-files")
CHROMA_DIR = os.path.join(DATA_PATH, "chroma_db")
AUDIO_CACHE_DIR = os.path.join(DATA_PATH, "audio_cache")
STATUS_FILE = os.path.join(SOP_FOLDER, "status.json")

os.makedirs(SOP_FOLDER, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)
os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)

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
# ✅ Allow specific origins with credentials
CORS(app, origins=["https://opsvoice-widget.vercel.app", "http://localhost:3000"], supports_credentials=True)

@app.after_request
def add_cors(response):
    # 🔧 Dynamically set origin from request headers
    origin = request.headers.get("Origin", "")
    allowed_origins = ["https://opsvoice-widget.vercel.app", "http://localhost:3000"]
    if origin in allowed_origins:
        response.headers["Access-Control-Allow-Origin"] = origin
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    return response

# ---- Utility Functions ----
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

def generate_followups(query):
    """Generate smarter contextual follow-up questions"""
    q = query.lower()
    base_followups = []
    
    # Context-specific followups
    if any(word in q for word in ["procedure", "process", "how"]):
        base_followups.extend([
            "Would you like the complete step-by-step procedure?",
            "Do you need information about related processes?"
        ])
    elif any(word in q for word in ["policy", "rule", "guideline"]):
        base_followups.extend([
            "Would you like to know about related policies?",
            "Do you need clarification on any specific requirements?"
        ])
    elif any(word in q for word in ["employee", "staff", "worker"]):
        base_followups.extend([
            "Do you need information about employee procedures?",
            "Would you like to know about training requirements?"
        ])
    elif any(word in q for word in ["time", "schedule", "hours"]):
        base_followups.extend([
            "Do you need information about scheduling procedures?",
            "Would you like details about time-off policies?"
        ])
    else:
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

# ---- Embedding Worker ----
def embed_sop_worker(fpath, metadata=None):
    """Background worker to embed documents"""
    fname = os.path.basename(fpath)
    try:
        ext = fname.rsplit(".", 1)[-1].lower()
        if ext == "docx":
            docs = UnstructuredWordDocumentLoader(fpath).load()
        elif ext == "pdf":
            docs = PyPDFLoader(fpath).load()
        else:
            raise ValueError(f"Unsupported file type: {ext}")
            
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=500, 
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""]
        ).split_documents(docs)
        
        # Add metadata to all chunks
        for c in chunks: 
            c.metadata.update(metadata or {})
            
        db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding)
        db.add_documents(chunks)
        db.persist()
        print(f"[EMBED] Successfully embedded {fname}")
        update_status(fname, {"status": "embedded", **(metadata or {})})
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

# ---- Routes ----
@app.route("/")
def home(): 
    return jsonify({
        "status": "ok", 
        "message": "🚀 OpsVoice RAG API is live!",
        "version": "1.0.1"
    })

@app.route("/healthz")
def healthz(): 
    return jsonify({
        "status": "healthy",
        "timestamp": time.time(),
        "vectorstore": "loaded" if vectorstore else "not_loaded"
    })

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
    """Enhanced query processing with better fallbacks"""
    global vectorstore
    if vectorstore is None: 
        load_vectorstore()

    payload = request.get_json() or {}
    qtext = clean_text(payload.get("query", ""))
    tenant = re.sub(r"[^\w\-]", "", payload.get("company_id_slug", ""))

    if not qtext or not tenant:
        return jsonify({"error": "Missing query or company_id_slug"}), 400

    # Rate limiting (simplified)
    if not check_rate_limit(tenant):
        return jsonify({"error": "Too many requests, try again in a minute."}), 429

    # Special handling for document listing queries
    doc_keywords = ['what documents', 'what files', 'what sops', 'uploaded documents', 'what do you have', 'what can you help']
    if any(keyword in qtext.lower() for keyword in doc_keywords):
        try:
            docs_response = requests.get(f"{request.host_url}company-docs/{tenant}")
            if docs_response.status_code == 200:
                docs = docs_response.json()
                if docs:
                    doc_titles = []
                    for doc in docs:
                        title = doc.get('title', doc.get('filename', 'Unknown Document'))
                        if title.endswith('.docx') or title.endswith('.pdf'):
                            title = title.rsplit('.', 1)[0]  # Remove extension
                        doc_titles.append(title)
                    
                    return jsonify({
                        "answer": f"I have access to {len(doc_titles)} documents: {', '.join(doc_titles)}. I can answer questions about any of these procedures and policies.",
                        "source": "document_list",
                        "followups": [
                            "Would you like details about any specific procedure?",
                            "Do you need help with a particular process?",
                            "What specific information are you looking for?"
                        ]
                    })
        except Exception as e:
            print(f"[DOC_LIST] Error: {e}")

    # Check for vague queries
    if is_vague(qtext):
        return jsonify({
            "answer": "Can you give me more detail—like the specific procedure or process you're referring to?",
            "source": "clarify",
            "followups": generate_followups(qtext)
        })

    # Filter out completely off-topic queries
    off_topic_keywords = ["weather", "news", "stock price", "sports", "celebrity", "movie"]
    if any(keyword in qtext.lower() for keyword in off_topic_keywords):
        return jsonify({
            "answer": "I'm focused on helping with your business procedures and operations. Please ask about your company's SOPs, policies, or general business questions.",
            "source": "off_topic"
        })

    try:
        # Set up retriever with company filtering
        retriever = vectorstore.as_retriever(
            search_kwargs={
                "k": 7,  # More documents for better context
                "filter": {"company_id_slug": tenant}
            }
        )
        
        # Create conversational chain
        memory = ConversationBufferMemory(
            memory_key="chat_history", 
            return_messages=True,
            output_key="answer"
        )
        
        qa = ConversationalRetrievalChain.from_llm(
            ChatOpenAI(temperature=0, model="gpt-4"),
            retriever=retriever,
            memory=memory,
            return_source_documents=True
        )
        
        # Query the chain
        result = qa.invoke({"question": qtext})
        answer = clean_text(result.get("answer", ""))

        # Skip sensitivity check for MVP
        # if contains_sensitive(answer):
        #     return jsonify({"answer": "Sorry, that information contains sensitive data—please contact your admin directly.", "source": "sensitive"})

        # Check if answer is helpful
        if is_unhelpful_answer(answer):
            # Enhanced fallback with business context
            try:
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
                
                fallback = ChatOpenAI(temperature=0.3, model="gpt-4").invoke(fallback_prompt)
                fallback_text = clean_text(getattr(fallback, "content", str(fallback)))
                
                return jsonify({
                    "answer": fallback_text,
                    "fallback_used": True,
                    "followups": [
                        "Would you like me to help you create documentation for this?",
                        "Do you want to know about related procedures?",
                        "Need help with anything else?"
                    ],
                    "source": "business_fallback"
                })
            except Exception as e:
                print(f"[FALLBACK] Error: {e}")
                return jsonify({
                    "answer": "I don't have specific information about that in your SOPs, but I'd be happy to help if you can provide more context or ask about other procedures I have access to.",
                    "source": "no_info",
                    "followups": generate_followups(qtext)
                })

        # Truncate long answers for voice
        if len(answer.split()) > 80:
            answer = "Here's a summary: " + " ".join(answer.split()[:70]) + "... For complete details, you can ask for more specifics."

        return jsonify({
            "answer": answer,
            "fallback_used": False,
            "followups": generate_followups(qtext),
            "source": "sop",
            "source_documents": len(result.get("source_documents", []))
        })

    except Exception as e:
        print(f"[QUERY] Error: {e}")
        # Better error handling
        return jsonify({
            "answer": "I'm having trouble accessing the information right now. Please try rephrasing your question or ask about a different topic.",
            "error": "Query processing failed", 
            "source": "error",
            "followups": ["Try asking about a specific procedure", "Rephrase your question", "Ask about available documents"]
        })

@app.route("/voice-reply", methods=["POST", "OPTIONS"])
def voice_reply():
    """Convert text to speech for voice responses"""
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

    # Generate cache key
    tenant = re.sub(r"[^\w\-]", "", data.get("company_id_slug", ""))
    cache_key = f"{tenant}_{hash(text)}.mp3"
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
    """Clear audio cache"""
    try:
        if os.path.exists(AUDIO_CACHE_DIR):
            for filename in os.listdir(AUDIO_CACHE_DIR):
                if filename.endswith('.mp3'):
                    os.remove(os.path.join(AUDIO_CACHE_DIR, filename))
        return jsonify({"message": "Audio cache cleared"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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
    
    print(f"[STARTUP] Starting OpsVoice API on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)