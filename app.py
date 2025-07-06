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
    txt = re.sub(r"\s+", " ", txt or "")
    return txt.replace("\u2022", "-").replace("\t", " ").strip()

def is_unhelpful_answer(text):
    if not text or not text.strip(): return True
    low = text.lower()
    triggers = ["don't know", "no information", "i'm not sure", "sorry", "unavailable", "not covered"]
    return any(t in low for t in triggers) or len(low.split()) < 6

def contains_sensitive(text):
    patterns = [r"\bssn\b|\bsocial security\b|\d{3}-\d{2}-\d{4}", r"$\d+[\,\d]*(\.\d\d)?"]
    return bool(re.search("|".join(patterns), text, re.IGNORECASE))

def generate_followups(q):
    base = ["Do you want to know more details?", "Would you like steps for a related task?", "Need help finding a specific document?"]
    q = q.lower()
    if "invoice" in q: base.insert(0, "Do you want steps to send an invoice?")
    if "onboard" in q or "hire" in q: base.insert(0, "Want to know about new-hire paperwork?")
    return base[:3]

def is_vague(query): return len(query.split()) < 3 or not query.strip().endswith("?")

# ---- Embedding Worker ----
def embed_sop_worker(fpath, metadata=None):
    fname = os.path.basename(fpath)
    try:
        ext = fname.rsplit(".", 1)[-1].lower()
        docs = UnstructuredWordDocumentLoader(fpath).load() if ext == "docx" else PyPDFLoader(fpath).load()
        chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100).split_documents(docs)
        for c in chunks: c.metadata.update(metadata or {})
        db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding)
        db.add_documents(chunks)
        db.persist()
        update_status(fname, {"status": "embedded", **(metadata or {})})
    except Exception as e:
        update_status(fname, {"status": f"error: {e}", **(metadata or {})})

def update_status(filename, status):
    try:
        data = json.load(open(STATUS_FILE)) if os.path.exists(STATUS_FILE) else {}
        data[filename] = status
        with open(STATUS_FILE, "w") as f: json.dump(data, f, indent=2)
    except Exception as e:
        print(f"[STATUS] Error: {e}")

def load_vectorstore():
    global vectorstore
    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding)

# ---- Routes ----
@app.route("/")
def home(): return "🚀 OpsVoice RAG API is live!"

@app.route("/healthz")
def healthz(): return jsonify({"status": "ok"})

@app.route("/list-sops")
def list_sops():
    docs = glob.glob(os.path.join(SOP_FOLDER, "*.docx")) + glob.glob(os.path.join(SOP_FOLDER, "*.pdf"))
    return jsonify(docs)

@app.route("/static/sop-files/<path:filename>")
def serve_sop(filename): return send_from_directory(SOP_FOLDER, filename)

@app.route("/upload-sop", methods=["POST"])
def upload_sop():
    file = request.files.get("file")
    if not file or not file.filename: return jsonify({"error":"No file uploaded"}), 400
    ext = file.filename.rsplit(".",1)[-1].lower()
    if ext not in ("docx","pdf"): return jsonify({"error":"Only .docx/.pdf allowed"}), 400

    tenant = re.sub(r"[^\w\-]", "", request.form.get("company_id_slug", ""))
    title = request.form.get("doc_title", file.filename)
    save_p = os.path.join(SOP_FOLDER, file.filename)
    file.save(save_p)

    meta = {"title": title, "company_id_slug": tenant}
    update_status(file.filename, {"status":"embedding...", **meta})
    Thread(target=embed_sop_worker, args=(save_p, meta), daemon=True).start()

    return jsonify({
        "message": f"Uploaded {file.filename}, embedding in background.",
        "doc_title": title,
        "company_id_slug": tenant,
        "sop_file_url": f"{request.host_url.rstrip('/')}/static/sop-files/{file.filename}"
    })

query_counts = {}
RATE_LIMIT_MIN = 15
COMPANY_VOICES = {
    "jaxdude-3057": "JaxDude: Straight to the point and helpful."
}

@app.route("/query", methods=["POST"])
def query_sop():
    global vectorstore
    if vectorstore is None: load_vectorstore()

    payload = request.get_json() or {}
    qtext = clean_text(payload.get("query", ""))
    tenant = re.sub(r"[^\w\-]", "", payload.get("company_id_slug", ""))

    if not qtext or not tenant:
        return jsonify({"error":"Missing query or tenant"}), 400

    if not check_rate_limit(tenant):
        return jsonify({"error":"Too many requests, try again in a minute."}), 429

    if is_vague(qtext):
        return jsonify({"answer":"Can you give me more detail—like the specific SOP or process you’re referring to?","source":"clarify"})

    if any(t in qtext.lower() for t in ["gmail","facebook","amazon account"]):
        return jsonify({"answer":"That’s outside your SOPs—please use the official help portal.","source":"off_topic"})

    try:
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 5, "filter": {"company_id_slug": tenant}, "score_threshold": 0.5}
        )
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        qa = ConversationalRetrievalChain.from_llm(
            ChatOpenAI(temperature=0),
            retriever=retriever,
            memory=memory
        )
        result = qa.invoke({"question": qtext})
        answer = clean_text(result.get("answer", ""))

        if contains_sensitive(answer):
            return jsonify({"answer":"Sorry, that info is private—please contact your admin.","source":"sensitive"})

        if is_unhelpful_answer(answer):
            fallback = ChatOpenAI(temperature=0).invoke(
                f"Your company SOPs don’t cover this. Provide a helpful, business-best-practice response to: “{qtext}”"
            )
            fb_txt = clean_text(getattr(fallback, "content", str(fallback)))
            return jsonify({
                "answer": f"{COMPANY_VOICES.get(tenant,'')} {fb_txt}",
                "fallback_used": True,
                "followups": generate_followups(qtext),
                "source": "fallback"
            })

        if len(answer.split()) > 100:
            answer = "Let me give you a short summary. " + " ".join(answer.split()[:50]) + "..."

        return jsonify({
            "answer": f"{COMPANY_VOICES.get(tenant,'')} {answer}",
            "fallback_used": False,
            "followups": generate_followups(qtext),
            "source": "sop"
        })

    except Exception as e:
        return jsonify({"error": "Query failed", "details": str(e)}), 500

def check_rate_limit(tenant: str) -> bool:
    minute = int(time.time() // 60)
    key = f"{tenant}-{minute}"
    query_counts.setdefault(key, 0)
    query_counts[key] += 1
    return query_counts[key] <= RATE_LIMIT_MIN

@app.route("/voice-reply", methods=["POST", "OPTIONS"])
def voice_reply():
    # 🔧 CORS preflight response
    if request.method == "OPTIONS":
        response = make_response()
        response.headers["Access-Control-Allow-Origin"] = request.headers.get("Origin", "")
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        return response

    data = request.get_json() or {}
    text = clean_text(data.get("query", ""))
    if not text: return jsonify({"error": "Empty text"}), 400

    # 🔧 Generate cache key
    tenant = re.sub(r"[^\w\-]", "", data.get("company_id_slug", ""))
    cache_key = f"{tenant}_{hash(text)}.mp3"
    cache_path = os.path.join(AUDIO_CACHE_DIR, cache_key)

    # Serve from cache if available
    if os.path.exists(cache_path):
        return send_file(cache_path, mimetype="audio/mp3", as_attachment=False)

    # Generate new audio if not cached
    try:
        tts_resp = requests.post(
            "https://api.elevenlabs.io/v1/text-to-speech/tnSpp4vdxKPjI9w0GnoV/stream",
            headers={"xi-api-key": os.getenv("sk_0b8e72ddb1dcb2092c83077f7d9d7a3c06c2cf0965a02df9")},
            json={"text": text}
        )
        if tts_resp.status_code != 200:
            return jsonify({"error": "TTS service unavailable"}), 502

        # 🔧 Cache audio
        with open(cache_path, "wb") as f:
            f.write(tts_resp.content)

        return send_file(io.BytesIO(tts_resp.content), mimetype="audio/mp3", as_attachment=False)

    except Exception as e:
        return jsonify({"error": "TTS request failed", "details": str(e)}), 500

@app.route("/sop-status")
def sop_status():
    if os.path.exists(STATUS_FILE): return send_file(STATUS_FILE)
    return jsonify({})

@app.route("/lookup-slug")
def lookup_slug():
    email = request.args.get("email", "").strip().lower()
    if not email or not os.path.exists(STATUS_FILE):
        return jsonify({"error": "Invalid email or missing status"}), 400

    data = json.load(open(STATUS_FILE))
    for meta in data.values():
        if meta.get("uploaded_by", "").strip().lower() == email:
            return jsonify({"slug": meta.get("company_id_slug")})

    return jsonify({"error": "Not found"}), 404

@app.route("/reload-db", methods=["POST"])
def reload_db():
    load_vectorstore()
    return jsonify({"message":"Vectorstore reloaded."})

@app.route("/company-docs/")
def company_docs(company_id_slug):
    if not os.path.exists(STATUS_FILE): return jsonify([])
    data = json.load(open(STATUS_FILE))
    return jsonify([
        {"filename":f, **m, "sop_file_url":f"{request.host_url}static/sop-files/{f}"}
        for f,m in data.items() if m.get("company_id_slug")==company_id_slug
    ])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    load_vectorstore()
    app.run(host="0.0.0.0", port=port)