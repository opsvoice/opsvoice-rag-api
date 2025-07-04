from flask import Flask, request, jsonify, send_from_directory, send_file, make_response
import os, glob, json, re, time, io, requests
from threading import Thread
from langchain_community.document_loaders import UnstructuredWordDocumentLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from flask_cors import CORS

# ---- Paths ----
DATA_PATH = "/data"
SOP_FOLDER = os.path.join(DATA_PATH, "sop-files")
CHROMA_DIR = os.path.join(DATA_PATH, "chroma_db")
STATUS_FILE = os.path.join(SOP_FOLDER, "status.json")

# Create required folders
os.makedirs(SOP_FOLDER, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

embedding = OpenAIEmbeddings()
vectorstore = None  # loaded later

def embed_sop_worker(fpath, metadata=None):
    fname = os.path.basename(fpath)
    try:
        print(f"[WORKER] Embedding file: {fpath}")
        ext = fname.split('.')[-1].lower()
        if ext == "docx":
            docs = UnstructuredWordDocumentLoader(fpath).load()
        elif ext == "pdf":
            docs = PyPDFLoader(fpath).load()
        else:
            raise Exception("Unsupported file type")

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(docs)

        print(f"[WORKER] {fname}: {len(chunks)} chunks")
        if not chunks:
            raise Exception("No content extracted from file")

        if metadata:
            for chunk in chunks:
                chunk.metadata.update(metadata)

        db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding)
        db.add_documents(chunks)
        db.persist()

        update_status(fname, {"status": "embedded", **metadata})
        print(f"[WORKER] Embedded {fname} successfully.")
    except Exception as e:
        update_status(fname, {"status": f"error: {str(e)}", **(metadata or {})})
        print(f"[WORKER] ERROR embedding {fname}: {e}")

def update_status(filename, status):
    try:
        if os.path.exists(STATUS_FILE):
            with open(STATUS_FILE, "r") as f:
                status_dict = json.load(f)
        else:
            status_dict = {}

        if isinstance(status, dict):
            status_dict[filename] = status
        else:
            existing = status_dict.get(filename, {})
            if isinstance(existing, dict):
                existing["status"] = status
                status_dict[filename] = existing
            else:
                status_dict[filename] = {"status": status}

        with open(STATUS_FILE, "w") as f:
            json.dump(status_dict, f, indent=2)
    except Exception as e:
        print("Error updating status:", e)

def load_vectorstore():
    global vectorstore
    print("[INFO] Loading vectorstore from disk...")
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embedding,
    )
    print("[INFO] Vectorstore loaded.")

app = Flask(__name__)
CORS(app, origins=["https://opsvoice-widget.vercel.app", "http://localhost:3000"])

# ---- Mobile Audio Endpoint for Voice Assistant ----
@app.route('/voice-reply', methods=['POST', 'OPTIONS'])
def voice_reply():
    # Handle CORS preflight for browsers (mobile/desktop)
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        return response

    # Main POST logic
    data = request.get_json()
    query_text = data.get('query', '')

    # ElevenLabs API (always returns MP3 stream)
    el_resp = requests.post(
        'https://api.elevenlabs.io/v1/text-to-speech/YOUR_VOICE_ID/stream',
        headers={
            "xi-api-key": "YOUR_ELEVENLABS_API_KEY"
        },
        json={
            "text": query_text
        }
    )

    audio_bytes = io.BytesIO(el_resp.content)
    response = make_response(send_file(
        audio_bytes,
        mimetype="audio/mpeg",
        as_attachment=False,
        download_name="reply.mp3"
    ))
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    return response

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    return response

# Serve uploaded files for public access
@app.route("/static/sop-files/<path:filename>")
def serve_file(filename):
    return send_from_directory(SOP_FOLDER, filename)

@app.route("/")
def home():
    return "🚀 OpsVoice API (multi-tenant) is live!"

@app.route("/list-sops", methods=["GET"])
def list_sops():
    files = glob.glob(os.path.join(SOP_FOLDER, "*.docx")) + glob.glob(os.path.join(SOP_FOLDER, "*.pdf"))
    return jsonify(files)

@app.route("/upload-sop", methods=["POST"])
def upload_sop():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in request"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    ext = file.filename.split('.')[-1].lower()
    if ext not in ['docx', 'pdf']:
        return jsonify({"error": "File must be a .docx or .pdf"}), 400

    doc_title = request.form.get("doc_title") or file.filename
    company_id = request.form.get("company_id") or "00000000-0000-0000-0000-000000000000"

    print(f"[UPLOAD] File: {file.filename}")
    print(f"[UPLOAD] Title: {doc_title}")
    print(f"[UPLOAD] Company ID: {company_id}")

    save_path = os.path.join(SOP_FOLDER, file.filename)
    file.save(save_path)

    # ✅ Generate public file URL from static route
    base_url = request.host_url.rstrip("/")  # Automatically picks up Render domain
    sop_file_url = f"{base_url}/static/sop-files/{file.filename}"

    metadata = {
        "title": doc_title,
        "company_id": company_id,
        "status": "embedding..."
    }
    update_status(file.filename, metadata)

    Thread(target=embed_sop_worker, args=(save_path, metadata)).start()

    return jsonify({
        "message": f"File {file.filename} uploaded. Embedding in background.",
        "doc_title": doc_title,
        "company_id": company_id,
        "sop_file_url": sop_file_url
    })

# ---- Advanced /query endpoint ----
# ---------------------------------
query_counts = {}
RATE_LIMIT_PER_MIN = 15  # e.g., 15 queries per company_id per minute

def check_rate_limit(company_id):
    now = int(time.time() / 60)  # current minute as int
    key = f"{company_id}-{now}"
    query_counts.setdefault(key, 0)
    query_counts[key] += 1
    return query_counts[key] <= RATE_LIMIT_PER_MIN

COMPANY_PERSONALITY = {
    "nzIvTy1QAd2bQvhs4d5Y": "Cades Market: Friendly, straightforward, local grocery expertise.",
    # Add more company_id: "Brand personality/voice" here
}

def contains_sensitive(text):
    if not text:
        return False
    patterns = [
        r"\bssn\b|\bsocial security\b|\b\d{3}-\d{2}-\d{4}\b",         # SSN
        r"\b\d{5,}\b",                                                # suspicious long numbers
        r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",                         # phone numbers
        r"\$[\d,]+(\.\d{2})?",                                        # dollar amounts
        r"\bsalary\b|\bwage\b|\bcompensation\b",                      # salary/wage terms
        r"\bemail\b|\b@\w+\.\w+\b",                                   # emails
    ]
    combined = "|".join(patterns)
    return bool(re.search(combined, text, re.IGNORECASE))

def generate_followups(question, answer):
    base = [
        "Do you want to know more details?",
        "Would you like steps for a related task?",
        "Need help finding a specific document?"
    ]
    q = (question or "").lower()
    if "invoice" in q:
        base = ["Do you want steps to send an invoice?", "Need help editing invoices?"] + base
    if "onboard" in q or "hire" in q:
        base = ["Want to know about required documents for new hires?"] + base
    return base[:3]

def is_vague_query(query):
    return not query or len(query.strip().split()) < 3 or query.strip().endswith("?") == False

@app.route("/query", methods=["POST"])
def query_sop():
    global vectorstore
    if vectorstore is None:
        load_vectorstore()

    data = request.get_json()
    user_query = data.get("query", "")
    company_id = data.get("company_id", "")

    if not user_query or not company_id:
        return jsonify({"error": "Missing query or company_id"}), 400

    # 0. If user asks for list of documents, short-circuit to doc list!
    DOC_LIST_TRIGGERS = [
        "what are my uploaded sops", "list my documents", "list uploaded docs", 
        "list company documents", "show company docs", "what documents do i have"
    ]
    if user_query.strip().lower() in DOC_LIST_TRIGGERS:
        # return document list for that company
        if not os.path.exists(STATUS_FILE):
            return jsonify({"docs": []})
        with open(STATUS_FILE, "r") as f:
            status_dict = json.load(f)
        docs = []
        for fname, meta in status_dict.items():
            if meta.get("company_id") == company_id:
                docs.append({
                    "filename": fname,
                    "title": meta.get("title", fname),
                    "company_id": company_id,
                    "status": meta.get("status"),
                    "sop_file_url": f"{request.host_url.rstrip('/')}/static/sop-files/{fname}"
                })
        return jsonify({
            "answer": f"You have {len(docs)} uploaded documents.",
            "docs": docs,
            "source": "company_docs"
        })

    # 1. Rate Limit
    if not check_rate_limit(company_id):
        return jsonify({"error": "Too many requests. Please wait a minute before asking again."}), 429

    # 2. Company Brand Personality
    personality = COMPANY_PERSONALITY.get(company_id, "")

    # 3. Context-Aware Redirect (very off-topic queries)
    REDIRECT_TOPICS = ["gmail", "outlook", "email password", "reset password", "facebook", "amazon account", "personal bank"]
    if any(t in user_query.lower() for t in REDIRECT_TOPICS):
        return jsonify({
            "answer": "Sorry, that question is outside of your company documents. For this type of issue, please contact your IT department or use the official help portal.",
            "source": "redirect"
        })

    # 4. Clarity Prompt
    if is_vague_query(user_query):
        return jsonify({
            "answer": "Could you provide a little more detail? (For example, specify which process or document you’re referring to.)",
            "source": "clarify"
        })

    try:
        retriever = vectorstore.as_retriever(
            search_kwargs={
                "k": 3,
                "filter": {"company_id": company_id}
            }
        )
        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(temperature=0, max_tokens=512),
            chain_type="stuff",
            retriever=retriever
        )
        sop_answer = qa_chain.invoke(user_query)
        answer_text = sop_answer.get("result") if isinstance(sop_answer, dict) else str(sop_answer)

        # 5. Sensitive Guardrails
        if contains_sensitive(answer_text):
            return jsonify({
                "answer": "Sorry, this answer may contain sensitive or private information and can’t be provided by voice. Please contact your company admin for access.",
                "source": "sensitive_guard"
            })

        # 6. Quick Summary
        summary_msg = ""
        if answer_text and len(answer_text.split()) > 50:
            summary_msg = "Quick summary: " + " ".join(answer_text.split()[:30]) + "... Want more details? Just ask!"

        # 7. Best Practices Fallback
        if not answer_text or "don't know" in answer_text.lower() or "no information" in answer_text.lower():
            llm = ChatOpenAI(temperature=0, max_tokens=256)
            prompt = f"The company SOPs do not cover this. Please provide a general business best practice for: {user_query}"
            best_practice_answer = llm.invoke(prompt)
            bp_text = best_practice_answer.content if hasattr(best_practice_answer, "content") else str(best_practice_answer)
            return jsonify({
                "source": "general_best_practice",
                "answer": f"{personality} A general best practice is: {bp_text}\n\nIf you want more specifics, try rephrasing or uploading a document.",
                "followups": generate_followups(user_query, bp_text)
            })

        # 8. Brand Personality + Summary + Followup Suggestions
        response = {
            "source": "sop",
            "answer": (personality + " " if personality else "") + (summary_msg + "\n" if summary_msg else "") + answer_text,
            "followups": generate_followups(user_query, answer_text)
        }
        return jsonify(response)

    except Exception as e:
        print(f"[ERROR] Query failed: {e}")
        return jsonify({"error": "Query failed", "details": str(e)}), 500

@app.route("/sop-status", methods=["GET"])
def sop_status():
    if os.path.exists(STATUS_FILE):
        with open(STATUS_FILE, "r") as f:
            return jsonify(json.load(f))
    else:
        return jsonify({})

@app.route("/reload-db", methods=["POST"])
def reload_db():
    load_vectorstore()
    return jsonify({"message": "Vectorstore reloaded from disk."})

@app.route("/company-docs/<company_id>", methods=["GET"])
def company_docs(company_id):
    # Get status.json mapping of file info
    if not os.path.exists(STATUS_FILE):
        return jsonify([])

    with open(STATUS_FILE, "r") as f:
        status_dict = json.load(f)
    
    # Filter files belonging to the requested company_id
    docs = []
    for fname, meta in status_dict.items():
        if meta.get("company_id") == company_id:
            docs.append({
                "filename": fname,
                "title": meta.get("title", fname),
                "company_id": company_id,
                "status": meta.get("status"),
                "sop_file_url": f"{request.host_url.rstrip('/')}/static/sop-files/{fname}"
            })

    return jsonify(docs)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    load_vectorstore()
    app.run(host="0.0.0.0", port=port)
