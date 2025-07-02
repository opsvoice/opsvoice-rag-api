from flask import Flask, request, jsonify
import os, glob, json
from threading import Thread
from langchain_community.document_loaders import UnstructuredWordDocumentLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

# ---- Paths ----
DATA_PATH = "/data"
SOP_FOLDER = os.path.join(DATA_PATH, "sop-files")
CHROMA_DIR = os.path.join(DATA_PATH, "chroma_db")
STATUS_FILE = os.path.join(SOP_FOLDER, "status.json")

# Only create subfolders; /data already exists on Render disk mount
os.makedirs(SOP_FOLDER, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

embedding = OpenAIEmbeddings()
vectorstore = None  # will be loaded after embedding

def embed_sop_worker(fpath, metadata=None):
    """Background embedding worker, called with full file path and optional metadata."""
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

        # Attach metadata to each chunk (multi-tenant support)
        if metadata:
            for chunk in chunks:
                chunk.metadata.update(metadata)

        # Load (or create) Chroma, append new docs
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
        "company_id": company_id
    })

@app.route("/query", methods=["POST"])
def query_sop():
    global vectorstore
    if vectorstore is None:
        load_vectorstore()

    data = request.get_json()
    user_query = data.get("query")
    company_id = data.get("company_id")

    if not user_query or not company_id:
        return jsonify({"error": "Missing query or company_id"}), 400

    print(f"[QUERY] Company ID: {company_id}")
    print(f"[QUERY] Question: {user_query}")

    try:
        # Filter documents by metadata
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
        answer_text = ""

        if isinstance(sop_answer, dict):
            answer_text = sop_answer.get("result") or sop_answer.get("answer") or ""
        else:
            answer_text = str(sop_answer)

        if not answer_text or "don't know" in answer_text.lower() or "no information" in answer_text.lower():
            llm = ChatOpenAI(temperature=0, max_tokens=256)
            prompt = f"The company SOPs do not cover this. Please provide a general business best practice for: {user_query}"
            best_practice_answer = llm.invoke(prompt)
            bp_text = best_practice_answer.content if hasattr(best_practice_answer, "content") else str(best_practice_answer)
            return jsonify({
                "source": "general_best_practice",
                "answer": bp_text
            })
        else:
            return jsonify({
                "source": "sop",
                "answer": answer_text
            })

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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    load_vectorstore()
    app.run(host="0.0.0.0", port=port)

