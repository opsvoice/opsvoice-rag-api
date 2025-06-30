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

def embed_sop_worker(fpath):
    """Background embedding worker, called with full file path."""
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
        # Load (or create) Chroma, append new docs
        db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding)
        db.add_documents(chunks)
        db.persist()
        update_status(fname, "embedded")
        print(f"[WORKER] Embedded {fname} successfully.")
    except Exception as e:
        update_status(fname, f"error: {str(e)}")
        print(f"[WORKER] ERROR embedding {fname}: {e}")

def update_status(filename, status):
    try:
        if os.path.exists(STATUS_FILE):
            with open(STATUS_FILE, "r") as f:
                status_dict = json.load(f)
        else:
            status_dict = {}
        status_dict[filename] = status
        with open(STATUS_FILE, "w") as f:
            json.dump(status_dict, f)
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
    return "🚀 OpsVoice API (single service) is live!"

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
    ext = f












