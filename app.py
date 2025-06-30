from flask import Flask, request, jsonify
import os
import json
import glob

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import pdfplumber

# ----- Path Setup -----
def get_persist_directory():
    if os.environ.get("RENDER", "") == "true" or "/data" in os.getcwd():
        return "/data/chroma_db"
    else:
        return os.path.join("chroma", "chroma_db")

def get_upload_folder():
    if os.environ.get("RENDER", "") == "true" or "/data" in os.getcwd():
        return "/data/sop-files/"
    else:
        return os.path.join("sop-files")

persist_directory = get_persist_directory()
UPLOAD_FOLDER = get_upload_folder()
STATUS_FILE = os.path.join(UPLOAD_FOLDER, "status.json")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(persist_directory, exist_ok=True)

# ----- Status Helper -----
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

def get_status():
    try:
        if os.path.exists(STATUS_FILE):
            with open(STATUS_FILE, "r") as f:
                status_dict = json.load(f)
        else:
            status_dict = {}
        return status_dict
    except Exception as e:
        print("Error reading status:", e)
        return {}

# ----- Embedding logic for a single doc/pdf upload -----
def embed_single_doc(file_path):
    file_ext = file_path.split('.')[-1].lower()
    docs = []

    if file_ext == "docx":
        loader = UnstructuredWordDocumentLoader(file_path)
        docs = loader.load()
    elif file_ext == "pdf":
        with pdfplumber.open(file_path) as pdf:
            text = "\n".join([page.extract_text() or "" for page in pdf.pages])
        if not text.strip():
            raise ValueError("PDF appears to be empty or not extractable.")
        # Wrap as LangChain Document type (minimal, for embedding)
        docs = [{"page_content": text}]
    else:
        raise ValueError("Unsupported file type.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    vectorstore.add_documents(chunks)
    vectorstore.persist()

# ----- Load ChromaDB vectorstore -----
embedding = OpenAIEmbeddings()
vectorstore = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding,
)
retriever = vectorstore.as_retriever()
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature=0),
    chain_type="stuff",
    retriever=retriever
)

app = Flask(__name__)

@app.route("/")
def home():
    return "🚀 OpsVoice API is live!"

@app.route("/list-sops", methods=["GET"])
def list_sops():
    files = glob.glob(os.path.join(UPLOAD_FOLDER, "*.docx")) + glob.glob(os.path.join(UPLOAD_FOLDER, "*.pdf"))
    return jsonify(files)

@app.route("/query", methods=["POST"])
def query_sop():
    data = request.get_json()
    user_query = data.get("query")
    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    sop_answer = qa_chain.invoke(user_query)
    if not sop_answer or "don't know" in sop_answer.lower() or "no information" in sop_answer.lower():
        llm = ChatOpenAI(temperature=0)
        prompt = f"The company SOPs do not cover this. Please provide a general business best practice for: {user_query}"
        best_practice_answer = llm.invoke(prompt)
        return jsonify({
            "source": "general_best_practice",
            "answer": best_practice_answer.content
        })
    else:
        return jsonify({
            "source": "sop",
            "answer": sop_answer
        })

@app.route("/voice-query", methods=["POST"])
def voice_query():
    return query_sop()

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

    save_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(save_path)
    update_status(file.filename, "processing")
    print(f"[UPLOAD] {file.filename} uploaded — status set to processing.")
    return jsonify({"message": f"File {file.filename} uploaded! It will be embedded and searchable soon."})

@app.route("/sop-status", methods=["GET"])
def sop_status():
    return jsonify(get_status())

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)









