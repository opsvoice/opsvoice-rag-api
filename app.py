from flask import Flask, request, jsonify
import os
import glob
import json
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

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

# Load vectorstore ONCE at startup
embedding = OpenAIEmbeddings()
vectorstore = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding,
)
# Limit memory/RAM use for retrieval
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature=0, max_tokens=512),
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
    # NOTE: Must manually run embed_sop.py + restart service to reload DB!
    return jsonify({"message": f"File {file.filename} uploaded! It will be embedded and searchable soon."})

@app.route("/query", methods=["POST"])
def query_sop():
    data = request.get_json()
    user_query = data.get("query")
    if not user_query:
        return jsonify({"error": "No query provided"}), 400
    try:
        sop_answer = qa_chain.invoke(user_query)
        # RetrievalQA result is usually a dict with 'result' or 'answer'
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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)










