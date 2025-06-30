import os
import glob
import json
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

import pdfplumber

# ---- Paths ----
DATA_PATH = "/data" if os.environ.get("RENDER", "") == "true" or "/data" in os.getcwd() else "."
SOP_FOLDER = os.path.join(DATA_PATH, "sop-files")
CHROMA_DIR = os.path.join(DATA_PATH, "chroma_db")
STATUS_FILE = os.path.join(SOP_FOLDER, "status.json")

os.makedirs(SOP_FOLDER, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

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

# ---- Embed All ----
all_docs = []
for fpath in glob.glob(os.path.join(SOP_FOLDER, "*.docx")) + glob.glob(os.path.join(SOP_FOLDER, "*.pdf")):
    try:
        ext = fpath.split('.')[-1].lower()
        if ext == "docx":
            docs = UnstructuredWordDocumentLoader(fpath).load()
        elif ext == "pdf":
            with pdfplumber.open(fpath) as pdf:
                text = "\n".join([page.extract_text() or "" for page in pdf.pages])
            docs = [{"page_content": text}]
        else:
            continue
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(docs)
        all_docs.extend(chunks)
        update_status(os.path.basename(fpath), "embedded")
        print(f"[EMBEDDED] {os.path.basename(fpath)}: {len(chunks)} chunks.")
    except Exception as e:
        update_status(os.path.basename(fpath), f"error: {str(e)}")
        print(f"[FAILED] {os.path.basename(fpath)}: {str(e)}")

embeddings = OpenAIEmbeddings()
Chroma.from_documents(all_docs, embedding=embeddings, persist_directory=CHROMA_DIR)
print(f"Embedded {len(all_docs)} total SOP chunks to ChromaDB!")

