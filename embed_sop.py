import os
import glob
import json
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# ---- Configurable paths ----
DATA_PATH = "/data"
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

# ---- Debug: print what's in the folder ----
try:
    print("Listing /data:", os.listdir(DATA_PATH))
    print("Listing /data/sop-files:", os.listdir(SOP_FOLDER))
except Exception as e:
    print(f"Error listing directory: {e}")

# ---- Find files ----
docx_files = glob.glob(os.path.join(SOP_FOLDER, "*.docx"))
pdf_files = glob.glob(os.path.join(SOP_FOLDER, "*.pdf"))

print("DOCX files found:", docx_files)
print("PDF files found:", pdf_files)

all_docs = []

def load_docx(fpath):
    return UnstructuredWordDocumentLoader(fpath).load()

def load_pdf(fpath):
    from langchain_community.document_loaders import PyPDFLoader
    loader = PyPDFLoader(fpath)
    return loader.load()

for fpath in docx_files + pdf_files:
    try:
        fname = os.path.basename(fpath)
        ext = fpath.split('.')[-1].lower()
        print(f"Processing: {fpath} (ext: {ext})")

        if ext == "docx":
            docs = load_docx(fpath)
        elif ext == "pdf":
            docs = load_pdf(fpath)
        else:
            print("Skipped unsupported file type:", ext)
            continue

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(docs)
        print(f"Extracted {len(chunks)} chunks from {fname}")
        all_docs.extend(chunks)
        update_status(fname, "embedded")
        print(f"[EMBEDDED] {fname}: {len(chunks)} chunks.")
    except Exception as e:
        update_status(fname, f"error: {str(e)}")
        print(f"[FAILED] {fname}: {str(e)}")

print(f"Total chunks to embed: {len(all_docs)}")

if not all_docs:
    print("No documents to embed! Exiting early.")
    exit(0)

# ---- Embedding ----
try:
    embeddings = OpenAIEmbeddings()
    Chroma.from_documents(all_docs, embedding=embeddings, persist_directory=CHROMA_DIR)
    print(f"Embedded {len(all_docs)} total SOP chunks to ChromaDB!")
except Exception as e:
    print(f"[EMBEDDING FAILED]: {e}")
    exit(1)



