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

# ---- DEBUG: Print found files ----
print(f"Looking for .docx and .pdf in: {SOP_FOLDER}")
docx_files = glob.glob(os.path.join(SOP_FOLDER, "*.docx"))
pdf_files = glob.glob(os.path.join(SOP_FOLDER, "*.pdf"))
print("DOCX files found:", docx_files)
print("PDF files found:", pdf_files)

all_docs = []
for fpath in docx_files + pdf_files:
    try:
        ext = fpath.split('.')[-1].lower()
        print(f"Processing: {fpath} (ext: {ext})")
        if ext == "docx":
            docs = UnstructuredWordDocumentLoader(fpath).load()
        elif ext == "pdf":
            with pdfplumber.open(fpath) as pdf:
                text = "\n".join([page.extract_text() or "" for page in pdf.pages])
            docs = [{"page_content": text}]
        else:
            print("Skipped unsupported file type:", ext)
            continue
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(docs)
        print(f"Extracted {len(chunks)} chunks from {os.path.basename(fpath)}")
        all_docs.extend(chunks)
        update_status(os.path.basename(fpath), "embedded")
        print(f"[EMBEDDED] {os.path.basename(fpath)}: {len(chunks)} chunks.")
    except Exception as e:
        update_status(os.path.basename(fpath), f"error: {str(e)}")
        print(f"[FAILED] {os.path.basename(fpath)}: {str(e)}")

# ---- DEBUG: Check if any docs/chunks were extracted ----
print(f"Total chunks to embed: {len(all_docs)}")

if not all_docs:
    print("No documents to embed! Exiting early.")
    exit(0)

embeddings = OpenAIEmbeddings()
Chroma.from_documents(all_docs, embedding=embeddings, persist_directory=CHROMA_DIR)
print(f"Embedded {len(all_docs)} total SOP chunks to ChromaDB!")


