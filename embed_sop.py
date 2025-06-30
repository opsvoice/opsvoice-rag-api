from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import os
import glob

# Make sure your OpenAI API key is set in your environment or directly here:
# openai_api_key = "sk-..."  # (uncomment and paste if not using env var)

# 1. Get all Word docs in sop-files folder
doc_paths = glob.glob("sop-files/*.docx")

all_docs = []
for path in doc_paths:
    loader = UnstructuredWordDocumentLoader(path)
    docs = loader.load()
    all_docs.extend(docs)

# 2. Split docs into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(all_docs)

# 3. Embed and save to ChromaDB
embeddings = OpenAIEmbeddings()  # will use env var OPENAI_API_KEY
persist_directory = "/data/chroma_db"
vectorstore = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=persist_directory)

print(f"Embedded {len(doc_paths)} SOPs and saved {len(chunks)} chunks to ChromaDB!")
