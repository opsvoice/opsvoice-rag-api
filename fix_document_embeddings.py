#!/usr/bin/env python3
"""
Emergency document embedding fix
Embeds existing documents into ChromaDB vectorstore
"""
import os
import sys
import json
import time
from pathlib import Path

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

def fix_document_embeddings():
    """Embed existing documents into ChromaDB"""
    
    # Get paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    sop_files_dir = os.path.join(data_dir, "sop-files")
    chroma_dir = os.path.join(data_dir, "chroma_db")
    status_file = os.path.join(sop_files_dir, "status.json")
    
    print(f"🔧 Fixing document embeddings...")
    print(f"📁 Script directory: {script_dir}")
    print(f"📁 Data directory: {data_dir}")
    print(f"📁 SOP files: {sop_files_dir}")
    print(f"📁 ChromaDB: {chroma_dir}")
    
    # Check if status.json exists
    if not os.path.exists(status_file):
        print("❌ status.json not found!")
        return False
    
    # Load status data
    with open(status_file, 'r') as f:
        status_data = json.load(f)
    
    print(f"📄 Found {len(status_data)} documents in status.json")
    
    # Initialize embeddings and vectorstore
    embedding = OpenAIEmbeddings()
    
    # Create or load vectorstore
    if os.path.exists(chroma_dir):
        print(f"📊 Loading existing ChromaDB from {chroma_dir}")
        vectorstore = Chroma(persist_directory=chroma_dir, embedding_function=embedding)
    else:
        print(f"📊 Creating new ChromaDB at {chroma_dir}")
        vectorstore = Chroma(persist_directory=chroma_dir, embedding_function=embedding)
    
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", " "]
    )
    
    total_chunks = 0
    processed_files = 0
    
    # Process each document
    for filename, metadata in status_data.items():
        file_path = os.path.join(sop_files_dir, filename)
        
        if not os.path.exists(file_path):
            print(f"⚠️ File not found: {filename}")
            continue
        
        print(f"📖 Processing: {metadata.get('title', filename)}")
        
        try:
            # Load document
            if filename.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
                docs = loader.load()
            else:
                print(f"⚠️ Unsupported file type: {filename}")
                continue
            
            # Split into chunks
            chunks = text_splitter.split_documents(docs)
            
            # Add metadata to chunks
            company_id_slug = metadata.get("company_id_slug", "demo-business-123")
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "company_id_slug": company_id_slug,
                    "filename": filename,
                    "chunk_id": f"{filename}_{i}",
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "source": file_path,
                    "uploaded_at": metadata.get("uploaded_at"),
                    "title": metadata.get("title", filename)
                })
            
            # Add to vectorstore
            vectorstore.add_documents(chunks)
            vectorstore.persist()
            
            # Update status
            metadata["chunk_count"] = len(chunks)
            metadata["processing_time"] = time.time() - metadata.get("uploaded_at", time.time())
            metadata["updated_at"] = int(time.time())
            
            total_chunks += len(chunks)
            processed_files += 1
            
            print(f"✅ Embedded {len(chunks)} chunks from {filename}")
            
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            continue
    
    # Save updated status
    try:
        with open(status_file, 'w') as f:
            json.dump(status_data, f, indent=2)
        print(f"✅ Updated status.json")
    except Exception as e:
        print(f"❌ Error saving status.json: {e}")
    
    print(f"\n🎉 Embedding complete!")
    print(f"📊 Processed {processed_files} files")
    print(f"📄 Total chunks: {total_chunks}")
    print(f"📁 Vectorstore: {chroma_dir}")
    
    # Verify vectorstore
    try:
        collection = vectorstore._collection
        count = collection.count()
        print(f"📊 ChromaDB collection count: {count}")
    except Exception as e:
        print(f"⚠️ Could not verify ChromaDB count: {e}")
    
    return True

if __name__ == "__main__":
    fix_document_embeddings() 