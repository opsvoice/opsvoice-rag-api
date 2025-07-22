#!/usr/bin/env python3
"""
Build script for Render deployment
Embeds demo documents during build process
"""
import os
import sys
import json
import time
import shutil

def build_demo_documents():
    """Embed demo documents for deployment"""
    
    print("🚀 BUILD: Embedding Demo Documents")
    
    # Get paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    demo_sops_dir = os.path.join(script_dir, "demo_sops")
    
    # For Render deployment, use /data directory
    if os.path.exists("/data"):
        data_dir = "/data"
    else:
        data_dir = os.path.join(script_dir, "data")
    
    sop_files_dir = os.path.join(data_dir, "sop-files")
    chroma_dir = os.path.join(data_dir, "chroma_db")
    status_file = os.path.join(sop_files_dir, "status.json")
    
    # Create directories
    os.makedirs(sop_files_dir, exist_ok=True)
    os.makedirs(chroma_dir, exist_ok=True)
    
    # Check if demo documents exist
    if not os.path.exists(demo_sops_dir):
        print("❌ Demo SOPs directory not found!")
        return False
    
    demo_files = [f for f in os.listdir(demo_sops_dir) if f.endswith(('.pdf', '.docx', '.txt'))]
    
    if not demo_files:
        print("❌ No demo documents found!")
        return False
    
    print(f"📄 Found {len(demo_files)} demo documents")
    
    # Try to import required modules
    try:
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_community.vectorstores import Chroma
        from langchain_openai import OpenAIEmbeddings
        from dotenv import load_dotenv
        
        load_dotenv()
        
        # Clear existing ChromaDB
        if os.path.exists(chroma_dir):
            shutil.rmtree(chroma_dir)
            os.makedirs(chroma_dir, exist_ok=True)
        
        # Initialize embeddings and vectorstore
        embedding = OpenAIEmbeddings()
        vectorstore = Chroma(persist_directory=chroma_dir, embedding_function=embedding)
        
        # Initialize text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", "? ", "! ", "; ", " "]
        )
        
        # Process each demo document
        status_data = {}
        timestamp = int(time.time())
        total_chunks = 0
        
        for i, filename in enumerate(demo_files):
            source_path = os.path.join(demo_sops_dir, filename)
            dest_filename = f"demo_{timestamp}_{i}_{filename}"
            dest_path = os.path.join(sop_files_dir, dest_filename)
            
            print(f"📖 Processing: {filename}")
            
            try:
                # Copy file
                shutil.copy2(source_path, dest_path)
                
                # Load and process document
                if filename.endswith('.pdf'):
                    loader = PyPDFLoader(dest_path)
                    docs = loader.load()
                else:
                    continue
                
                # Split into chunks
                chunks = text_splitter.split_documents(docs)
                
                # Add metadata to chunks
                company_id_slug = "demo-business-123"
                for j, chunk in enumerate(chunks):
                    chunk.metadata.update({
                        "company_id_slug": company_id_slug,
                        "filename": dest_filename,
                        "chunk_id": f"{dest_filename}_{j}",
                        "chunk_index": j,
                        "total_chunks": len(chunks),
                        "source": dest_path,
                        "uploaded_at": timestamp,
                        "title": filename.replace('.pdf', '').replace('_', ' ').title()
                    })
                
                # Add to vectorstore
                vectorstore.add_documents(chunks)
                vectorstore.persist()
                
                # Create status entry
                status_data[dest_filename] = {
                    "title": filename.replace('.pdf', '').replace('_', ' ').title(),
                    "company_id_slug": company_id_slug,
                    "filename": dest_filename,
                    "uploaded_at": timestamp,
                    "file_size": os.path.getsize(dest_path),
                    "file_extension": filename.split('.')[-1].lower(),
                    "status": "embedded",
                    "chunk_count": len(chunks),
                    "processing_time": 0,
                    "updated_at": timestamp
                }
                
                total_chunks += len(chunks)
                print(f"✅ Embedded {len(chunks)} chunks from {filename}")
                
            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")
                continue
        
        # Save status.json
        with open(status_file, 'w') as f:
            json.dump(status_data, f, indent=2)
        
        print(f"🎉 BUILD COMPLETE: {len(status_data)} documents, {total_chunks} chunks")
        return True
        
    except ImportError as e:
        print(f"⚠️ Required modules not available: {e}")
        print("📝 Creating basic status.json...")
        
        # Create basic status.json without embedding
        status_data = {}
        timestamp = int(time.time())
        
        for i, filename in enumerate(demo_files):
            dest_filename = f"demo_{timestamp}_{i}_{filename}"
            dest_path = os.path.join(sop_files_dir, dest_filename)
            
            # Copy file
            source_path = os.path.join(demo_sops_dir, filename)
            shutil.copy2(source_path, dest_path)
            
            # Create status entry
            status_data[dest_filename] = {
                "title": filename.replace('.pdf', '').replace('_', ' ').title(),
                "company_id_slug": "demo-business-123",
                "filename": dest_filename,
                "uploaded_at": timestamp,
                "file_size": os.path.getsize(dest_path),
                "file_extension": filename.split('.')[-1].lower(),
                "status": "pending_embedding",
                "chunk_count": 0,
                "processing_time": 0,
                "updated_at": timestamp
            }
        
        # Save status.json
        with open(status_file, 'w') as f:
            json.dump(status_data, f, indent=2)
        
        print(f"✅ Created basic status.json with {len(status_data)} entries")
        return True

if __name__ == "__main__":
    build_demo_documents()