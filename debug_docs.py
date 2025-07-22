#!/usr/bin/env python3
"""
Debug script to test document retrieval
"""
import os
import json
import sys

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def debug_document_retrieval():
    """Debug the document retrieval process"""
    
    # Get paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    sop_files_dir = os.path.join(data_dir, "sop-files")
    status_file = os.path.join(sop_files_dir, "status.json")
    
    print(f"🔍 Debugging document retrieval...")
    print(f"📁 Script directory: {script_dir}")
    print(f"📁 Data directory: {data_dir}")
    print(f"📁 SOP files: {sop_files_dir}")
    print(f"📄 Status file: {status_file}")
    
    # Check if status file exists
    print(f"📄 Status file exists: {os.path.exists(status_file)}")
    
    if not os.path.exists(status_file):
        print("❌ Status file not found!")
        return
    
    # Read status file
    try:
        with open(status_file, 'r') as f:
            data = json.load(f)
        print(f"✅ Status file loaded successfully")
        print(f"📄 Number of documents in status: {len(data)}")
        
        # List all documents
        for filename, metadata in data.items():
            print(f"📖 {filename}: {metadata.get('title', 'No title')} (company: {metadata.get('company_id_slug', 'No company')})")
        
        # Test company filtering
        company_id = "demo-business-123"
        docs = []
        for filename, metadata in data.items():
            if metadata.get("company_id_slug") == company_id:
                doc_info = {
                    "filename": filename,
                    "title": metadata.get("title", filename),
                    "status": metadata.get("status", "unknown"),
                    "uploaded_at": metadata.get("uploaded_at"),
                    "file_size": metadata.get("file_size"),
                    "chunk_count": metadata.get("chunk_count", 0)
                }
                docs.append(doc_info)
        
        print(f"\n🔍 Documents for company '{company_id}': {len(docs)}")
        for doc in docs:
            print(f"  - {doc['title']} (chunks: {doc['chunk_count']})")
        
    except Exception as e:
        print(f"❌ Error reading status file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_document_retrieval() 