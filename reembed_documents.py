#!/usr/bin/env python3
"""
Simple build script for Render deployment
"""
import os
import json
import shutil
import time

def build_simple():
    """Simple document setup for deployment"""
    
    print("🚀 SIMPLE BUILD: Setting up demo documents")
    
    # Get paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    demo_sops_dir = os.path.join(script_dir, "demo_sops")
    
    # For Render deployment, use /data directory
    if os.path.exists("/data"):
        data_dir = "/data"
        print("📁 Using Render persistent disk: /data")
    else:
        data_dir = os.path.join(script_dir, "data")
        print(f"📁 Using local data directory: {data_dir}")
    
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
    
    # Create basic status.json (without embedding for now)
    status_data = {}
    timestamp = int(time.time())
    
    for i, filename in enumerate(demo_files):
        dest_filename = f"demo_{timestamp}_{i}_{filename}"
        dest_path = os.path.join(sop_files_dir, dest_filename)
        
        # Copy file
        source_path = os.path.join(demo_sops_dir, filename)
        shutil.copy2(source_path, dest_path)
        
        print(f"✅ Copied: {filename} -> {dest_filename}")
        
        # Create status entry
        status_data[dest_filename] = {
            "title": filename.replace('.pdf', '').replace('_', ' ').title(),
            "company_id_slug": "demo-business-123",
            "filename": dest_filename,
            "uploaded_at": timestamp,
            "file_size": os.path.getsize(dest_path),
            "file_extension": filename.split('.')[-1].lower(),
            "status": "ready_for_embedding",
            "chunk_count": 0,
            "processing_time": 0,
            "updated_at": timestamp
        }
    
    # Save status.json
    try:
        with open(status_file, 'w') as f:
            json.dump(status_data, f, indent=2)
        print(f"✅ Created status.json with {len(status_data)} entries")
        print(f"📄 Status file: {status_file}")
        return True
    except Exception as e:
        print(f"❌ Error creating status.json: {e}")
        return False

if __name__ == "__main__":
    build_simple()