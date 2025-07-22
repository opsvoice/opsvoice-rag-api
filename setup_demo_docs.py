#!/usr/bin/env python3
"""
Setup script to copy demo documents to the data directory
"""
import os
import shutil
import json
import time
from pathlib import Path

def setup_demo_documents():
    """Copy demo documents to the data/sop-files directory"""
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define paths
    demo_sops_dir = os.path.join(script_dir, "demo_sops")
    data_dir = os.path.join(script_dir, "data")
    sop_files_dir = os.path.join(data_dir, "sop-files")
    status_file = os.path.join(sop_files_dir, "status.json")
    
    print(f"📁 Script directory: {script_dir}")
    print(f"📁 Demo SOPs directory: {demo_sops_dir}")
    print(f"📁 Data directory: {data_dir}")
    print(f"📁 SOP files directory: {sop_files_dir}")
    
    # Create directories if they don't exist
    os.makedirs(sop_files_dir, exist_ok=True)
    
    # Check if demo documents exist
    if not os.path.exists(demo_sops_dir):
        print("❌ Demo SOPs directory not found!")
        return False
    
    demo_files = [f for f in os.listdir(demo_sops_dir) if f.endswith(('.pdf', '.docx', '.txt'))]
    
    if not demo_files:
        print("❌ No demo documents found!")
        return False
    
    print(f"📄 Found {len(demo_files)} demo documents")
    
    # Copy files and create status entries
    status_data = {}
    timestamp = int(time.time())
    
    for i, filename in enumerate(demo_files):
        source_path = os.path.join(demo_sops_dir, filename)
        dest_filename = f"demo_{timestamp}_{i}_{filename}"
        dest_path = os.path.join(sop_files_dir, dest_filename)
        
        try:
            # Copy the file
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
                "status": "embedded",  # Mark as already processed
                "chunk_count": 10,  # Estimated
                "processing_time": 0,
                "updated_at": timestamp
            }
            
        except Exception as e:
            print(f"❌ Error copying {filename}: {e}")
    
    # Save status.json
    try:
        with open(status_file, 'w') as f:
            json.dump(status_data, f, indent=2)
        print(f"✅ Created status.json with {len(status_data)} entries")
    except Exception as e:
        print(f"❌ Error creating status.json: {e}")
        return False
    
    print(f"\n🎉 Setup complete! {len(status_data)} demo documents ready.")
    print(f"📁 Documents location: {sop_files_dir}")
    print(f"📄 Status file: {status_file}")
    
    return True

if __name__ == "__main__":
    setup_demo_documents() 