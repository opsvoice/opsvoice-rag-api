import json
import os
from datetime import datetime

STATUS_FILE = 'sop_processing_status.json'

def cleanup_error_documents():
    """Remove documents with database errors"""
    
    print("🧹 Cleaning up error documents...")
    
    if not os.path.exists(STATUS_FILE):
        print("❌ Status file not found. Nothing to clean.")
        return
    
    try:
        # Read current status
        with open(STATUS_FILE, 'r') as f:
            data = json.load(f)
        
        print(f"📊 Found {len(data)} total documents")
        
        # Count documents by company
        demo_docs = [doc for doc in data if doc.get('company_id_slug') == 'demo-business-123']
        other_docs = [doc for doc in data if doc.get('company_id_slug') != 'demo-business-123']
        
        print(f"📊 Demo business documents: {len(demo_docs)}")
        print(f"📊 Other company documents: {len(other_docs)}")
        
        # Show demo document status
        print("\n🔍 Demo documents status:")
        for doc in demo_docs:
            filename = doc.get('filename', 'Unknown')
            status = doc.get('status', 'Unknown')
            has_error = 'error' in str(doc).lower() or 'readonly' in str(doc).lower()
            print(f"   - {filename}: {status} {'❌ (HAS ERROR)' if has_error else '✅'}")
        
        # Filter out error documents
        print("\n🧹 Removing documents with errors...")
        
        clean_demo_docs = []
        removed_count = 0
        
        for doc in demo_docs:
            # Check if document has errors
            doc_str = str(doc).lower()
            has_error = ('error' in doc_str or 
                        'readonly' in doc_str or 
                        'database error' in doc_str or
                        doc.get('status') != 'completed')
            
            if has_error:
                print(f"   ❌ Removing: {doc.get('filename', 'Unknown')}")
                removed_count += 1
            else:
                clean_demo_docs.append(doc)
                print(f"   ✅ Keeping: {doc.get('filename', 'Unknown')}")
        
        # Combine clean demo docs with other company docs
        cleaned_data = clean_demo_docs + other_docs
        
        # Backup original file
        backup_file = f"{STATUS_FILE}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        with open(backup_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"💾 Backup created: {backup_file}")
        
        # Write cleaned data
        with open(STATUS_FILE, 'w') as f:
            json.dump(cleaned_data, f, indent=2)
        
        print(f"\n✅ Cleanup complete!")
        print(f"   - Removed: {removed_count} error documents")
        print(f"   - Kept: {len(clean_demo_docs)} good demo documents")
        print(f"   - Total remaining: {len(cleaned_data)} documents")
        
        if len(clean_demo_docs) == 0:
            print("\n🔄 No demo documents remaining. They will be recreated on next app startup.")
        
    except Exception as e:
        print(f"❌ Error during cleanup: {e}")

def show_current_status():
    """Show current document status"""
    
    print("📊 CURRENT STATUS")
    print("=" * 50)
    
    if not os.path.exists(STATUS_FILE):
        print("❌ No status file found")
        return
    
    try:
        with open(STATUS_FILE, 'r') as f:
            data = json.load(f)
        
        demo_docs = [doc for doc in data if doc.get('company_id_slug') == 'demo-business-123']
        
        print(f"Total demo documents: {len(demo_docs)}")
        
        for i, doc in enumerate(demo_docs, 1):
            filename = doc.get('filename', 'Unknown')
            status = doc.get('status', 'Unknown')
            has_error = 'error' in str(doc).lower()
            
            print(f"{i}. {filename}")
            print(f"   Status: {status}")
            print(f"   Has Error: {'Yes ❌' if has_error else 'No ✅'}")
            print()
        
    except Exception as e:
        print(f"❌ Error reading status: {e}")

if __name__ == "__main__":
    print("🚀 Document Error Cleanup Tool")
    print("=" * 40)
    
    # Show current status
    show_current_status()
    
    # Ask for confirmation
    response = input("\n🤔 Do you want to remove error documents? (y/n): ").lower()
    
    if response in ['y', 'yes']:
        cleanup_error_documents()
        print("\n🎯 Next steps:")
        print("1. Restart your Flask app: python app.py")
        print("2. Test your demo widget")
        print("3. If no demo docs remain, they'll be auto-created")
    else:
        print("❌ Cleanup cancelled")