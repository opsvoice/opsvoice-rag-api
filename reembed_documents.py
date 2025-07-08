#!/usr/bin/env python3
"""
Re-embed existing documents with enhanced chunking
Run this from VS Code terminal in your project directory
"""

import requests
import os
import time

def reembed_demo_documents():
    """Re-upload demo documents to trigger enhanced embedding"""
    
    print("🔄 Re-embedding Demo Documents with Enhanced Chunking")
    print("=" * 60)
    
    # Your demo documents folder
    demo_folder = "demo_sops"
    
    if not os.path.exists(demo_folder):
        print(f"❌ Demo folder '{demo_folder}' not found!")
        print("Make sure you're running this from your project root directory")
        return False
    
    # List of demo documents
    demo_files = [
        "customer_service_procedures.pdf",
        "daily_operations_procedures.pdf", 
        "emergency_procedures_manual.pdf",
        "employee_procedures_manual.pdf",
        "onboarding_training_manual.pdf"
    ]
    
    # API endpoint
    upload_url = "https://opsvoice-rag-api.onrender.com/upload-sop"
    
    uploaded_count = 0
    
    for filename in demo_files:
        file_path = os.path.join(demo_folder, filename)
        
        if not os.path.exists(file_path):
            print(f"⚠️ Skipping {filename} - file not found")
            continue
        
        print(f"\n📄 Re-uploading: {filename}")
        
        try:
            # Prepare file for upload
            with open(file_path, 'rb') as file:
                files = {'file': (filename, file, 'application/pdf')}
                data = {
                    'company_id_slug': 'demo-business-123',
                    'doc_title': filename.replace('.pdf', '').replace('_', ' ').title()
                }
                
                # Upload file
                response = requests.post(upload_url, files=files, data=data, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"   ✅ Upload successful: {result.get('message', 'OK')}")
                    uploaded_count += 1
                else:
                    print(f"   ❌ Upload failed: {response.status_code}")
                    print(f"      Error: {response.text[:200]}")
        
        except Exception as e:
            print(f"   ❌ Error uploading {filename}: {e}")
        
        # Small delay to prevent rate limiting
        time.sleep(2)
    
    print(f"\n📊 Upload Summary: {uploaded_count}/{len(demo_files)} files uploaded")
    
    if uploaded_count > 0:
        print("\n⏳ Waiting for embedding to complete...")
        print("This may take 30-60 seconds...")
        
        # Wait for embedding to complete
        time.sleep(30)
        
        # Check embedding status
        print("\n🔍 Checking embedding status...")
        check_embedding_status()
        
        return True
    else:
        print("\n❌ No files were uploaded successfully")
        return False

def check_embedding_status():
    """Check the status of document embedding"""
    
    try:
        status_url = "https://opsvoice-rag-api.onrender.com/sop-status"
        response = requests.get(status_url, timeout=10)
        
        if response.status_code == 200:
            status_data = response.json()
            
            if not status_data:
                print("⚠️ No documents found in status")
                return
            
            print(f"📁 Document Status:")
            
            embedded_count = 0
            for filename, metadata in status_data.items():
                status = metadata.get('status', 'unknown')
                company = metadata.get('company_id_slug', 'unknown')
                chunks = metadata.get('chunk_count', '?')
                
                if status == 'embedded':
                    print(f"   ✅ {filename} - {chunks} chunks")
                    embedded_count += 1
                elif 'embedding' in status:
                    print(f"   ⏳ {filename} - {status}")
                else:
                    print(f"   ❌ {filename} - {status}")
            
            if embedded_count > 0:
                print(f"\n🎉 {embedded_count} documents ready with enhanced chunking!")
                print("✅ Ready to test enhanced queries!")
            else:
                print("\n⚠️ Documents still processing. Wait a bit longer and check again.")
                
        else:
            print(f"❌ Could not check status: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error checking status: {e}")

def test_enhanced_query():
    """Test if the enhanced query processing is working"""
    
    print("\n🧪 Testing Enhanced Query Processing...")
    
    test_query = "How do I handle an angry customer?"
    
    try:
        response = requests.post(
            "https://opsvoice-rag-api.onrender.com/query",
            json={
                "query": test_query,
                "company_id_slug": "demo-business-123"
            },
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            
            source = data.get('source', 'unknown')
            fallback_used = data.get('fallback_used', False)
            source_docs = data.get('source_documents', 0)
            answer = data.get('answer', 'No answer')
            
            print(f"\n🔍 Test Query: '{test_query}'")
            print(f"📊 Source: {source}")
            print(f"📚 Documents Found: {source_docs}")
            print(f"🔄 Fallback Used: {fallback_used}")
            print(f"💬 Answer: {answer[:200]}...")
            
            if source == 'sop' and source_docs > 0 and not fallback_used:
                print("\n🎉 SUCCESS! Enhanced processing is working!")
                print("✅ Found relevant company documents")
                return True
            else:
                print("\n⚠️ Still using fallback - may need more time for embedding")
                return False
                
        else:
            print(f"❌ Test query failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

def main():
    """Main execution function"""
    
    print("🚀 OpsVoice AI Document Re-embedding Tool")
    print("This will re-upload your demo documents with enhanced chunking")
    print()
    
    # Check if we're in the right directory
    if not os.path.exists('app.py'):
        print("❌ app.py not found!")
        print("Please run this script from your project root directory")
        print("(The same directory that contains app.py)")
        return
    
    # Re-embed documents
    success = reembed_demo_documents()
    
    if success:
        print("\n" + "="*60)
        
        # Test the enhanced processing
        test_success = test_enhanced_query()
        
        if test_success:
            print("\n✅ COMPLETE SUCCESS!")
            print("Your enhanced query processing is working!")
            print("\n🎯 Next Steps:")
            print("1. Run your full test script: python test_fixes.py")
            print("2. Test with your n8n workflow")
            print("3. Try voice queries end-to-end")
        else:
            print("\n⚠️ PARTIAL SUCCESS")
            print("Documents uploaded but may still be processing")
            print("\n🔧 Wait 1-2 minutes and run:")
            print("python test_fixes.py")
    else:
        print("\n❌ FAILED")
        print("Could not re-upload documents")
        print("Check your internet connection and try again")

if __name__ == "__main__":
    main()