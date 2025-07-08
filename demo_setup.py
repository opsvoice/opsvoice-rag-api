#!/usr/bin/env python3
"""
Demo SOP Upload Script for OpsVoice AI
This script uploads the 5 demo SOPs to create a comprehensive business demo
Updated to work with your VS Code file structure
"""

import requests
import json
import time
from pathlib import Path
import os

# Configuration
API_BASE_URL = "https://opsvoice-rag-api.onrender.com"
DEMO_COMPANY_SLUG = "demo-business-123"

# Demo SOPs to upload - Updated paths for your VS Code structure
DEMO_SOPS = [
    {
        "filename": "demo_sops/customer_service_procedures.pdf",
        "title": "Customer Service Procedures Manual",
        "description": "Comprehensive customer service standards, complaint resolution, refunds, and quality assurance"
    },
    {
        "filename": "demo_sops/employee_procedures_manual.pdf", 
        "title": "Employee Procedures & HR Policies",
        "description": "Time-off requests, attendance policies, workplace safety, and professional conduct"
    },
    {
        "filename": "demo_sops/daily_operations_procedures.pdf",
        "title": "Daily Operations & Cash Handling",
        "description": "Opening/closing procedures, equipment management, cash handling, and troubleshooting"
    },
    {
        "filename": "demo_sops/emergency_procedures_manual.pdf",
        "title": "Emergency Procedures & Safety Manual", 
        "description": "Medical emergencies, security incidents, natural disasters, and workplace safety"
    },
    {
        "filename": "demo_sops/onboarding_training_manual.pdf",
        "title": "New Employee Onboarding & Training",
        "description": "90-day training program, performance expectations, and career development"
    }
]

def clear_existing_data():
    """Clear existing vectorstore and start fresh"""
    print("🧹 Clearing existing data...")
    try:
        # Force clean the database
        response = requests.post(f"{API_BASE_URL}/force-clean-db", timeout=30)
        if response.status_code == 200:
            print("✅ Database cleared successfully")
            return True
        else:
            print(f"⚠️  Database clear returned: {response.status_code}")
            return True  # Continue anyway
    except Exception as e:
        print(f"⚠️  Error clearing database: {e}")
        return True  # Continue anyway

def upload_sop(filename, title, description):
    """Upload a single SOP document to the API"""
    
    # Check if file exists
    if not Path(filename).exists():
        print(f"❌ File not found: {filename}")
        return False
    
    try:
        print(f"📤 Uploading: {title}")
        
        # Prepare the upload
        with open(filename, 'rb') as file:
            files = {'file': file}
            data = {
                'company_id_slug': DEMO_COMPANY_SLUG,
                'doc_title': title
            }
            
            # Make the upload request
            response = requests.post(
                f"{API_BASE_URL}/upload-sop",
                files=files,
                data=data,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Successfully uploaded: {title}")
                print(f"   📄 File URL: {result.get('sop_file_url')}")
                return True
            else:
                print(f"❌ Upload failed for {title}: {response.status_code}")
                print(f"   Error: {response.text}")
                return False
                
    except Exception as e:
        print(f"❌ Error uploading {title}: {str(e)}")
        return False

def check_api_health():
    """Check if the API is responsive"""
    try:
        response = requests.get(f"{API_BASE_URL}/healthz", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ API is healthy")
            print(f"   📊 Vectorstore: {health_data.get('vectorstore')}")
            print(f"   📚 Total documents: {health_data.get('total_documents', 0)}")
            return True
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API not responding: {str(e)}")
        return False

def check_debug_docs():
    """Check what's actually in the vectorstore"""
    try:
        response = requests.get(f"{API_BASE_URL}/debug-docs", timeout=10)
        if response.status_code == 200:
            debug_data = response.json()
            print(f"🔍 Debug Info:")
            print(f"   📊 Total documents in vectorstore: {debug_data.get('total_docs', 0)}")
            print(f"   🏢 Demo company documents: {debug_data.get('demo_company_docs', 0)}")
            
            if debug_data.get('sample_content'):
                print(f"   📄 Sample content preview:")
                for i, content in enumerate(debug_data['sample_content'][:2]):
                    print(f"      {i+1}. {content[:100]}...")
            
            return debug_data
        else:
            print(f"⚠️  Debug endpoint returned: {response.status_code}")
            return {}
    except Exception as e:
        print(f"❌ Error checking debug info: {str(e)}")
        return {}

def check_company_docs():
    """Check what documents are uploaded for the demo company"""
    try:
        response = requests.get(f"{API_BASE_URL}/company-docs/{DEMO_COMPANY_SLUG}")
        if response.status_code == 200:
            docs = response.json()
            print(f"📚 Found {len(docs)} documents for {DEMO_COMPANY_SLUG}")
            for doc in docs:
                status_icon = "✅" if doc.get('status') == 'embedded' else "⏳" if 'embedding' in str(doc.get('status')) else "❌"
                print(f"   {status_icon} {doc.get('title')} - Status: {doc.get('status')}")
            return docs
        else:
            print(f"📚 No existing documents found for {DEMO_COMPANY_SLUG}")
            return []
    except Exception as e:
        print(f"❌ Error checking existing docs: {str(e)}")
        return []

def wait_for_embedding_completion():
    """Wait for all documents to finish embedding"""
    print("\n⏳ Waiting for document embedding to complete...")
    
    max_wait = 120  # 2 minutes max
    wait_time = 0
    
    while wait_time < max_wait:
        docs = check_company_docs()
        
        if not docs:
            print("   No documents found yet, continuing to wait...")
        else:
            # Check if all are embedded
            embedded_count = sum(1 for doc in docs if doc.get('status') == 'embedded')
            total_count = len(docs)
            
            print(f"   📊 Embedding progress: {embedded_count}/{total_count} complete")
            
            if embedded_count == total_count:
                print("✅ All documents embedded successfully!")
                return True
            
            # Check for errors
            error_docs = [doc for doc in docs if 'error' in str(doc.get('status', '')).lower()]
            if error_docs:
                print(f"❌ Found {len(error_docs)} documents with errors:")
                for doc in error_docs:
                    print(f"   ❌ {doc.get('title')}: {doc.get('status')}")
        
        time.sleep(10)
        wait_time += 10
        print(f"   ⏱️  Waited {wait_time}s...")
    
    print("⚠️  Timeout waiting for embedding completion")
    return False

def test_demo_queries():
    """Test the demo with sample queries"""
    
    test_queries = [
        {
            "query": "What documents do I have uploaded?",
            "expected": "document_list"
        },
        {
            "query": "What are the cash management procedures?", 
            "expected": "sop"
        },
        {
            "query": "How do I request time off?",
            "expected": "sop"
        },
        {
            "query": "What's the procedure for opening the store?",
            "expected": "sop"  
        },
        {
            "query": "How do I handle a customer refund?",
            "expected": "sop"
        },
        {
            "query": "What happens during employee onboarding?",
            "expected": "sop"
        }
    ]
    
    print("\n🧪 Testing demo queries...")
    successful_queries = 0
    
    for test in test_queries:
        try:
            print(f"\n❓ Query: {test['query']}")
            
            response = requests.post(
                f"{API_BASE_URL}/query",
                json={
                    "query": test['query'],
                    "company_id_slug": DEMO_COMPANY_SLUG
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get('answer', '')
                source = result.get('source', '')
                
                print(f"✅ Response ({source}): {answer[:150]}...")
                
                # Check response quality
                if source == test['expected'] or (source == 'document_list' and 'documents' in test['query'].lower()):
                    print("   🎯 Perfect response!")
                    successful_queries += 1
                elif source in ['sop', 'document_list'] and len(answer) > 50:
                    print("   ✨ Good response!")
                    successful_queries += 1
                elif source == 'fallback':
                    print("   ⚠️  Fallback response - check SOP content")
                else:
                    print("   ❓ Unexpected response type")
                    
            else:
                print(f"❌ Query failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Query error: {str(e)}")
        
        time.sleep(2)  # Rate limiting
    
    print(f"\n📊 Query Test Results: {successful_queries}/{len(test_queries)} successful")
    return successful_queries >= len(test_queries) * 0.7  # 70% success rate

def main():
    """Main upload and setup process"""
    
    print("🚀 OpsVoice AI Demo Setup - Fresh Install")
    print("=" * 60)
    
    # Step 0: Check working directory
    print(f"📁 Working directory: {os.getcwd()}")
    print(f"📁 Looking for demo_sops folder...")
    
    # Step 1: Check API health
    print("\n1️⃣ Checking API health...")
    if not check_api_health():
        print("❌ API is not responding. Please check the service status.")
        return
    
    # Step 2: Clear existing data for fresh start
    print("\n2️⃣ Clearing existing data...")
    clear_existing_data()
    time.sleep(5)  # Give it time to clear
    
    # Step 3: Check current state
    print("\n3️⃣ Checking current vectorstore state...")
    debug_info = check_debug_docs()
    
    # Step 4: Upload SOPs
    print(f"\n4️⃣ Uploading {len(DEMO_SOPS)} demo SOPs...")
    successful_uploads = 0
    
    for sop in DEMO_SOPS:
        if upload_sop(sop['filename'], sop['title'], sop['description']):
            successful_uploads += 1
        time.sleep(3)  # Wait between uploads for processing
    
    print(f"\n📊 Upload Summary: {successful_uploads}/{len(DEMO_SOPS)} successful")
    
    # Step 5: Wait for processing
    if successful_uploads > 0:
        # Wait for embedding completion
        embedding_success = wait_for_embedding_completion()
        
        # Step 6: Final verification
        print("\n5️⃣ Final verification...")
        final_debug = check_debug_docs()
        final_docs = check_company_docs()
        
        # Step 7: Test the demo
        if embedding_success and final_debug.get('demo_company_docs', 0) > 0:
            test_success = test_demo_queries()
            
            if test_success:
                print("\n🎉 Demo setup completed successfully!")
                print(f"🎯 Demo company slug: {DEMO_COMPANY_SLUG}")
                print(f"📚 Documents embedded: {final_debug.get('demo_company_docs', 0)}")
                print(f"🌐 Test at: https://demo.opsvoice.ai/try")
            else:
                print("\n⚠️  Demo setup complete but some queries failed")
        else:
            print("\n❌ Demo setup completed but embedding verification failed")
    else:
        print("\n❌ No documents were uploaded successfully")

if __name__ == "__main__":
    main()