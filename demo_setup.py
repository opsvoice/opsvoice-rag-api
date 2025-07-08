#!/usr/bin/env python3
"""
Demo SOP Upload Script for OpsVoice AI
This script uploads the 5 demo SOPs to create a comprehensive business demo
"""

import requests
import json
import time
from pathlib import Path

# Configuration
API_BASE_URL = "https://opsvoice-rag-api.onrender.com"
DEMO_COMPANY_SLUG = "demo-business-123"

# Demo SOPs to upload
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
            print(f"   💾 Cache size: {health_data.get('cache_size')}")
            return True
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API not responding: {str(e)}")
        return False

def check_company_docs():
    """Check what documents are already uploaded for the demo company"""
    try:
        response = requests.get(f"{API_BASE_URL}/company-docs/{DEMO_COMPANY_SLUG}")
        if response.status_code == 200:
            docs = response.json()
            print(f"📚 Found {len(docs)} existing documents for {DEMO_COMPANY_SLUG}")
            for doc in docs:
                print(f"   📄 {doc.get('title')} - Status: {doc.get('status')}")
            return docs
        else:
            print(f"📚 No existing documents found for {DEMO_COMPANY_SLUG}")
            return []
    except Exception as e:
        print(f"❌ Error checking existing docs: {str(e)}")
        return []

def test_demo_queries():
    """Test the demo with sample queries"""
    
    test_queries = [
        "What documents do I have uploaded that I can ask questions about?",
        "What do I do if a customer wants a refund?",
        "How do I request time off?", 
        "What's the procedure for opening the store?",
        "What happens on my first day of work?",
        "How do I handle a workplace injury?"
    ]
    
    print("\n🧪 Testing demo queries...")
    
    for query in test_queries:
        try:
            print(f"\n❓ Query: {query}")
            
            response = requests.post(
                f"{API_BASE_URL}/query",
                json={
                    "query": query,
                    "company_id_slug": DEMO_COMPANY_SLUG
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get('answer', '')
                source = result.get('source', '')
                
                print(f"✅ Response ({source}): {answer[:100]}...")
                
                # Check for good responses
                if len(answer) > 50 and source in ['sop', 'document_list']:
                    print("   ✨ Great response!")
                elif source == 'business_fallback':
                    print("   ⚠️  Fallback response - may need SOP improvement")
                else:
                    print("   ❓ Vague response - check query clarity")
                    
            else:
                print(f"❌ Query failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Query error: {str(e)}")
        
        time.sleep(2)  # Rate limiting

def main():
    """Main upload and setup process"""
    
    print("🚀 OpsVoice AI Demo Setup")
    print("=" * 50)
    
    # Step 1: Check API health
    print("\n1️⃣ Checking API health...")
    if not check_api_health():
        print("❌ API is not responding. Please check the service status.")
        return
    
    # Step 2: Check existing documents
    print("\n2️⃣ Checking existing documents...")
    existing_docs = check_company_docs()
    
    # Step 3: Upload SOPs
    print(f"\n3️⃣ Uploading {len(DEMO_SOPS)} demo SOPs...")
    successful_uploads = 0
    
    for sop in DEMO_SOPS:
        if upload_sop(sop['filename'], sop['title'], sop['description']):
            successful_uploads += 1
        time.sleep(3)  # Wait between uploads for processing
    
    print(f"\n📊 Upload Summary: {successful_uploads}/{len(DEMO_SOPS)} successful")
    
    # Step 4: Wait for processing
    if successful_uploads > 0:
        print("\n4️⃣ Waiting for document processing...")
        print("   ⏳ Documents are being embedded in the background...")
        time.sleep(30)  # Give time for embedding
        
        # Check final status
        print("\n5️⃣ Final document status...")
        check_company_docs()
        
        # Step 5: Test the demo
        test_demo_queries()
    
    print("\n✅ Demo setup complete!")
    print(f"🎯 Demo company slug: {DEMO_COMPANY_SLUG}")
    print(f"🌐 Test at: https://demo.opsvoice.ai/try")

if __name__ == "__main__":
    main()