#!/usr/bin/env python3
"""
Test script to verify enhanced query processing
"""

import requests
import json
import time

def test_query(query_text, company_slug="demo-business-123"):
    """Test a single query"""
    print(f"\n🔍 Testing: '{query_text}'")
    
    try:
        response = requests.post(
            "https://opsvoice-rag-api.onrender.com/query",
            json={
                "query": query_text,
                "company_id_slug": company_slug
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
            
            print(f"📊 Source: {source}")
            print(f"📚 Documents: {source_docs}")
            print(f"🔄 Fallback: {fallback_used}")
            print(f"💬 Answer: {answer[:200]}...")
            
            if source == 'sop' and source_docs > 0 and not fallback_used:
                print("✅ SUCCESS: Found company documents!")
                return True
            elif source == 'fallback':
                print("⚠️ FALLBACK: No relevant documents found")
                return False
            else:
                print(f"⚠️ PARTIAL: Got response but may need improvement")
                return False
                
        else:
            print(f"❌ Request failed: {response.status_code}")
            print(f"Error: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Test the key failing queries"""
    print("🧪 Testing OpsVoice AI Enhanced Query Processing")
    print("=" * 60)
    
    # Test the exact queries that were failing
    test_queries = [
        "How do I handle an angry customer?",  # This was failing before
        "What do I do when a customer is upset?",
        "What are the cash management procedures?",
        "What happens on my first day of work?",
        "How do I process a refund?"
    ]
    
    passed = 0
    total = len(test_queries)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n[{i}/{total}] " + "="*40)
        if test_query(query):
            passed += 1
        time.sleep(2)  # Prevent rate limiting
    
    print("\n" + "="*60)
    print(f"📊 FINAL RESULTS: {passed}/{total} queries successful")
    
    if passed >= 4:
        print("🎉 EXCELLENT! Your enhanced processing is working!")
        print("✅ Ready to test with your n8n workflow!")
    elif passed >= 2:
        print("⚠️ PARTIAL SUCCESS: Some improvement, may need fine-tuning")
    else:
        print("❌ STILL ISSUES: Need to check document embedding")
        print("💡 Try re-uploading documents through your demo form")

if __name__ == "__main__":
    main()