"""
Enhanced functions for OpsVoice AI - Windows Compatible
"""

import re
import time
import os
import json
from langchain_community.document_loaders import UnstructuredWordDocumentLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

def extract_simple_keywords(text):
    """Simple keyword extraction"""
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    stopwords = {'that', 'this', 'with', 'they', 'have', 'will', 'from', 'been', 'were'}
    keywords = [word for word in set(words) if word not in stopwords]
    return keywords[:5]

def enhanced_embed_sop_worker(fpath, metadata, CHROMA_DIR, embedding, update_status):
    """Enhanced embedding worker - Windows compatible"""
    fname = os.path.basename(fpath)
    try:
        ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""
        if ext == "docx":
            docs = UnstructuredWordDocumentLoader(fpath).load()
        elif ext == "pdf":
            docs = PyPDFLoader(fpath).load()
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        
        # ENHANCED CHUNKING
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # INCREASED from 500
            chunk_overlap=200,  # INCREASED from 100
            separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]
        ).split_documents(docs)
        
        # Add enhanced metadata
        for i, chunk in enumerate(chunks):
            content_lower = chunk.page_content.lower()
            if any(word in content_lower for word in ["angry", "upset", "difficult", "complaint"]):
                chunk_type = "customer_service"
            elif any(word in content_lower for word in ["cash", "money", "payment", "refund"]):
                chunk_type = "financial"
            elif any(word in content_lower for word in ["first day", "onboard", "training"]):
                chunk_type = "onboarding"
            else:
                chunk_type = "general"
            
            chunk.metadata.update({
                **(metadata or {}),
                "chunk_id": f"{fname}_{i}",
                "chunk_type": chunk_type,
                "keywords": extract_simple_keywords(chunk.page_content)
            })
        
        # Create/update database
        db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding)
        db.add_documents(chunks)
        db.persist()
        
        print(f"[EMBED] Successfully embedded {len(chunks)} chunks from {fname}")
        update_status(fname, {"status": "embedded", "chunk_count": len(chunks), **(metadata or {})})
        
    except Exception as e:
        print(f"[EMBED] Error with {fname}: {e}")
        update_status(fname, {"status": f"error: {str(e)}", **(metadata or {})})

def expand_query_with_synonyms(query):
    """Expand query with synonyms"""
    synonyms = {
        "angry": "angry upset difficult frustrated mad",
        "customer": "customer client guest patron",
        "handle": "handle deal manage respond",
        "procedure": "procedure process protocol steps",
        "policy": "policy rule guideline standard",
        "refund": "refund return money back exchange",
        "cash": "cash money payment transaction",
        "first day": "first day onboarding orientation training new employee"
    }
    
    expanded = query
    for word, expansion in synonyms.items():
        if word in query.lower():
            expanded += f" {expansion}"
    
    return expanded

def create_contextual_prompt(query, docs):
    """Create contextual prompts"""
    query_lower = query.lower()
    
    context_text = ""
    for i, doc in enumerate(docs):
        context_text += f"\n--- Document {i+1} ---\n{doc.page_content}\n"
    
    if any(word in query_lower for word in ["angry", "upset", "difficult", "complaint"]):
        template = f"""You are a customer service expert. Use the company documents below to help handle difficult customer situations.

INSTRUCTIONS:
- Look for specific policies about handling upset customers
- Provide step-by-step guidance
- Be empathetic and professional
- Reference company procedures when available

Company Documents:
{context_text}

Previous conversation: {{chat_history}}

Customer Service Question: {{question}}

Provide helpful, specific guidance based on company policies:"""

    elif any(word in query_lower for word in ["cash", "money", "payment", "refund"]):
        template = f"""You are a financial procedures expert. Use the company documents below to provide accurate financial guidance.

INSTRUCTIONS:
- Look for specific cash handling or refund procedures
- Provide exact steps to follow
- Reference policy numbers or section names when available
- Be precise about authorization requirements

Company Documents:
{context_text}

Previous conversation: {{chat_history}}

Financial Question: {{question}}

Provide specific procedure steps based on company policies:"""

    else:
        template = f"""You are a helpful company assistant. Use the company documents below to provide accurate information.

INSTRUCTIONS:
- Answer based on the provided company documents
- Be specific and reference sources
- Provide step-by-step guidance when possible
- If information is missing, say so clearly

Company Documents:
{context_text}

Previous conversation: {{chat_history}}

Question: {{question}}

Provide a helpful answer based on company documentation:"""

    return template

def is_generic_response(answer):
    """Check if response is too generic"""
    if not answer:
        return True
    generic_phrases = [
        "i don't have", "i'm not sure", "sorry, i don't", 
        "i cannot", "i'm unable", "no information available"
    ]
    return any(phrase in answer.lower() for phrase in generic_phrases)

def create_enhanced_fallback(query, tenant):
    """Create better fallback responses"""
    query_lower = query.lower()
    
    if any(word in query_lower for word in ["angry", "upset", "customer"]):
        return f"""I don't see specific customer service policies in your uploaded documents for {tenant.replace('-', ' ').title()}. 

For handling difficult customers, here are general best practices:
1. Listen actively and remain calm
2. Acknowledge their concerns
3. Apologize for any inconvenience  
4. Focus on finding a solution
5. Escalate to a manager if needed

Please check your employee handbook or contact your supervisor for company-specific policies."""

    elif any(word in query_lower for word in ["cash", "money", "refund"]):
        return f"""I don't see specific financial procedures in your uploaded documents for {tenant.replace('-', ' ').title()}.

For financial processes, please:
1. Refer to your company's financial procedures manual
2. Check with your manager for authorization limits
3. Contact the accounting department for guidance

Try asking about other topics from your uploaded company documents."""

    else:
        return f"""I don't see information about that in your company documents for {tenant.replace('-', ' ').title()}.

This might be because:
- The document hasn't been uploaded yet
- It's covered under a different topic
- It requires manager approval

Try asking about procedures, policies, or customer service topics that might be in your uploaded documents."""

def generate_smart_followups(query, answer):
    """Generate contextual follow-up questions"""
    query_lower = query.lower()
    answer_lower = answer.lower()
    
    if "step" in answer_lower:
        return [
            "What if something goes wrong in this process?",
            "Who should I contact for help?",
            "Are there any exceptions to these steps?"
        ]
    elif "customer" in query_lower:
        return [
            "What if the customer is still not satisfied?",
            "Should I escalate to a manager?",
            "How do I document this interaction?"
        ]
    else:
        return [
            "Can you provide more details?",
            "What's the next step?",
            "Who can help me with this?"
        ]

class DummyRetriever:
    """Simple retriever for selected documents"""
    def __init__(self, docs):
        self.docs = docs
    
    def get_relevant_documents(self, query):
        return self.docs

def enhanced_query_processing(qtext, tenant, vectorstore, clean_text, check_rate_limit, is_vague, COMPANY_VOICES, generate_followups):
    """Enhanced query processing function"""
    from flask import jsonify
    
    # Rate limiting
    if not check_rate_limit(tenant):
        return jsonify({"error": "Too many requests, try again in a minute."}), 429

    # Check for vague queries
    if is_vague(qtext):
        return jsonify({
            "answer": "Can you give me more detail—like the specific SOP or process you're referring to?",
            "source": "clarify"
        })

    # Filter out off-topic queries
    off_topic_keywords = ["gmail", "facebook", "amazon account", "weather", "news", "stock price"]
    if any(keyword in qtext.lower() for keyword in off_topic_keywords):
        return jsonify({
            "answer": "That's outside your company SOPs—please use the official help portal or ask about your internal procedures.",
            "source": "off_topic"
        })

    try:
        print(f"[QUERY] Processing: {qtext} for company: {tenant}")
        
        # Enhanced retrieval
        retriever = vectorstore.as_retriever(
            search_kwargs={
                "k": 8,  # INCREASED from 5
                "filter": {"company_id_slug": tenant}
            }
        )
        
        # Expand query with synonyms
        expanded_query = expand_query_with_synonyms(qtext)
        print(f"[QUERY] Expanded query: {expanded_query}")
        
        # Get documents
        relevant_docs = retriever.get_relevant_documents(expanded_query)
        
        if not relevant_docs:
            print(f"[QUERY] No docs with company filter, trying without filter")
            general_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
            relevant_docs = general_retriever.get_relevant_documents(expanded_query)
        
        print(f"[QUERY] Found {len(relevant_docs)} relevant documents")
        
        if relevant_docs:
            # Create enhanced prompt
            enhanced_prompt = create_contextual_prompt(qtext, relevant_docs)
            
            # Create memory
            memory = ConversationBufferMemory(
                memory_key="chat_history", 
                return_messages=True,
                output_key="answer"
            )
            
            # Custom prompt template
            prompt_template = PromptTemplate(
                input_variables=["context", "question", "chat_history"],
                template=enhanced_prompt
            )
            
            qa = ConversationalRetrievalChain.from_llm(
                ChatOpenAI(temperature=0, model="gpt-4"),
                retriever=DummyRetriever(relevant_docs),
                memory=memory,
                return_source_documents=True,
                combine_docs_chain_kwargs={"prompt": prompt_template}
            )
            
            # Get answer
            result = qa.invoke({"question": expanded_query})
            answer = clean_text(result.get("answer", ""))

            # Validate answer
            if answer and len(answer.split()) > 10 and not is_generic_response(answer):
                print(f"[QUERY] Success - found relevant answer")
                
                # Truncate for TTS
                if len(answer.split()) > 80:
                    answer = "Here's a summary: " + " ".join(answer.split()[:70]) + "... For complete details, check your SOPs."

                company_voice = COMPANY_VOICES.get(tenant, "")
                
                return jsonify({
                    "answer": f"{company_voice} {answer}",
                    "fallback_used": False,
                    "followups": generate_smart_followups(qtext, answer),
                    "source": "sop",
                    "source_documents": len(relevant_docs)
                })

        # Enhanced fallback
        print(f"[QUERY] Using enhanced fallback")
        fallback_answer = create_enhanced_fallback(qtext, tenant)
        
        return jsonify({
            "answer": fallback_answer,
            "fallback_used": True,
            "followups": ["Can you be more specific?", "What department handles this?", "Try asking about a specific procedure"],
            "source": "fallback"
        })

    except Exception as e:
        print(f"[QUERY] Error: {e}")
        return jsonify({"error": "Query failed", "details": str(e)}), 500