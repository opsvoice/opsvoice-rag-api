from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import os

CHROMA_DIR = "/data/chroma_db"  # or your local chroma_db path
embedding = OpenAIEmbeddings()
db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding)

# Print all doc metadata
docs = db.get()
print("TOTAL EMBEDDED CHUNKS:", len(docs['metadatas']))
for i, meta in enumerate(docs['metadatas']):
    print(f"{i}: {meta}")

# Try filtered retrieval by company_id_slug
company_id = "demo-business-123"
results = db.similarity_search("onboarding", k=5, filter={"company_id_slug": company_id})
print("\nFiltered search results for 'onboarding':")
for r in results:
    print("---")
    print("Content:", r.page_content[:120])
    print("Metadata:", r.metadata)
