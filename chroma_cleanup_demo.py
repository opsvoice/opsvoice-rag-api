from chromadb import Client
from chromadb.config import Settings

CHROMA_DIR = "chroma_db"  # Make sure this matches your app.py and exists!

client = Client(settings=Settings(persist_directory=CHROMA_DIR))

try:
    # List all collections
    collections = client.list_collections()
    print("Collections found:", [col.name for col in collections])

    # Try to get the demo collection
    try:
        collection = client.get_collection("demo-business-123")
        # List docs before deleting
        docs = collection.get()
        print(f"Docs before deleting: {len(docs['ids'])}")
        if len(docs['ids']) > 0:
            print("Sample doc IDs:", docs['ids'][:3])
        # Delete all docs
        collection.delete(where={})
        print("Deleted all docs for 'demo-business-123'.")
    except Exception as e:
        print("Collection 'demo-business-123' not found or already empty.", str(e))

except Exception as e:
    print("Error connecting to ChromaDB:", str(e))
