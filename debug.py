from src.vectorstore.store import VectorStore

print("\n--- 1. LOADING DATABASE ---")
try:
    store = VectorStore().load_existing()
    chroma_db = store._store
    print("✅ Database loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load database: {e}")
    exit()

print("\n--- 2. CHECKING CHUNKS ---")
# This pulls all data directly out of Chroma
collection_data = chroma_db.get()
doc_count = len(collection_data['documents'])

print(f"Total chunks stored in database: {doc_count}")

if doc_count == 0:
    print("❌ CRITICAL ERROR: The database is empty! Ingestion is failing to save.")
    exit()
else:
    print("\n✅ Database is populated. Here is a sample of Chunk #1:")
    print("-" * 40)
    print(collection_data['documents'][0][:200] + "...")
    print("-" * 40)

print("\n--- 3. TESTING RAW SEARCH ---")
question = "how many paid leaves i will get for bereavment ?"
print(f"Searching for: '{question}'")

# We bypass the LLM and the retriever, asking the database directly
results = chroma_db.similarity_search(question, k=3)

if not results:
    print("❌ CRITICAL ERROR: The search returned 0 documents.")
else:
    print("\n✅ SUCCESS: The database found matches! Here is the top result:")
    print(f"File: {results[0].metadata.get('source', 'Unknown')}")
    print("-" * 40)
    print(results[0].page_content)
    print("-" * 40)

print("\nDEBUG COMPLETE.")