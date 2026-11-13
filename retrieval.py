
from sentence_transformers import SentenceTransformer
import chromadb

# STEP 1: Load the embedding model
print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

# STEP 2: Connect to ChromaDB
client = chromadb.PersistentClient(path="chroma_db")

try:
    collection = client.get_collection(name="smartdoc")
except Exception:
    print("Database collection not found!")
    print("Run python vector_store.py first.")
    raise SystemExit

# STEP 3: Get question from user
question = input("\nAsk a question about your PDF: ").strip()

if not question:
    print("Please enter a question.")
    raise SystemExit

# STEP 4: Convert question into embedding
question_embedding = model.encode(question).tolist()

# STEP 5: Retrieve relevant chunks
results = collection.query(
    query_embeddings=[question_embedding],
    n_results=min(3, collection.count()),
    include=["documents", "distances"]
)

# STEP 6: Display results
documents = results["documents"][0]
distances = results["distances"][0]

print("\n========== RETRIEVAL RESULTS ==========")

# This is an adjustable distance threshold,
# not a universal accuracy value.
distance_threshold = 1.0

found_results = False

for i, (document, distance) in enumerate(
    zip(documents, distances), start=1
):
    print(f"\n--- Result {i} ---")
    print("Distance:", round(distance, 4))

    if distance <= distance_threshold:
        print("Status: Relevant candidate")
        print("\nContent:")
        print(document)
        found_results = True
    else:
        print("Status: Filtered out")

if not found_results:
    print("\nNo sufficiently similar content found.")
    print("Try asking a question using different words.")

print("\n======================================")
