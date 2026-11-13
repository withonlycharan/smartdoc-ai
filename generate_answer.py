print("SMARTDOC PROGRAM STARTED")

import requests
import chromadb
from sentence_transformers import SentenceTransformer

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_collection(name="smartdoc")

question = input("Ask a question about your PDF: ")

question_embedding = model.encode(question).tolist()

results = collection.query(
    query_embeddings=[question_embedding],
    n_results=min(3, collection.count())
)

context = "\n\n".join(results["documents"][0])

prompt = f"""
Answer using only the PDF context.
If the answer is missing, say so.

PDF context:
{context}

Question:
{question}

Answer:
"""

print("Generating answer...")

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "llama3.2:1b",
        "prompt": prompt,
        "stream": False
    },
    timeout=180
)

response.raise_for_status()

print("\nSMARTDOC ANSWER:")
print(response.json()["response"])