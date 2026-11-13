
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb

# Step 1: Read PDF
reader = PdfReader("documents/sample.pdf")

full_text = ""

for page in reader.pages:
    text = page.extract_text()
    if text:
        full_text += text + "\n"

# Step 2: Create chunks
chunk_size = 500
overlap = 50

chunks = []
start = 0

while start < len(full_text):
    end = min(start + chunk_size, len(full_text))

    if end < len(full_text):
        space = full_text.rfind(" ", start, end)
        if space > start:
            end = space

    chunk = full_text[start:end].strip()

    if chunk:
        chunks.append(chunk)

    if end >= len(full_text):
        break

    next_start = max(start + 1, end - overlap)

    while next_start < len(full_text) and not full_text[next_start - 1].isspace():
        next_start += 1

    start = next_start

# Step 3: Create embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks).tolist()

# Step 4: Create ChromaDB database
client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_or_create_collection(
    name="smartdoc"
)

# Step 5: Store chunks and embeddings
collection.upsert(
    ids=[f"chunk_{i}" for i in range(len(chunks))],
    documents=chunks,
    embeddings=embeddings
)

print("Successfully stored data in ChromaDB!")
print("Total chunks stored:", collection.count())
