
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# Step 1: Read PDF
reader = PdfReader("documents/sample.pdf")

full_text = ""

for page in reader.pages:
    text = page.extract_text()
    if text:
        full_text += text + "\n"

# Step 2: Split text into chunks
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

# Step 3: Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 4: Convert chunks into embeddings
embeddings = model.encode(chunks)

# Step 5: Display results
print("Total Chunks:", len(chunks))
print("Embedding Shape:", embeddings.shape)

for i, embedding in enumerate(embeddings):
    print(f"\nChunk {i + 1}")
    print("First 5 values:", embedding[:5])
