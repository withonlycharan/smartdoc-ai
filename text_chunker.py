
from pypdf import PdfReader

# Step 1: Read PDF
reader = PdfReader("documents/sample.pdf")

# Step 2: Extract all text
full_text = ""

for page in reader.pages:
    text = page.extract_text()

    if text:
        full_text += text + "\n"

# Step 3: Chunk settings
chunk_size = 500
overlap = 50

# Step 4: Create chunks without splitting words
chunks = []
start = 0

while start < len(full_text):

    end = min(start + chunk_size, len(full_text))

    # Find the nearest word boundary
    if end < len(full_text):
        space = full_text.rfind(" ", start, end)

        if space > start:
            end = space

    chunk = full_text[start:end].strip()

    if chunk:
        chunks.append(chunk)

    if end >= len(full_text):
        break

    # Calculate next starting position
    next_start = max(start + 1, end - overlap)

    # Move to the next word boundary
    while next_start < len(full_text) and not full_text[next_start - 1].isspace():
        next_start += 1

    start = next_start

# Step 5: Display results
print("Total Chunks:", len(chunks))

for i, chunk in enumerate(chunks, start=1):
    print(f"\n--- Chunk {i} ---")
    print(chunk)
    print("Characters:", len(chunk))
