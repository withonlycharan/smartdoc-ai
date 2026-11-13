
from pypdf import PdfReader

# 1. PDF file location
pdf_path = "documents/sample.pdf"

# 2. Open the PDF
reader = PdfReader(pdf_path)

# 3. Find the total number of pages
print("Total Pages:", len(reader.pages))

# 4. Read each page
for page_number, page in enumerate(reader.pages, start=1):

    # Extract text
    text = page.extract_text()

    print("\nPage:", page_number)

    if text:
        print(text)
    else:
        print("No text found on this page.")
