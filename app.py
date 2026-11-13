
import streamlit as st
import chromadb
import requests
import re

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="SmartDoc AI",
    page_icon="🤖",
    layout="wide"
)


# ==========================================
# 2. LOAD AI MODEL AND DATABASE
# ==========================================

MODEL_NAME = "llama3.2:1b"
OLLAMA_URL = "http://localhost:11434/api/generate"


@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def load_database():
    return chromadb.PersistentClient(path="chroma_db")


model = load_model()
client = load_database()


# ==========================================
# 3. SESSION STATE
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = {
        "PDF Assistant": [],
        "General AI Chat": []
    }

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None


# ==========================================
# 4. TEXT CLEANING
# ==========================================

def clean_text(text):

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ==========================================
# 5. TEXT CHUNKING
# ==========================================

def create_chunks(text, max_words=120, overlap_words=25):

    text = clean_text(text)

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    chunks = []
    current_words = []

    for sentence in sentences:

        words = sentence.split()

        if not words:
            continue

        # Split unusually long sentences safely
        if len(words) > max_words:

            if current_words:
                chunks.append(" ".join(current_words))
                current_words = []

            for start in range(0, len(words), max_words - overlap_words):

                part = words[start:start + max_words]

                if part:
                    chunks.append(" ".join(part))

            continue

        if len(current_words) + len(words) > max_words:

            chunks.append(" ".join(current_words))

            current_words = current_words[-overlap_words:]

        current_words.extend(words)

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


# ==========================================
# 6. OLLAMA AI FUNCTION
# ==========================================

def ask_ollama(prompt):

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.4
            }
        },
        timeout=180
    )

    response.raise_for_status()

    return response.json()["response"]


# ==========================================
# 7. SIDEBAR
# ==========================================

with st.sidebar:

    st.title("🤖 SmartDoc AI")

    st.caption("Your Personal AI Assistant")

    st.divider()

    chat_mode = st.radio(
        "Choose AI Mode",
        [
            "PDF Assistant",
            "General AI Chat"
        ]
    )

    st.divider()

    # PDF upload section
    if chat_mode == "PDF Assistant":

        st.header("📄 Upload PDF")

        uploaded_file = st.file_uploader(
            "Choose your PDF",
            type=["pdf"]
        )

        if st.button(
            "Process PDF",
            use_container_width=True
        ):

            if uploaded_file is None:

                st.warning("Please upload a PDF first.")

            else:

                try:

                    with st.spinner("Reading your PDF..."):

                        reader = PdfReader(uploaded_file)

                        full_text = ""

                        for page in reader.pages:

                            page_text = page.extract_text()

                            if page_text:
                                full_text += page_text + "\n"

                    if not full_text.strip():

                        st.error(
                            "No readable text found. "
                            "This PDF may contain scanned pages."
                        )

                    else:

                        chunks = create_chunks(full_text)

                        if not chunks:

                            st.error("Could not create text chunks.")

                        else:

                            with st.spinner(
                                "Creating AI embeddings..."
                            ):

                                embeddings = model.encode(
                                    chunks,
                                    normalize_embeddings=True
                                ).tolist()

                            # Remove previous PDF collection
                            try:

                                client.delete_collection(
                                    name="smartdoc"
                                )

                            except Exception:

                                pass

                            # Create new cosine similarity collection
                            collection = client.create_collection(
                                name="smartdoc",
                                metadata={
                                    "hnsw:space": "cosine"
                                }
                            )

                            # Store PDF information
                            collection.add(
                                ids=[
                                    f"chunk_{i}"
                                    for i in range(len(chunks))
                                ],
                                documents=chunks,
                                embeddings=embeddings
                            )

                            st.session_state.pdf_name = uploaded_file.name

                            st.session_state.messages[
                                "PDF Assistant"
                            ] = []

                            st.success("PDF processed successfully!")

                            st.write(
                                "Pages:",
                                len(reader.pages)
                            )

                            st.write(
                                "Chunks:",
                                len(chunks)
                            )

                except Exception as error:

                    st.error(
                        f"PDF processing error: {error}"
                    )

        if st.session_state.pdf_name:

            st.success(
                f"Loaded: {st.session_state.pdf_name}"
            )

    else:

        st.header("💬 AI Chat")

        st.write(
            "Ask SmartDoc anything. "
            "You don't need to upload a PDF."
        )

    st.divider()

    if st.button(
        "🗑️ Clear Current Chat",
        use_container_width=True
    ):

        st.session_state.messages[chat_mode] = []

        st.rerun()


# ==========================================
# 8. MAIN WEBSITE
# ==========================================

st.title("🤖 SmartDoc AI")

if chat_mode == "PDF Assistant":

    st.subheader("📚 Chat with your PDF")

    st.write(
        "Ask questions about your uploaded document."
    )

else:

    st.subheader("✨ General AI Assistant")

    st.write(
        "Ask questions, learn new things, "
        "write code and have conversations."
    )


st.divider()


# ==========================================
# 9. DISPLAY CHAT HISTORY
# ==========================================

current_messages = st.session_state.messages[chat_mode]

for message in current_messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ==========================================
# 10. QUESTION INPUT
# ==========================================

if chat_mode == "PDF Assistant":

    placeholder = "Ask something about your PDF..."

else:

    placeholder = "Ask SmartDoc anything..."


question = st.chat_input(placeholder)


# ==========================================
# 11. PROCESS USER QUESTION
# ==========================================

if question:

    # Display user question
    st.session_state.messages[chat_mode].append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):

        st.markdown(question)

    with st.chat_message("assistant"):

        answer = ""

        try:

            with st.spinner("SmartDoc is thinking..."):

                # ==================================
                # PDF ASSISTANT MODE
                # ==================================

                if chat_mode == "PDF Assistant":

                    try:

                        collection = client.get_collection(
                            name="smartdoc"
                        )

                    except Exception:

                        collection = None

                    if collection is None or collection.count() == 0:

                        answer = (
                            "Please upload and process a PDF first."
                        )

                    else:

                        # Create question embedding
                        search_question = question

                        if re.search(
                            r"\bAI\b",
                            question,
                            re.IGNORECASE
                        ):

                            search_question += (
                                " Artificial Intelligence"
                            )

                        question_embedding = model.encode(
                            search_question,
                            normalize_embeddings=True
                        ).tolist()

                        # Search PDF
                        results = collection.query(
                            query_embeddings=[
                                question_embedding
                            ],
                            n_results=min(
                                5,
                                collection.count()
                            ),
                            include=[
                                "documents",
                                "distances"
                            ]
                        )

                        documents = results["documents"][0]
                        distances = results["distances"][0]

                        # Display retrieved information
                        with st.expander(
                            "🔍 View retrieved PDF information"
                        ):

                            for i, (doc, distance) in enumerate(
                                zip(documents, distances),
                                start=1
                            ):

                                st.write(
                                    f"Chunk {i} | Distance: {distance:.4f}"
                                )

                                st.write(doc)

                                st.divider()

                        if not documents:

                            answer = (
                                "I could not find relevant information "
                                "in your PDF."
                            )

                        else:

                            context = "\n\n".join(documents)

                            prompt = f"""
You are SmartDoc, an intelligent PDF assistant.

Answer the user's question using the supplied PDF context.

Instructions:
- Give direct and clear answers.
- Explain concepts in simple language.
- Use examples when useful.
- Do not invent facts.
- Do not use outside knowledge.
- If the information is missing, say that
  the PDF does not contain the answer.

PDF CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

                            answer = ask_ollama(prompt)


                # ==================================
                # GENERAL AI CHAT MODE
                # ==================================

                else:

                    # Include recent conversation history
                    history = ""

                    previous_messages = (
                        st.session_state.messages[chat_mode][:-1]
                    )

                    for message in previous_messages[-10:]:

                        if message["role"] == "user":

                            history += (
                                "User: "
                                + message["content"]
                                + "\n"
                            )

                        else:

                            history += (
                                "SmartDoc: "
                                + message["content"]
                                + "\n"
                            )

                    prompt = f"""
You are SmartDoc, a helpful and intelligent AI assistant.

Your personality:
- Friendly and polite.
- Clear and informative.
- Helpful to students and beginners.
- Explain difficult topics in simple language.
- Give examples when useful.
- Help with Python and programming.
- Answer general knowledge questions.
- Maintain natural conversations.

Use the previous conversation when relevant.

CONVERSATION HISTORY:
{history}

USER:
{question}

SMARTDOC:
"""

                    answer = ask_ollama(prompt)


        except requests.exceptions.ConnectionError:

            answer = (
                "Cannot connect to Ollama. "
                "Please open the Ollama application."
            )

        except requests.exceptions.Timeout:

            answer = (
                "The AI is taking too long to respond. "
                "Please try again."
            )

        except requests.exceptions.RequestException as error:

            answer = f"Ollama API error: {error}"

        except Exception as error:

            answer = f"Something went wrong: {error}"


        # Display AI answer
        st.markdown(answer)

    # Save AI response
    st.session_state.messages[chat_mode].append({
        "role": "assistant",
        "content": answer
    })


# ==========================================
# 12. FOOTER
# ==========================================

st.divider()

st.caption(
    "SmartDoc AI | Python | Streamlit | "
    "ChromaDB | Sentence Transformers | Ollama"
)
