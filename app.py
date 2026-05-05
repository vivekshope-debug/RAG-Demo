import streamlit as st
import io
import numpy as np

# -------- FAISS SAFE IMPORT --------
try:
    import faiss
except:
    import faiss_cpu as faiss

# -------- LOCAL EMBEDDINGS --------
from sentence_transformers import SentenceTransformer

# -------- GEMINI --------
import google.generativeai as genai

st.set_page_config(page_title="RAG Demo (Full)", layout="centered")
st.title("📚 RAG Demo (Files + Text + Cosine Similarity)")

# ---------------- CONFIG ----------------
st.sidebar.header("🔑 API Key")
gemini_api = st.sidebar.text_input("Gemini API Key", type="password")

GEN_MODEL = "gemini-2.5-flash"

if gemini_api:
    genai.configure(api_key=gemini_api)

# -------- LOAD EMBEDDING MODEL --------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# ---------------- FILE READING ----------------
def read_files(files):
    docs = []
    for f in files:
        name = f.name
        content = ""

        try:
            if name.lower().endswith(".pdf"):
                from PyPDF2 import PdfReader
                reader = PdfReader(io.BytesIO(f.read()))
                content = "\n".join([p.extract_text() or "" for p in reader.pages])
            else:
                content = f.read().decode("utf-8", errors="ignore")
        except:
            content = ""

        if content.strip():
            docs.append((name, content))

    return docs

# ---------------- CHUNKING ----------------
def chunk_text(text, size=800, overlap=120):
    chunks = []
    start = 0

    while start < len(text):
        chunk = text[start:start + size]
        if chunk.strip():
            chunks.append(chunk)
        start += size - overlap

    return chunks

# ---------------- EMBEDDING ----------------
def embed_texts(texts):
    embeddings = embedder.encode(texts)

    # normalize → cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    return np.array(embeddings).astype("float32")

# ---------------- FAISS ----------------
def build_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index

# ---------------- RETRIEVAL ----------------
def retrieve(query, index, chunks, k=4):
    q_emb = embed_texts([query])
    D, I = index.search(q_emb, k)

    results = []
    for score, idx in zip(D[0], I[0]):
        if 0 <= idx < len(chunks):
            results.append({
                "text": chunks[idx],
                "score": float(score)
            })

    return results, q_emb

# ---------------- GENERATION ----------------
def generate_answer(query, contexts=None):
    model = genai.GenerativeModel(GEN_MODEL)

    if contexts:
        context_block = "\n\n---\n\n".join([c["text"] for c in contexts])
        prompt = f"""
You are a helpful assistant.

Answer ONLY using the provided context.
If the answer is not present, say "I don't know".

Context:
{context_block}

Question:
{query}
"""
    else:
        # fallback → direct LLM
        prompt = query

    return model.generate_content(prompt).text

# ---------------- SESSION ----------------
if "index" not in st.session_state:
    st.session_state.index = None
    st.session_state.chunks = []
    st.session_state.embeddings = None

# ---------------- UI ----------------
st.header("1️⃣ Upload Documents OR Paste Text")

files = st.file_uploader(
    "Upload PDF / TXT / MD",
    type=["pdf", "txt", "md"],
    accept_multiple_files=True
)

manual_text = st.text_area("Or paste your text here")

# ---------------- BUILD INDEX ----------------
if st.button("Build Index"):
    all_chunks = []

    # files
    if files:
        st.info("Reading documents...")
        docs = read_files(files)
        for name, text in docs:
            chunks = chunk_text(text)
            all_chunks.extend([f"[{name}] {c}" for c in chunks])

    # manual text
    if manual_text.strip():
        st.info("Processing input text...")
        chunks = chunk_text(manual_text)
        all_chunks.extend([f"[Manual Input] {c}" for c in chunks])

    if not all_chunks:
        st.error("Provide file or text to build index")
        st.stop()

    st.info(f"Created {len(all_chunks)} chunks")

    st.info("Generating embeddings...")
    embeddings = embed_texts(all_chunks)

    st.info("Building FAISS index...")
    index = build_index(embeddings)

    st.session_state.index = index
    st.session_state.chunks = all_chunks
    st.session_state.embeddings = embeddings

    st.success("Index ready!")

    # show embedding preview
    st.markdown("### 🧠 Sample Embedding Vector (first chunk)")
    st.write(embeddings[0][:10])

# ---------------- QUERY ----------------
st.header("2️⃣ Ask Questions")

query = st.text_input("Enter your question")
k = st.slider("Top-K results", 2, 8, 4)

if st.button("Search & Answer"):
    if not query:
        st.warning("Enter a query")
    elif not gemini_api:
        st.error("Enter Gemini API key")
    else:
        # if index exists → RAG
        if st.session_state.index is not None:
            st.info("Retrieving context...")
            contexts, q_emb = retrieve(query, st.session_state.index, st.session_state.chunks, k)

            st.markdown("### 🔎 Retrieved Context")
            for i, r in enumerate(contexts, 1):
                st.markdown(f"""
**Chunk {i} (Score: {r['score']:.4f})**  
{r['text'][:300]}...
""")

            st.markdown("### 🧠 Query Vector (first 10 values)")
            st.write(q_emb[0][:10])

            answer = generate_answer(query, contexts)

        else:
            st.info("No index found → using LLM directly")
            answer = generate_answer(query)

        st.markdown("### 🤖 Answer")
        st.write(answer)
