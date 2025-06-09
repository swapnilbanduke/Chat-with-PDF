# app.py ─────────────────────────────────────────────────────────────
# 0️⃣  Imports ­and page config ( MUST be first Streamlit call )
import streamlit as st
st.set_page_config(page_title="Chat with PDF", layout="wide", page_icon="📄")

import os, io
from typing import List, Dict

import openai
import PyPDF2
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ───────────────────────── CONFIG ────────────────────────── #
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

CHUNK_SIZE = 500          # characters per chunk
TOP_K      = 3            # how many chunks to send to GPT


@st.cache_resource(show_spinner="🔄 Loading embedding model …")
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")


EMBED_MODEL = get_embedder()

# ───────────────────────── HELPERS ───────────────────────── #
def extract_chunks(file: io.BytesIO, chunk_size: int = CHUNK_SIZE) -> List[str]:
    """Return a list of text chunks from a PDF file-like object."""
    try:
        reader = PyPDF2.PdfReader(file)
    except Exception as e:
        st.error(f"❌ Could not read PDF: {e}")
        return []

    chunks = []
    for page in reader.pages:
        text = page.extract_text() or ""
        for i in range(0, len(text), chunk_size):
            chunks.append(text[i : i + chunk_size])
    return chunks


def embed_chunks(chunks: List[str]) -> np.ndarray:
    """Return 2-D NumPy array of sentence embeddings."""
    return EMBED_MODEL.encode(chunks, convert_to_tensor=True).cpu().numpy()


def find_best_chunks(
    query: str, chunks: List[str], embeds: np.ndarray, k: int = TOP_K
) -> List[str]:
    """Return top-k most similar chunks."""
    q_emb = EMBED_MODEL.encode([query], convert_to_tensor=True).cpu().numpy()
    sims  = cosine_similarity(q_emb, embeds)[0]      # shape: (n_chunks,)
    idxs  = np.argsort(sims)[-k:][::-1]              # best → worst
    return [chunks[i] for i in idxs]


# ───────────────────────── SIDEBAR ───────────────────────── #
st.sidebar.title("📑 Upload PDF")
pdf_file = st.sidebar.file_uploader("Choose a PDF file", type=["pdf"])
st.sidebar.markdown("---")
st.sidebar.markdown("Built with **Streamlit chat**, Sentence-Transformers, OpenAI GPT-3.5-turbo")

# ───────────────────────── SESSION STATE ─────────────────── #
if "messages"   not in st.session_state: st.session_state.messages   = []  # chat history
if "chunks"     not in st.session_state: st.session_state.chunks     = []
if "embeddings" not in st.session_state: st.session_state.embeddings = None

# ───────────────────────── PROCESS PDF ───────────────────── #
if pdf_file and not st.session_state.chunks:
    with st.spinner("📚 Reading & embedding PDF …"):
        chunks = extract_chunks(pdf_file)
        if chunks:
            st.session_state.chunks     = chunks
            st.session_state.embeddings = embed_chunks(chunks)
        else:
            st.error("No selectable text detected in this PDF.")

# ───────────────────────── DISPLAY HISTORY ───────────────── #
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ───────────────────────── CHAT INPUT ────────────────────── #
prompt = st.chat_input("Ask a question about the PDF")

if prompt is None:           # nothing typed yet; just render page
    st.stop()

# ─────────────── Echo user prompt & store in history ─────── #
st.session_state.messages.append({"role": "user", "content": prompt})
with st.chat_message("user"):
    st.write(prompt)

# ─────────────── Guard rails ─────────────────────────────── #
if not pdf_file:
    with st.chat_message("assistant"):
        st.warning("📑 Please upload a PDF first.")
    st.stop()

if not st.session_state.chunks:
    with st.chat_message("assistant"):
        st.warning("⏳ Still processing the PDF—try again in a few seconds.")
    st.stop()

# ─────────────── Find context & call OpenAI ──────────────── #
top_chunks = find_best_chunks(prompt,
                              st.session_state.chunks,
                              st.session_state.embeddings)
context = "\n---\n".join(top_chunks)

MEMORY_TURNS = 4
history_slice = st.session_state.messages[-MEMORY_TURNS * 2 :]

messages = [{"role": "system",
             "content": "You are a helpful assistant."}]
messages += history_slice
messages.append({
    "role": "user",
    "content": f"Context from the PDF:\n{context}\n\n"
               f"Answer strictly from the context.\n"
               f"Question: {prompt}"
})

with st.spinner("🤖 Generating answer …"):
    try:
        resp = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.2,
        )
        answer = resp.choices[0].message.content.strip()
    except Exception as e:
        answer = f"Error from OpenAI: {e}"

# ─────────────── Show assistant reply & save ─────────────── #
st.session_state.messages.append({"role": "assistant", "content": answer})
with st.chat_message("assistant"):
    st.write(answer)
