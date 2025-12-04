import streamlit as st
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_FILE = "embeddings_out/faiss.index"
META_FILE = "embeddings_out/meta.json"

st.set_page_config(page_title="Local RAG Demo", layout="centered")
st.title("Local RAG — Document QA Demo")

if "index" not in st.session_state:
    st.session_state["index"] = None
if "meta" not in st.session_state:
    st.session_state["meta"] = None
if "model" not in st.session_state:
    st.session_state["model"] = None
if "loaded" not in st.session_state:
    st.session_state["loaded"] = False
if "status_msg" not in st.session_state:
    st.session_state["status_msg"] = "Click 'Load index & model' to initialize (first time may take 10–60s)."
if "query" not in st.session_state:
    st.session_state["query"] = ""

def load_resources():
    if st.session_state["loaded"]:
        st.session_state["status_msg"] = f"Index already loaded ({len(st.session_state['meta'])} chunks)."
        return
    try:
        index = faiss.read_index(INDEX_FILE)
        with open(META_FILE, "r", encoding="utf-8") as f:
            meta = json.load(f)
        model = SentenceTransformer("all-MiniLM-L6-v2")
        st.session_state["index"] = index
        st.session_state["meta"] = meta
        st.session_state["model"] = model
        st.session_state["loaded"] = True
        st.session_state["status_msg"] = f"Index loaded ({len(meta)} chunks). Model ready."
    except Exception as e:
        st.session_state["status_msg"] = f"Load failed: {e}"
        st.error(f"Load failed: {e}")

col1, col2 = st.columns([1, 3])
with col1:
    if st.button("Load index & model", key="load_btn"):
        with st.spinner("Loading resources..."):
            load_resources()

with col2:
    st.info(st.session_state["status_msg"])

if not st.session_state["loaded"]:
    st.stop()

query = st.text_input("Enter a question about the document:", key="query")
k = st.slider("Top K", 1, 12, 4)
if st.button("Search", key="search_btn"):
    if not st.session_state["query"].strip():
        st.warning("Please enter a query.")
    else:
        model = st.session_state["model"]
        index = st.session_state["index"]
        meta = st.session_state["meta"]
        q_emb = model.encode([st.session_state["query"]], normalize_embeddings=True).astype("float32")
        distances, indices = index.search(q_emb, k)
        st.write("### 🔎 Top Matches:")
        for i, idx in enumerate(indices[0], start=1):
            if idx == -1:
                continue
            item = meta[idx]
            snippet = (item.get("text","") or "")[:400].replace("\n", " ")
            st.write(f"**{i}. Page {item.get('page','?')} —** {snippet} ...")
