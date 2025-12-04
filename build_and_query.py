# build_and_query.py
import json
import numpy as np
import faiss
import os

EMB_FILE = "embeddings_out/embeddings.npy"
META_FILE = "embeddings_out/meta.json"
INDEX_FILE = "embeddings_out/faiss.index"

k = 4  # number of nearest neighbors


def main():
    # -----------------------------
    # 1. Load embeddings + metadata
    # -----------------------------
    assert os.path.exists(EMB_FILE), f"{EMB_FILE} missing"
    assert os.path.exists(META_FILE), f"{META_FILE} missing"

    print("🔵 Loading embeddings and metadata...")
    embeddings = np.load(EMB_FILE)

    with open(META_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)

    n, d = embeddings.shape
    print(f"Loaded {n} embeddings of dimension {d}")

    # -----------------------------
    # 2. Build FAISS index
    # -----------------------------
    print("🔵 Building FAISS index...")
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)

    # Save index to file
    faiss.write_index(index, INDEX_FILE)
    print(f"✔ Index saved to {INDEX_FILE}")

    # -----------------------------
    # 3. Query Loop
    # -----------------------------
    print("\n🔍 Enter a query to search (or type 'exit')")

    while True:
        query = input("\nYour query: ").strip()

        if query.lower() == "exit":
            print("👋 Exiting.")
            break

        if not query:
            print("Please type something!")
            continue

        # Convert query to embedding using the same model
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
            q_emb = model.encode([query], convert_to_numpy=True)
        except Exception as e:
            print("⚠ Unable to encode query:", e)
            continue

        # Search the FAISS index
        distances, indices = index.search(q_emb, k)

        print("\n🔎 Top Matches:")
        for rank, idx in enumerate(indices[0]):
            if idx == -1:
                continue
            item = meta[idx]
            snippet = item["text"][:200].replace("\n", " ")
            print(f"{rank+1}. Page {item['page']} — {snippet}...\n")


# Ensure the script runs
if __name__ == "__main__":
    main()
