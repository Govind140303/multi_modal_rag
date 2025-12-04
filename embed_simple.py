# embed_simple.py
import json
from sentence_transformers import SentenceTransformer
import numpy as np
import os

INPUT_FILE = "data_out/ingested.json"
OUTPUT_EMB = "embeddings_out/embeddings.npy"
OUTPUT_META = "embeddings_out/meta.json"

os.makedirs("embeddings_out", exist_ok=True)

print("🔄 Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("🔄 Loading ingested text...")
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    pages = json.load(f)

texts = [p["text"] for p in pages]

print("🔄 Creating embeddings...")
embeddings = model.encode(texts, convert_to_numpy=True)

print("💾 Saving embeddings...")
np.save(OUTPUT_EMB, embeddings)

with open(OUTPUT_META, "w", encoding="utf-8") as f:
    json.dump(pages, f, indent=2)

print("✅ Embeddings created and saved!")
