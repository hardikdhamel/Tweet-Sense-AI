
import json
import os

import numpy as np
import faiss

from sentence_transformers import SentenceTransformer


# Configuration
DATASET_FILE = "data/tweets.json"
INDEX_FILE = "data/tweets.index"
METADATA_FILE = "data/tweets_metadata.json"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def generate_embeddings():

    print("\n========== TWEETSENSE AI ==========")

    # Step 1: Load tweets
    print("\n[1] Loading tweets...")

    if not os.path.exists(DATASET_FILE):
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_FILE}"
        )

    with open(DATASET_FILE, "r", encoding="utf-8") as file:
        tweets = json.load(file)

    if not tweets:
        raise ValueError("No tweets found!")

    texts = [tweet["text"] for tweet in tweets]

    print("Total tweets:", len(texts))

    # Step 2: Load embedding model
    print("\n[2] Loading embedding model...")

    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Embedding model loaded!")

    # Step 3: Generate embeddings
    print("\n[3] Generating embeddings...")

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    print("Embedding shape:", embeddings.shape)

    # Step 4: Create FAISS index
    print("\n[4] Creating FAISS index...")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    print("Total vectors indexed:", index.ntotal)

    # Step 5: Save index and metadata
    print("\n[5] Saving index and metadata...")

    os.makedirs("data", exist_ok=True)

    faiss.write_index(index, INDEX_FILE)

    with open(METADATA_FILE, "w", encoding="utf-8") as file:
        json.dump(
            tweets,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\n========== FINAL REPORT ==========")

    print("Total tweets:", len(tweets))
    print("Embedding dimensions:", dimension)
    print("FAISS vectors:", index.ntotal)

    print("Index saved:", INDEX_FILE)
    print("Metadata saved:", METADATA_FILE)

    print("\n[SUCCESS] Embeddings generated successfully!")


if __name__ == "__main__":
    generate_embeddings()
