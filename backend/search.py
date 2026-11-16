
import json
import os
import threading

from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

INDEX_FILE = BASE_DIR / "data" / "tweets.index"
METADATA_FILE = BASE_DIR / "data" / "tweets_metadata.json"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Use temporary storage for downloaded model files in serverless hosting.
MODEL_CACHE = os.getenv("HF_HOME", "/tmp/huggingface")


# ==========================================
# GLOBAL RESOURCES
# ==========================================

model = None
index = None
tweets = None

# Prevent multiple simultaneous initializations.
load_lock = threading.Lock()


# ==========================================
# LAZY LOADING
# ==========================================

def load_resources():
    """
    Load the embedding model, FAISS index and tweet metadata
    only when the first search request is received.

    Resources remain cached in memory for subsequent requests
    handled by the same application instance.
    """

    global model, index, tweets

    if model is not None and index is not None and tweets is not None:
        return

    with load_lock:

        # Check again after acquiring the lock.
        if model is not None and index is not None and tweets is not None:
            return

        print("\n========== LOADING TWEETSENSE RESOURCES ==========")

        # Check required files.
        if not INDEX_FILE.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {INDEX_FILE}"
            )

        if not METADATA_FILE.exists():
            raise FileNotFoundError(
                f"Tweet metadata not found: {METADATA_FILE}"
            )

        # Load embedding model on CPU.
        print("Loading embedding model...")

        loaded_model = SentenceTransformer(
            EMBEDDING_MODEL,
            cache_folder=MODEL_CACHE,
            device="cpu"
        )

        print("Embedding model loaded!")

        # Load FAISS index.
        print("Loading FAISS index...")

        loaded_index = faiss.read_index(str(INDEX_FILE))

        print("FAISS index loaded!")
        print("Total indexed vectors:", loaded_index.ntotal)

        # Load tweet metadata.
        print("Loading tweet metadata...")

        with open(METADATA_FILE, "r", encoding="utf-8") as file:
            loaded_tweets = json.load(file)

        if loaded_index.ntotal != len(loaded_tweets):
            raise ValueError(
                "FAISS index and tweet metadata counts do not match."
            )

        # Verify embedding dimensions.
        embedding_dimension = (
            loaded_model.get_sentence_embedding_dimension()
        )

        if loaded_index.d != embedding_dimension:
            raise ValueError(
                "Embedding model dimension does not match FAISS index."
            )

        # Assign resources only after successful initialization.
        model = loaded_model
        index = loaded_index
        tweets = loaded_tweets

        print("Tweet metadata loaded!")
        print("Total tweets:", len(tweets))
        print("Embedding dimensions:", embedding_dimension)
        print("========== RESOURCES READY ==========\n")


# ==========================================
# SEMANTIC TWEET SEARCH
# ==========================================

def search_tweets(query, top_k=5):

    if not isinstance(query, str) or not query.strip():
        raise ValueError("Search query cannot be empty.")

    if top_k < 1:
        raise ValueError("top_k must be greater than zero.")

    # Load resources only when needed.
    load_resources()

    print("\nSearching relevant tweets...")

    # Generate query embedding.
    query_embedding = model.encode(
        [query.strip()],
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32
    )

    # Search only as many results as the index contains.
    result_count = min(top_k, index.ntotal)

    if result_count == 0:
        return []

    scores, indices = index.search(
        query_embedding,
        result_count
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx == -1:
            continue

        results.append({
            "similarity": float(score),
            "text": tweets[idx]["text"],
            "brand": tweets[idx]["brand"]
        })

    return results


# ==========================================
# INDEPENDENT TESTING
# ==========================================

if __name__ == "__main__":

    print("\n========== TWEETSENSE AI SEARCH TEST ==========")

    query = input("\nAsk about a laptop: ")

    results = search_tweets(query)

    print("\nRelevant tweets:")

    for result in results:

        print("\nBrand:", result["brand"])
        print("Similarity:", round(result["similarity"], 3))
        print("Tweet:", result["text"])
