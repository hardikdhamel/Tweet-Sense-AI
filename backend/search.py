
import json
import os
import threading
from pathlib import Path

import faiss
import numpy as np
import requests
from dotenv import load_dotenv


# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

INDEX_FILE = BASE_DIR / "data" / "tweets_jina.index"
METADATA_FILE = BASE_DIR / "data" / "tweets_jina_metadata.json"

JINA_API_URL = "https://api.jina.ai/v1/embeddings"
JINA_EMBEDDING_MODEL = os.getenv(
    "JINA_EMBEDDING_MODEL",
    "jina-embeddings-v5-text-small"
)

load_dotenv(BASE_DIR / ".env")

JINA_API_KEY = os.getenv("JINA_API_KEY")


# ==========================================
# GLOBAL RESOURCES
# ==========================================

index = None
tweets = None

load_lock = threading.Lock()


# ==========================================
# LAZY LOADING
# ==========================================

def load_resources():
    """
    Load the Jina FAISS index and tweet metadata
    only when the first search request is received.
    """

    global index, tweets

    if index is not None and tweets is not None:
        return

    with load_lock:

        if index is not None and tweets is not None:
            return

        print("\n========== LOADING TWEETSENSE RESOURCES ==========")

        if not JINA_API_KEY:
            raise ValueError(
                "JINA_API_KEY is missing. Please configure it in your .env file."
            )

        if not INDEX_FILE.exists():
            raise FileNotFoundError(
                f"Jina FAISS index not found: {INDEX_FILE}"
            )

        if not METADATA_FILE.exists():
            raise FileNotFoundError(
                f"Jina tweet metadata not found: {METADATA_FILE}"
            )

        print("Loading Jina FAISS index...")

        loaded_index = faiss.read_index(str(INDEX_FILE))

        print("FAISS index loaded!")
        print("Total indexed vectors:", loaded_index.ntotal)

        print("Loading tweet metadata...")

        with open(METADATA_FILE, "r", encoding="utf-8") as file:
            loaded_tweets = json.load(file)

        if loaded_index.ntotal != len(loaded_tweets):
            raise ValueError(
                "FAISS index and tweet metadata counts do not match."
            )

        if loaded_index.ntotal == 0:
            raise ValueError("The Jina FAISS index is empty.")

        if loaded_index.d != 1024:
            raise ValueError(
                f"Expected 1024 dimensions, but index has {loaded_index.d}."
            )

        index = loaded_index
        tweets = loaded_tweets

        print("Tweet metadata loaded!")
        print("Total tweets:", len(tweets))
        print("Embedding dimensions:", index.d)

        print("========== RESOURCES READY ==========\n")


# ==========================================
# JINA QUERY EMBEDDING
# ==========================================

def generate_query_embedding(query):
    """
    Generate a normalized query embedding using
    the Jina Embedding API.
    """

    if not JINA_API_KEY:
        raise ValueError("JINA_API_KEY is missing.")

    headers = {
        "Authorization": f"Bearer {JINA_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": JINA_EMBEDDING_MODEL,
        "task": "retrieval.query",
        "dimensions": 1024,
        "normalized": True,
        "input": [f"Query: {query.strip()}"]
    }

    try:
        response = requests.post(
            JINA_API_URL,
            headers=headers,
            json=payload,
            timeout=30
        )

        response.raise_for_status()

        response_data = response.json()

        embedding = response_data["data"][0]["embedding"]

        query_embedding = np.asarray(
            embedding,
            dtype=np.float32
        ).reshape(1, -1)

        if query_embedding.shape[1] != 1024:
            raise ValueError(
                f"Expected 1024 embedding dimensions, "
                f"received {query_embedding.shape[1]}."
            )

        if not np.all(np.isfinite(query_embedding)):
            raise ValueError("Jina returned invalid embedding values.")

        return query_embedding

    except requests.exceptions.RequestException as error:
        print("Jina API request failed:", str(error))
        raise RuntimeError(
            "Unable to generate query embedding using Jina API."
        ) from error

    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError(
            "Unexpected response received from Jina API."
        ) from error


# ==========================================
# SEMANTIC TWEET SEARCH
# ==========================================

def search_tweets(query, top_k=5):

    if not isinstance(query, str) or not query.strip():
        raise ValueError("Search query cannot be empty.")

    if not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")

    # Load resources only when needed.
    load_resources()

    print("\nSearching relevant tweets using Jina...")

    # Generate query embedding through Jina API.
    query_embedding = generate_query_embedding(query)

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
