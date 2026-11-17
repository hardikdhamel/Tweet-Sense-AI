
import os
import time
from pathlib import Path

import requests
import numpy as np
import faiss

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

load_dotenv(BASE_DIR / ".env")

JINA_API_KEY = os.getenv("JINA_API_KEY")
JINA_MODEL = os.getenv(
    "JINA_EMBEDDING_MODEL",
    "jina-embeddings-v5-text-small"
)

JINA_URL = "https://api.jina.ai/v1/embeddings"

MINILM_INDEX_FILE = DATA_DIR / "tweets.index"
MINILM_METADATA_FILE = DATA_DIR / "tweets_metadata.json"

JINA_INDEX_FILE = DATA_DIR / "tweets_jina.index"
JINA_METADATA_FILE = DATA_DIR / "tweets_jina_metadata.json"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 5

TEST_QUESTIONS = [
    "What do customers think about Dell laptops?",
    "What are the common problems with HP laptops?",
    "Which laptop brands have good build quality?",
    "What do customers say about Lenovo battery life?",
    "Are Dell laptops worth buying?"
]


# ==========================================
# JINA QUERY EMBEDDING
# ==========================================

def get_jina_embedding(question):

    headers = {
        "Authorization": f"Bearer {JINA_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": JINA_MODEL,
        "input": ["Query: " + question],
        "task": "retrieval.query",
        "dimensions": 1024,
        "normalized": True
    }

    start = time.perf_counter()

    response = requests.post(
        JINA_URL,
        headers=headers,
        json=payload,
        timeout=60
    )

    response.raise_for_status()

    data = response.json()["data"][0]["embedding"]

    elapsed = time.perf_counter() - start

    vector = np.asarray(
        data,
        dtype=np.float32
    ).reshape(1, -1)

    return vector, elapsed


# ==========================================
# LOAD MODELS AND INDEXES
# ==========================================

def load_test_resources():

    print("\n========== LOADING COMPARISON RESOURCES ==========")

    if not JINA_API_KEY:
        raise ValueError("JINA_API_KEY missing from .env")

    required_files = [
        MINILM_INDEX_FILE,
        MINILM_METADATA_FILE,
        JINA_INDEX_FILE,
        JINA_METADATA_FILE
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required file not found: {file_path}"
            )

    print("\nLoading MiniLM model...")

    model = SentenceTransformer(
        EMBEDDING_MODEL,
        device="cpu"
    )

    print("MiniLM loaded!")

    print("\nLoading FAISS indexes...")

    minilm_index = faiss.read_index(
        str(MINILM_INDEX_FILE)
    )

    jina_index = faiss.read_index(
        str(JINA_INDEX_FILE)
    )

    print("Both indexes loaded!")

    import json

    with open(
        MINILM_METADATA_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        minilm_tweets = json.load(file)

    with open(
        JINA_METADATA_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        jina_tweets = json.load(file)

    if minilm_index.d != 384:
        raise ValueError("Unexpected MiniLM dimensions.")

    if jina_index.d != 1024:
        raise ValueError("Unexpected Jina dimensions.")

    if minilm_index.ntotal != len(minilm_tweets):
        raise ValueError("MiniLM index/metadata mismatch.")

    if jina_index.ntotal != len(jina_tweets):
        raise ValueError("Jina index/metadata mismatch.")

    if minilm_tweets != jina_tweets:
        raise ValueError(
            "Tweet order or metadata differs between indexes."
        )

    print("\nMiniLM vectors:", minilm_index.ntotal)
    print("Jina vectors:", jina_index.ntotal)

    return (
        model,
        minilm_index,
        jina_index,
        minilm_tweets,
        jina_tweets
    )


# ==========================================
# SEARCH AND COMPARE
# ==========================================

def compare_models():

    (
        model,
        minilm_index,
        jina_index,
        minilm_tweets,
        jina_tweets
    ) = load_test_resources()

    total_minilm_latency = 0
    total_jina_latency = 0

    print("\n========== MODEL COMPARISON ==========")

    for number, question in enumerate(TEST_QUESTIONS, start=1):

        print("\n" + "=" * 70)
        print(f"QUESTION {number}: {question}")
        print("=" * 70)

        # MiniLM query embedding
        start = time.perf_counter()

        minilm_vector = model.encode(
            [question],
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        minilm_vector = np.asarray(
            minilm_vector,
            dtype=np.float32
        )

        minilm_embedding_time = time.perf_counter() - start

        # MiniLM FAISS search
        start = time.perf_counter()

        minilm_scores, minilm_ids = minilm_index.search(
            minilm_vector,
            TOP_K
        )

        minilm_search_time = time.perf_counter() - start

        minilm_total = (
            minilm_embedding_time + minilm_search_time
        )

        # Jina query embedding
        jina_vector, jina_embedding_time = get_jina_embedding(
            question
        )

        # Jina FAISS search
        start = time.perf_counter()

        jina_scores, jina_ids = jina_index.search(
            jina_vector,
            TOP_K
        )

        jina_search_time = time.perf_counter() - start

        jina_total = (
            jina_embedding_time + jina_search_time
        )

        total_minilm_latency += minilm_total
        total_jina_latency += jina_total

        # Display MiniLM results
        print("\n---------- MINILM RESULTS ----------")

        for rank, (score, idx) in enumerate(
            zip(minilm_scores[0], minilm_ids[0]),
            start=1
        ):

            tweet = minilm_tweets[idx]

            print(f"\n{rank}. Brand: {tweet['brand']}")
            print(f"   Score: {score:.4f}")
            print(f"   Tweet: {tweet['text']}")

        # Display Jina results
        print("\n---------- JINA RESULTS ----------")

        for rank, (score, idx) in enumerate(
            zip(jina_scores[0], jina_ids[0]),
            start=1
        ):

            tweet = jina_tweets[idx]

            print(f"\n{rank}. Brand: {tweet['brand']}")
            print(f"   Score: {score:.4f}")
            print(f"   Tweet: {tweet['text']}")

        # Latency comparison
        print("\n---------- LATENCY ----------")

        print(
            f"MiniLM embedding: {minilm_embedding_time * 1000:.2f} ms"
        )
        print(
            f"MiniLM FAISS search: {minilm_search_time * 1000:.2f} ms"
        )
        print(
            f"MiniLM total: {minilm_total * 1000:.2f} ms"
        )

        print(
            f"Jina API embedding: {jina_embedding_time * 1000:.2f} ms"
        )
        print(
            f"Jina FAISS search: {jina_search_time * 1000:.2f} ms"
        )
        print(
            f"Jina total: {jina_total * 1000:.2f} ms"
        )

    # Final average
    count = len(TEST_QUESTIONS)

    print("\n" + "=" * 70)
    print("FINAL AVERAGE LATENCY")
    print("=" * 70)

    print(
        f"MiniLM: {(total_minilm_latency / count) * 1000:.2f} ms"
    )

    print(
        f"Jina: {(total_jina_latency / count) * 1000:.2f} ms"
    )

    print("\nComparison completed!")


if __name__ == "__main__":
    compare_models()
