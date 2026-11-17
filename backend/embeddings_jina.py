
import os
import json
import time
from pathlib import Path

import requests
import numpy as np
import faiss
from dotenv import load_dotenv


# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

DATASET_FILE = DATA_DIR / "tweets.json"

# Separate Jina files: original MiniLM files remain untouched.
INDEX_FILE = DATA_DIR / "tweets_jina.index"
METADATA_FILE = DATA_DIR / "tweets_jina_metadata.json"

load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("JINA_API_KEY")
MODEL = os.getenv(
    "JINA_EMBEDDING_MODEL",
    "jina-embeddings-v5-text-small"
)

API_URL = "https://api.jina.ai/v1/embeddings"

DIMENSIONS = 1024
BATCH_SIZE = 32
MAX_RETRIES = 5


# ==========================================
# JINA EMBEDDING FUNCTION
# ==========================================

def generate_embeddings(texts):

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL,
        "input": texts,
        "task": "retrieval.passage",
        "dimensions": DIMENSIONS,
        "normalized": True
    }

    for attempt in range(MAX_RETRIES):

        try:
            response = requests.post(
                API_URL,
                headers=headers,
                json=payload,
                timeout=120
            )

            if response.status_code == 429:
                wait_time = min(2 ** attempt, 60)
                print(f"Rate limited. Waiting {wait_time}s...")
                time.sleep(wait_time)
                continue

            if response.status_code >= 500:
                wait_time = min(2 ** attempt, 60)
                print(f"Jina server error. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue

            response.raise_for_status()

            result = response.json()

            # Restore original input order using API index.
            items = sorted(
                result["data"],
                key=lambda item: item["index"]
            )

            vectors = [
                item["embedding"]
                for item in items
            ]

            embeddings = np.asarray(
                vectors,
                dtype=np.float32
            )

            if embeddings.shape != (len(texts), DIMENSIONS):
                raise ValueError(
                    f"Unexpected embedding shape: {embeddings.shape}"
                )

            return embeddings

        except requests.exceptions.RequestException as error:

            if attempt == MAX_RETRIES - 1:
                raise

            wait_time = min(2 ** attempt, 60)

            print("Request failed:", str(error))
            print(f"Retrying in {wait_time}s...")

            time.sleep(wait_time)

    raise RuntimeError("Jina embedding generation failed.")


# ==========================================
# MAIN PIPELINE
# ==========================================

def generate_jina_index():

    print("\n========== JINA EMBEDDING PIPELINE ==========")

    if not API_KEY:
        raise ValueError("JINA_API_KEY missing from .env")

    # Step 1: Load dataset.
    print("\n[1] Loading tweets...")

    with open(DATASET_FILE, "r", encoding="utf-8") as file:
        tweets = json.load(file)

    if not tweets:
        raise ValueError("No tweets found!")

    texts = []

    for tweet in tweets:

        if not isinstance(tweet.get("text"), str):
            raise ValueError("Invalid tweet text found.")

        # Jina model's retrieval convention for passages.
        texts.append("Document: " + tweet["text"].strip())

    print("Total tweets:", len(texts))

    # Step 2: Generate embeddings in batches.
    print("\n[2] Generating Jina embeddings...")

    all_embeddings = []

    total_batches = (
        len(texts) + BATCH_SIZE - 1
    ) // BATCH_SIZE

    for start in range(0, len(texts), BATCH_SIZE):

        batch = texts[start:start + BATCH_SIZE]

        batch_number = (
            start // BATCH_SIZE
        ) + 1

        print(
            f"Processing batch {batch_number}/{total_batches} "
            f"({len(batch)} tweets)"
        )

        batch_embeddings = generate_embeddings(batch)

        all_embeddings.append(batch_embeddings)

        print("Batch completed!")

    embeddings = np.vstack(all_embeddings).astype(np.float32)

    print("\nEmbedding shape:", embeddings.shape)

    # Step 3: Verify normalization.
    print("\n[3] Validating embeddings...")

    norms = np.linalg.norm(embeddings, axis=1)

    if not np.allclose(norms, 1.0, atol=1e-3):
        raise ValueError("Embeddings are not properly normalized.")

    if len(embeddings) != len(tweets):
        raise ValueError("Tweet and embedding counts do not match.")

    # Step 4: Build FAISS index.
    print("\n[4] Building FAISS index...")

    index = faiss.IndexFlatIP(DIMENSIONS)

    index.add(embeddings)

    # Step 5: Save separate Jina files.
    print("\n[5] Saving Jina index and metadata...")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(INDEX_FILE))

    with open(METADATA_FILE, "w", encoding="utf-8") as file:
        json.dump(
            tweets,
            file,
            indent=4,
            ensure_ascii=False
        )

    # Step 6: Final report.
    print("\n========== FINAL REPORT ==========")

    print("Model:", MODEL)
    print("Total tweets:", len(tweets))
    print("Embedding dimensions:", DIMENSIONS)
    print("FAISS vectors:", index.ntotal)

    print("Index saved:", INDEX_FILE)
    print("Metadata saved:", METADATA_FILE)

    print("\n[SUCCESS] Jina embeddings generated!")


if __name__ == "__main__":
    generate_jina_index()