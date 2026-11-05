
import json
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer


INDEX_FILE = "data/tweets.index"
METADATA_FILE = "data/tweets_metadata.json"

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

index = faiss.read_index(INDEX_FILE)

with open(METADATA_FILE, "r", encoding="utf-8") as file:
    tweets = json.load(file)


def search_tweets(query, top_k=5):

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32
    )

    scores, indices = index.search(
        query_embedding,
        top_k
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


if __name__ == "__main__":

    query = input("\nAsk about a laptop: ")

    results = search_tweets(query)

    print("\nRelevant tweets:")

    for result in results:

        print("\nBrand:", result["brand"])
        print("Similarity:", round(result["similarity"], 3))
        print("Tweet:", result["text"])
