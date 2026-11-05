
import os
import requests
from dotenv import load_dotenv
from search import search_tweets

# Load environment variables
load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:latest")

if not OLLAMA_BASE_URL:
    raise ValueError("OLLAMA_BASE_URL is missing from .env")

OLLAMA_URL = f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat"


def generate_answer(question):

    print("\nSearching relevant tweets...")

    # Retrieve relevant tweets using FAISS
    results = search_tweets(question, top_k=5)

    if not results:
        return {
            "answer": "I couldn't find relevant tweets for your question.",
            "sources": []
        }

    # Prepare tweet context
    context = ""

    for i, tweet in enumerate(results, start=1):
        context += f"""
Tweet {i}:
Brand: {tweet['brand']}
Content: {tweet['text']}
Similarity: {tweet['similarity']:.3f}
"""

    # System prompt
    system_prompt = """
You are TweetSense AI, a product opinion analysis assistant.

Your task is to answer user questions based on retrieved tweets.

Instructions:
1. Use only the supplied tweet evidence.
2. Analyze positive and negative opinions.
3. Give balanced and understandable answers.
4. Never invent product specifications or opinions.
5. Clearly mention when evidence is insufficient.
6. Do not treat advertisements as verified customer experiences.
7. Mention advantages and disadvantages when available.
8. Give a short conclusion.
9. Refer to supporting tweets using their tweet numbers.

Keep your answer concise and conversational.
"""

    user_prompt = f"""
User Question:
{question}

Retrieved Tweet Context:
{context}

Analyze the tweets and answer the user's question.
"""

    print("Generating AI response using Ollama...")

    try:

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],
                "stream": False,
                "options": {
                    "temperature": 0.3
                }
            },
            timeout=180
        )

        response.raise_for_status()

        answer = response.json()["message"]["content"]

        return {
            "answer": answer,
            "sources": results
        }

    except requests.exceptions.Timeout:

        return {
            "answer": "The AI model took too long to respond. Please try again.",
            "sources": results
        }

    except requests.exceptions.ConnectionError:

        return {
            "answer": "Unable to connect to the AI model. Please check the Ollama URL.",
            "sources": results
        }

    except requests.exceptions.HTTPError as e:

        print("Ollama HTTP Error:", str(e))

        return {
            "answer": "The AI service returned an error. Please check the model and API configuration.",
            "sources": results,
            "error": str(e)
        }

    except Exception as e:

        print("Ollama Error:", str(e))

        return {
            "answer": "Unable to generate an AI response.",
            "sources": results,
            "error": str(e)
        }


# Run chatbot independently
if __name__ == "__main__":

    print("\n========== TWEETSENSE AI CHATBOT ==========")

    question = input("\nAsk your question: ")

    result = generate_answer(question)

    print("\n========== AI RESPONSE ==========")
    print(result["answer"])

    print("\n========== SUPPORTING TWEETS ==========")

    for i, tweet in enumerate(result["sources"], start=1):

        print(f"\nTweet {i}")
        print("Brand:", tweet["brand"])
        print("Similarity:", round(tweet["similarity"], 3))
        print("Content:", tweet["text"])
