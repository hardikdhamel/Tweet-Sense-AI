
import os
from dotenv import load_dotenv
from google import genai

from search import search_tweets

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing from .env")

client = genai.Client(api_key=GEMINI_API_KEY)


def generate_answer(question: str):
    """
    Retrieve relevant tweets using FAISS and generate
    a grounded answer using the Gemini API.
    """

    # Step 1: Retrieve relevant tweets
    results = search_tweets(question, top_k=5)

    if not results:
        return {
            "answer": "I couldn't find relevant tweets for your question. Please try asking about another product.",
            "sources": []
        }

    # Step 2: Prepare tweet context
    tweet_context = "\n\n".join(
        [
            f"Tweet {i + 1}:\n"
            f"Brand: {tweet.get('brand', 'Unknown')}\n"
            f"Text: {tweet.get('text', '')}\n"
            f"Similarity: {tweet.get('similarity', 0):.3f}"
            for i, tweet in enumerate(results)
        ]
    )

    # Step 3: System instructions
    system_instruction = """
    You are TweetSense AI, an intelligent product review
    and opinion analysis assistant.

    Your job is to answer user questions using the provided tweets.

    Rules:
    1. Base your answer primarily on the provided tweets.
    2. Do not invent customer opinions, reviews, or statistics.
    3. If the tweets do not contain enough information,
       clearly mention that.
    4. Identify positive and negative opinions when relevant.
    5. Give clear, simple, and useful answers.
    6. Do not claim that the tweets represent all customers.
    7. If the question is unrelated to product opinions,
       politely explain what you can help with.
    """

    # Step 4: Prepare user prompt
    prompt = f"""
    User Question:
    {question}

    Retrieved Tweets:
    {tweet_context}

    Based on these tweets, answer the user's question.
    """

    # Step 5: Generate response using Gemini
    try:
        interaction = client.interactions.create(
            model=GEMINI_MODEL,
            system_instruction=system_instruction,
            input=prompt,
            generation_config={
                "temperature": 0.3,
                "max_output_tokens": 512
            }
        )

        answer = interaction.output_text

        if not answer:
            raise ValueError("Gemini returned an empty response.")

        return {
            "answer": answer,
            "sources": results
        }

    except Exception as error:
        print("Gemini API Error:", str(error))
        raise RuntimeError(
            "Unable to generate an answer right now. Please try again later."
        ) from error


# Local testing
if __name__ == "__main__":

    test_question = "What do customers think about Dell laptops?"

    response = generate_answer(test_question)

    print("\n========== TWEETSENSE AI ==========")
    print("\nQuestion:", test_question)
    print("\nAI Answer:", response["answer"])

    print("\n========== SOURCES ==========")

    for source in response["sources"]:
        print("\nBrand:", source.get("brand"))
        print("Similarity:", source.get("similarity"))
        print("Tweet:", source.get("text"))
