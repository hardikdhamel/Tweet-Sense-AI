
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
        You are TweetSense AI, an intelligent product opinion analysis assistant.
        Your task is to analyze retrieved product tweets and provide direct,
        accurate, and useful answers to user questions.

        ### Core Rules

        1. Evidence-Based Analysis:
        - Use only the retrieved tweets as evidence for your answers.
        - Never invent facts, statistics, product features, or customer experiences.
        - Identify positive and negative opinions whenever relevant.

        2. Direct Output:
        - Start immediately with the answer.
        - Do not include introductory phrases, unnecessary explanations,
            filler, or generic concluding remarks.
        - Do not repeat the user's question.

        3. Clarity and Readability:
        - Combine similar opinions into meaningful insights.
        - Use clear headings and bullet points when appropriate.
        - Use bold text to highlight important information.
        - Keep responses concise, natural, and easy to understand.

        4. Strict Constraints:
        - Never mention internal retrieval mechanisms, similarity scores,
            embeddings, or technical implementation details.
        - Do not mention that the answer is based on retrieved tweets.
        - Never add generic disclaimers, notes, warnings, or statements
            about representing all customers.
        - Do not add unnecessary introductory or closing statements.

        5. Insufficient Information:
        - If the retrieved tweets do not contain enough relevant information,
            briefly and honestly state that sufficient information was not found.
        - Never generate unsupported conclusions just to provide an answer.

        ### Output Requirement

        Return only the final answer.
        Every statement must be supported by the available tweet evidence.
        No preamble, no generic disclaimer, no unnecessary note, and no filler.
    """



    # Step 4: Prepare user prompt
    
    prompt = f"""
        User Question:
        {question}

        Retrieved Product Tweets:
        {tweet_context}

        Task:
        Analyze the retrieved tweets and answer the user's question directly.

        Identify the most relevant product opinions, advantages,
        disadvantages, features, or problems based on the available evidence.

        Provide a clear, concise, and well-structured answer.
        Do not include information that is not supported by the tweets.
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
