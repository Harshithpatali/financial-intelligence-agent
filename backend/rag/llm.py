from groq import Groq
import os

_client = None

def get_client():
    global _client

    if _client is None:
        print("Initializing Groq client...")
        _client = Groq(
            api_key=os.getenv("GROQ_API_KEY")
        )

    return _client


def generate_answer(prompt):

    client = get_client()

    print("Sending request to Groq...")

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.1
    )

    print("Groq response received")

    return response.choices[0].message.content