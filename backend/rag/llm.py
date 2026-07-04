from groq import Groq
import os

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

def generate_answer(prompt):

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

print("llm.py imported")