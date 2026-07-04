import os
from groq import Groq

print("llm.py imported")

client = Groq(
    api_key=os.getenv(
        "GROQ_API_KEY"
    )
)


def generate_answer(prompt):

    print(
        "Sending prompt to Groq..."
    )

    response = (
        client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,
            max_tokens=1200
        )
    )

    print(
        "Groq response received"
    )

    return (
        response
        .choices[0]
        .message.content
    )