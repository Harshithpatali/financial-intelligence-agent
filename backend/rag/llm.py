import os
from groq import Groq
def generate_answer(prompt):
 api_key=os.getenv("GROQ_API_KEY")
 if not api_key:raise RuntimeError("GROQ_API_KEY is not configured")
 response=Groq(api_key=api_key).chat.completions.create(model=os.getenv("GROQ_MODEL","llama-3.3-70b-versatile"),messages=[{"role":"user","content":prompt}],temperature=0.1,max_tokens=1200)
 return response.choices[0].message.content
