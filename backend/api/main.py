from fastapi import FastAPI

app = FastAPI(
    title="TCS Financial Intelligence Agent"
)

@app.get("/")
def root():
    return {"status": "healthy"}

@app.get("/ask")
def ask(question: str):
    return {"question": question}