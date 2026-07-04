from fastapi import FastAPI

print("MAIN: starting import")

app = FastAPI(
    title="TCS Financial Intelligence Agent"
)

print("MAIN: app created")

@app.get("/")
def root():
    return {"status": "healthy"}

@app.get("/ask")
def ask(question: str):
    return {"question": question}


print("vector_store.py imported")