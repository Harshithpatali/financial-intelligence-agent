from fastapi import FastAPI,HTTPException,Query
from pydantic import BaseModel,Field
import logging
app=FastAPI(title="TCS Financial Intelligence Agent",version="2.0.0",description="Source-grounded financial research using hybrid Graph RAG.")
logger=logging.getLogger(__name__)
class AskRequest(BaseModel):
 question:str=Field(min_length=3,max_length=2000)
@app.get("/")
def root():return {"status":"healthy","service":"tcs-financial-intelligence-agent","version":"2.0.0"}
@app.get("/health")
def health():return {"status":"ok"}
@app.get("/ask")
def ask(question:str=Query(min_length=3,max_length=2000)):return _answer(question)
@app.post("/ask")
def ask_post(payload:AskRequest):return _answer(payload.question)
def _answer(question):
 try:
  from backend.rag.rag_pipeline import answer_question
  return answer_question(question)
 except ValueError as exc:raise HTTPException(status_code=422,detail=str(exc)) from exc
 except Exception as exc:
  logger.exception("Question answering failed")
  raise HTTPException(status_code=503,detail="The research service is temporarily unavailable.") from exc
