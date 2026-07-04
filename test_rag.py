from backend.rag.rag_pipeline import (
    answer_question
)

question = input("Question: ")

result = answer_question(
    question
)

print("\nANSWER\n")
print(result["answer"])

print("\nSOURCES\n")

for source in result["sources"]:
    print(source)