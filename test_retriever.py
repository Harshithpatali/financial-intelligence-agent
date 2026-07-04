from backend.rag.retriever import retrieve

query = "What was TCS revenue in FY2024?"

results = retrieve(query)

for i, doc in enumerate(results["documents"][0]):

    print("\n")
    print("=" * 80)
    print(f"Result {i+1}")
    print("=" * 80)

    print(doc[:1000])