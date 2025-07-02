import requests

ENDPOINT = "https://opsvoice-rag-api.onrender.com/query"
headers = {"Content-Type": "application/json"}

test_queries = [
    "What are the manager opening duties?",
    "How do I submit a time-off request?",
    "Where can I find payroll procedures?",
    "Who approves expense reports?",
    "What is our vacation policy?",
    "How do I reset my password?"
]

for q in test_queries:
    r = requests.post(ENDPOINT, json={"query": q}, headers=headers)
    try:
        answer = r.json()
    except Exception:
        answer = r.text
    print(f"Q: {q}\nA: {answer}\n{'-'*40}")
