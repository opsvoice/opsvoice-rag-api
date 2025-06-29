from flask import Flask, request, jsonify
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
import os

app = Flask(__name__)

# Load ChromaDB vectorstore (DO NOT re-embed here!)
persist_directory = "./chroma_db"
embedding = OpenAIEmbeddings()
vectorstore = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding,
)
retriever = vectorstore.as_retriever()
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature=0),
    chain_type="stuff",
    retriever=retriever
)

@app.route("/query", methods=["POST"])
def query_sop():
    data = request.get_json()
    user_query = data.get("query")
    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    # Try to answer from SOPs
    sop_answer = qa_chain.run(user_query)
    # Fallback logic
    if not sop_answer or "don't know" in sop_answer.lower() or "no information" in sop_answer.lower():
        llm = ChatOpenAI(temperature=0)
        prompt = f"The company SOPs do not cover this. Please provide a general business best practice for: {user_query}"
        best_practice_answer = llm.invoke(prompt)
        return jsonify({
            "source": "general_best_practice",
            "answer": best_practice_answer.content
        })
    else:
        return jsonify({
            "source": "sop",
            "answer": sop_answer
        })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)



