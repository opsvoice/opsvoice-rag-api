from flask import Flask, request, jsonify
from langchain.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import whisper
import requests
import os

app = Flask(__name__)

@app.route("/")
def home():
    return "🚀 OpsVoice API is live!"

@app.route("/voice-query", methods=["POST"])
def voice_query():
    data = request.json
    audio_url = data.get("audio_url")

    # 1. download audio
    response = requests.get(audio_url)
    with open("downloaded.wav", "wb") as f:
        f.write(response.content)

    # 2. load tiny model & transcribe (load inside endpoint to save RAM)
    model = whisper.load_model("tiny")
    result = model.transcribe("downloaded.wav")
    query = result["text"]

    # 3. load vectorstore
    emb = OpenAIEmbeddings()
    vectordb = Chroma(persist_directory="./chroma_db", embedding_function=emb)
    retriever = vectordb.as_retriever()

    # 4. RAG
    qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(temperature=0), retriever=retriever)
    answer = qa.invoke({"query": query})

    return jsonify({"answer": answer["result"]})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render sets PORT env variable
    app.run(host="0.0.0.0", port=port)


