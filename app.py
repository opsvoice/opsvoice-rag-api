from flask import Flask, request, jsonify
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import whisper
import requests

app = Flask(__name__)
model = whisper.load_model("base")

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

    # 2. transcribe
    result = model.transcribe("downloaded.wav")
    query = result["text"]

    # 3. load vectorstore
    emb = OpenAIEmbeddings()
    vectordb = Chroma(persist_directory="./chroma_db", embedding_function=emb)
    retriever = vectordb.as_retriever()

    # 4. RAG
    qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(temperature=0), retriever=retriever)
    response = qa.invoke({"query": query})

    return jsonify({"answer": response["result"]})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Use the PORT env variable, default to 10000 locally
    app.run(host="0.0.0.0", port=5000)


