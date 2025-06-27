from flask import Flask, request, jsonify
import whisper, os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_chroma.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_community.text_splitters import RecursiveCharacterTextSplitter


app = Flask(__name__)
model = whisper.load_model("base")

@app.route("/voice-query", methods=["POST"])
def voice_query():
    data = request.json
    audio_url = data.get("audio_url")
    # 1. download audio locally (e.g. via requests)
    # 2. transcribe using Whisper
    result = model.transcribe("downloaded.wav")
    query = result["text"]

    # 3. load your Chroma DB with the SOP
    emb = OpenAIEmbeddings()
    vectordb = Chroma(persist_directory="./chroma_db", embedding_function=emb)
    retriever = vectordb.as_retriever()

    from langchain_community.chat_models import ChatOpenAI
    from langchain import RetrievalQA

    qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(temperature=0),
                                     retriever=retriever)
    response = qa.invoke({"query": query})
    return jsonify({"answer": response["result"]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
