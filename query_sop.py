from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
import os

# 1. Load ChromaDB with existing embeddings (DON'T re-embed every time)
persist_directory = "/data/chroma_db"   # <--- use persistent Render disk path!
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

query = input("Ask your question: ")
sop_answer = qa_chain.run(query)

# Check for fallback
if not sop_answer or "don't know" in sop_answer.lower() or "no information" in sop_answer.lower():
    llm = ChatOpenAI(temperature=0)
    prompt = f"The company SOPs do not cover this. Please provide a general business best practice for: {query}"
    best_practice_answer = llm.invoke(prompt)
    print(f"Not found in SOP. General business best practice:\n{best_practice_answer.content}")
else:
    print(f"Answer from company SOPs:\n{sop_answer}")


