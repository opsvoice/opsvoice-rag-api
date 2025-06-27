from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import os

# Set your OpenAI key
openai_api_key = os.environ.get("OPENAI_API_KEY")

# 1. Load your SOP Word doc
loader = UnstructuredWordDocumentLoader("sop-files/Part Time Assistant Manager Job Procedure.docx")
documents = loader.load()

# 2. Split SOP into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
docs = text_splitter.split_documents(documents)

# 3. Embed chunks into ChromaDB
embedding = OpenAIEmbeddings()
persist_directory = "./chroma_db"
vectorstore = Chroma.from_documents(docs, embedding=embedding, persist_directory=persist_directory)

# 4. Set up retriever
retriever = vectorstore.as_retriever()

# 5. Set up QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature=0),
    chain_type="stuff",
    retriever=retriever
)

# 6. Ask a question
query = "What are the duties listed in the assistant manager SOP?"
response = qa_chain.run(query)

print("Question:", query)
print("Answer:", response)

