from dotenv import load_dotenv
import os

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="openai/gpt-4o-mini"
)

loader = PyMuPDFLoader("nodeJs.pdf")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50
)

docs = splitter.split_documents(documents)

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory="./vectorsydb"
)

retriever = vectorstore.as_retriever(search_kwargs={"k":3})


question = input("Ask your question: ")

retrieved_docs = retriever.invoke(question)

context = ""
for doc in retrieved_docs:
    context += doc.page_content + "\n\n"


prompt = f"""
Answer the question only using the context below.
If answer is not in context, say I don't know.

Context:
{context}

Question:
{question}
"""

response = llm.invoke(prompt)

print("\nAnswer:\n")
print(response.content)
