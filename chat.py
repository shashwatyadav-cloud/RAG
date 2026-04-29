from dotenv import load_dotenv
import os

from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

load_dotenv()

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

client = QdrantClient(
    url="http://localhost:6333"
)

vectorstore = QdrantVectorStore(
    client=client,
    collection_name="pdf_docs",
    embedding=embedding_model
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="openai/gpt-4o-mini"
)

question = input("Ask your question: ")

retrieved_docs = retriever.invoke(question)


context = ""
for doc in retrieved_docs:
    context += doc.page_content + "\n\n"

prompt = f"""
Answer only from the context.
If answer is not in context say I don't know.

Context:
{context}

Question:
{question}
"""

response = llm.invoke(prompt)

print("\nAnswer:\n")
print(response.content)