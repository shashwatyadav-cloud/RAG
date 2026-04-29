from dotenv import load_dotenv
import os

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, END

load_dotenv()


loader = PyMuPDFLoader("nodeJs.pdf")
documents = loader.load()


splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = splitter.split_documents(documents)


embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory="./vectordbsy"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="openai/gpt-4o-mini"
)

def classify_query(state):
    question = state["question"].lower()

    if "compare" in question or "vs" in question:
        state["query_type"] = "comparison"

    elif "what" in question or "how" in question:
        state["query_type"] = "factual"

    else:
        state["query_type"] = "out_of_scope"

    return state

def factual_node(state):
    question = state["question"]

    retrieved_docs = retriever.invoke(question)

    context = ""
    for doc in retrieved_docs:
        context += doc.page_content + "\n\n"

    prompt = f"""
    Answer only from the context below.
    If answer is not in context, say I don't know.

    Context:
    {context}

    Question:
    {question}
    """

    response = llm.invoke(prompt)

    state["answer"] = response.content
    return state

def comparison_node(state):
    question = state["question"]

    retrieved_docs = retriever.invoke(question)

    context = ""
    for doc in retrieved_docs:
        context += doc.page_content + "\n\n"

    prompt = f"""
    Compare the items in the question using only the context below.

    Context:
    {context}

    Question:
    {question}
    """

    response = llm.invoke(prompt)

    state["answer"] = response.content
    return state


def out_of_scope_node(state):
    state["answer"] = (
        "Sorry, this question is outside my document knowledge base."
    )
    return state


def validate_node(state):
    answer = state["answer"]

    if not answer.strip():
        state["answer"] = "I could not generate a reliable answer."

    return state

def route_query(state):
    return state["query_type"]


graph = StateGraph(dict)

graph.add_node("classify", classify_query)
graph.add_node("factual", factual_node)
graph.add_node("comparison", comparison_node)
graph.add_node("refusal", out_of_scope_node)
graph.add_node("validate", validate_node)

graph.set_entry_point("classify")

graph.add_conditional_edges(
    "classify",
    route_query,
    {
        "factual": "factual",
        "comparison": "comparison",
        "out_of_scope": "refusal"
    }
)

graph.add_edge("factual", "validate")
graph.add_edge("comparison", "validate")
graph.add_edge("refusal", "validate")

graph.add_edge("validate", END)

app = graph.compile()

question = input("Ask your question: ")

result = app.invoke({
    "question": question
})

print("\nAnswer:\n")
print(result["answer"])
