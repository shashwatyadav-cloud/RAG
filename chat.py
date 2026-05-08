from dotenv import load_dotenv
import os

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from langchain.tools import tool
from langchain.agents import create_agent

load_dotenv()



embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})



llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="openai/gpt-4o-mini"
)


@tool
def pdf_search(query: str):
    """
    Use this tool FIRST for questions about programming, Node.js, backend, or internal documents.

    If relevant content is found, return the answer.
    If no useful content is found, return exactly: NO_ANSWER_FOUND
    """

    docs = retriever.invoke(query)

    if not docs:
        return "NO_ANSWER_FOUND"

    context = "\n\n".join([doc.page_content for doc in docs])

    if len(context.strip()) < 50:
        return "NO_ANSWER_FOUND"

    print("TOOL USED: PDF")

    prompt = f"""
    Answer ONLY using the context below.
    If answer is not clearly present, say: I don't know.

    Context:
    {context}

    Question:
    {query}
    """

    return llm.invoke(prompt).content



@tool
def web_search(query: str):
    """
    Use this tool ONLY if pdf_search returns NO_ANSWER_FOUND
    or if the question requires latest/current information.
    """

    print("TOOL USED: WEB")

    web_tool = TavilySearch(max_results=3)
    results = web_tool.invoke({"query": query})

    if isinstance(results, list):
        texts = []
        for r in results:
            if isinstance(r, dict):
                texts.append(r.get("content", ""))
            else:
                texts.append(str(r))
        context = "\n\n".join(texts)
    else:
        context = str(results)

    prompt = f"""
    Answer clearly using this information:

    {context}

    Question:
    {query}
    """

    return llm.invoke(prompt).content



agent = create_agent(
    model=llm,
    tools=[pdf_search, web_search],
    system_prompt="""
    You are a smart assistant.

    You must follow these strict rules:

    1. Always try pdf_search FIRST for programming or internal knowledge questions.
    2. If pdf_search returns NO_ANSWER_FOUND, then use web_search.
    3. NEVER use both tools for the same question.
    4. Use ONLY ONE tool per question.
    5. Do NOT guess — rely on tools.

    Return the final answer clearly.
    """
)



while True:
    question = input("\nAsk your question (type 'exit' to quit): ")

    if question.lower() == "exit":
        break

    response = agent.invoke({
        "messages": [{"role": "user", "content": question}]
    })

    print("\nFinal Answer:\n")
    print(response["messages"][-1].content)