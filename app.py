import streamlit as st
from dotenv import load_dotenv
import os

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

load_dotenv()

# -------------------------
# 🔹 PAGE CONFIG
# -------------------------
st.set_page_config(page_title="RAG Agent", layout="wide")

st.title("🧠 RAG Agent (PDF + Web)")
st.write("Ask questions from your PDF or the web")

# -------------------------
# 🔹 INIT MODELS (CACHE)
# -------------------------
@st.cache_resource
def load_system():
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

    return retriever, llm

retriever, llm = load_system()

# -------------------------
# 🔹 PDF SEARCH
# -------------------------
def pdf_search(query: str):
    docs = retriever.invoke(query)

    if not docs:
        return "NO_ANSWER_FOUND", "❌ PDF EMPTY"

    context = "\n\n".join([doc.page_content for doc in docs])

    if len(context.strip()) < 50:
        return "NO_ANSWER_FOUND", "❌ PDF TOO WEAK"

    prompt = f"""
    Answer only from the context.
    If answer not found, say I don't know.

    Context:
    {context}

    Question:
    {query}
    """

    answer = llm.invoke(prompt).content
    return answer, "📄 USED PDF"

# -------------------------
# 🔹 WEB SEARCH
# -------------------------
def web_search(query: str):
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

    Question: {query}
    """

    answer = llm.invoke(prompt).content
    return answer, "🌐 USED WEB"

# -------------------------
# 🔹 DECISION
# -------------------------
def decide_source(query: str):
    prompt = f"""
    Decide where to answer from:

    - PDF → for programming, Node.js, internal docs
    - WEB → for latest info, news

    Only return one word: PDF or WEB

    Question: {query}
    """

    decision = llm.invoke(prompt).content.strip().upper()
    return decision

# -------------------------
# 🔹 ROUTER
# -------------------------
def smart_router(query: str):
    decision = decide_source(query)

    if "PDF" in decision:
        answer, source = pdf_search(query)

        if answer != "NO_ANSWER_FOUND":
            return answer, decision, source

        # fallbackstreamlit run app.py
        answer, source = web_search(query)
        return answer, "PDF → WEB", source

    else:
        answer, source = web_search(query)
        return answer, decision, source

# -------------------------
# 🔹 UI INPUT
# -------------------------
query = st.text_input("Ask your question:")

if st.button("Ask"):
    if query.strip() == "":
        st.warning("Please enter a question")
    else:
        with st.spinner("Thinking..."):
            answer, decision, source = smart_router(query)

        # -------------------------
        # 🔹 OUTPUT
        # -------------------------
        st.subheader("Answer")
        st.write(answer)

        st.markdown("---")

        st.subheader("Debug Info")
        st.write(f"🧠 Decision: {decision}")
        st.write(f"{source}")