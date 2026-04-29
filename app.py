import fitz
from sentence_transformers import SentenceTransformer
import chromadb 
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client_openrouter = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key = os.getenv("OPENROUTER_API_KEY")
)

def extract_pdf(path):
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text+=page.get_text()

    return text


def split_text(text,chunk_size=500):
    chunks = []

    for i in range(0,len(text),chunk_size):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)
    
    return chunks 

model = SentenceTransformer("all-MiniLM-L6-v2")

def create_embeddings(chunks):
    embeddings = model.encode(chunks)
    return embeddings


client = chromadb.PersistentClient(path="./vectordb")

collection = client.get_or_create_collection(
    name="pdf_data"
)

def search_chunks(question):
    question_embedding = model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=3
    )

    return results["documents"][0]

def store_in_vector_db(chunks, embeddings):
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        collection.add(
            ids=[str(i)],
            documents=[chunk],
            embeddings=[embedding.tolist()]
        )
def generate_answer(question, chunks):
    context = "\n".join(chunks)

    prompt = f"""
    Answer the question only using the context below.
    if not in context say i dont know

    Context:
    {context}

    Question:
    {question}
    """

    response = client_openrouter.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content

print("ChromaDB setup complete")

pdf_text = extract_pdf("nodeJs.pdf")
chunks = split_text(pdf_text)

embeddings = create_embeddings(chunks)
store_in_vector_db(chunks, embeddings)
print("Stored in vector database successfully")

# print("Total chunks:", len(chunks))
# print("Embedding shape:", embeddings.shape)


question = input("Ask your question: ")

matched_chunks = search_chunks(question)


answer = generate_answer(question, matched_chunks)

print("\nAnswer:\n")
print(answer)