import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
import ollama
st.set_page_config(
    page_title="Mini RAG Q&A",
    page_icon="🤖"
)
st.title("🤖 Mini RAG Q&A")
st.write("Paste a document, store it in ChromaDB, and ask questions about it.")
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")
embedding_model = load_embedding_model()
client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="documents"
)
document = st.text_area(
    "📄 Paste your document here:",
    height=250
)
if st.button("📥 Store Document"):
    if document.strip() == "":
        st.warning("Please paste a document first.")
    else:
        words = document.split()
        chunk_size = 100
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        embeddings = embedding_model.encode(chunks).tolist()
        ids = [
            f"chunk_{i}"
            for i in range(len(chunks))
        ]
        collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings
        )
        st.success("✅ Document stored successfully!")
question = st.text_input(
    "❓ Ask a question about the document:"
)
if st.button("🔍 Ask"):

    if question.strip() == "":
        st.warning("Please enter a question.")
    else:
        question_embedding = embedding_model.encode(
            [question]
        ).tolist
        results = collection.query(
            query_embeddings=question_embedding,
            n_results=3
        )
        documents = results["documents"][0]
        context = "\n\n".join(documents)
        prompt = f"""
You are a helpful question-answering assistant.
Answer the question using ONLY the context provided below.
If the answer is not present in the context,
say: "I don't know based on the document."
Context:
{context}
Question:
{question}
Answer:
"""
        response = ollama.chat(
            model="llama3.2",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        st.subheader("🤖 Answer")
        st.write(response["message"]["content"])