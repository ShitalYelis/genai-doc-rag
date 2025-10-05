import os
import streamlit as st
import pdfplumber
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

# --------------------------
# Helper: Extract text
# --------------------------
def extract_text(file_path: str) -> str:
    if file_path.endswith(".pdf"):
        with pdfplumber.open(file_path) as pdf:
            return "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
    elif file_path.endswith(".docx"):
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="GenAI Document Q&A", layout="wide")
st.title("📑 GenAI Document Q&A Demo")
st.write("Upload your documents and ask questions powered by RAG (LangChain + HuggingFace).")

uploaded_files = st.file_uploader("📤 Upload Documents", type=["pdf", "docx", "txt"], accept_multiple_files=True)

if uploaded_files:
    texts = []
    for file in uploaded_files:
        temp_path = os.path.join("temp", file.name)
        os.makedirs("temp", exist_ok=True)
        with open(temp_path, "wb") as f:
            f.write(file.getbuffer())
        text = extract_text(temp_path)
        if text.strip():
            texts.append(text)
            st.success(f"✅ Loaded {file.name}")
        else:
            st.warning(f"⚠️ Skipped {file.name} (no readable text)")

    if texts:
        # Step 1: Split text
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = []
        for t in texts:
            chunks.extend(splitter.split_text(t))
        st.info(f"📑 Created {len(chunks)} chunks")

        # Step 2: Embeddings + ChromaDB
        embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        vectordb = Chroma.from_texts(chunks, embedding_model, collection_name="genai_docs")

        # Step 3: Load LLM
        hf_pipeline = pipeline(
            "text-generation",
            model="distilgpt2",  # small model, runs on CPU
            max_new_tokens=200,
            temperature=0.7
        )
        llm = HuggingFacePipeline(pipeline=hf_pipeline)

        # Step 4: Setup QA
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
            chain_type="stuff"
        )

        # Step 5: Query input
        st.subheader("🔎 Ask a Question")
        query = st.text_input("Enter your query")
        if query:
            # Show retrieved docs
            docs = vectordb.similarity_search(query, k=2)
            st.write("📂 Retrieved Contexts:")
            for i, d in enumerate(docs, 1):
                st.markdown(f"**Chunk {i}:** {d.page_content[:300]}...")

            # Show AI Answer
            answer = qa.run(query)
            st.success(f"🤖 Answer: {answer}")
