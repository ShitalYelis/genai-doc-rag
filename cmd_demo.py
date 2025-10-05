import os
import pdfplumber
from docx import Document
import tkinter as tk
from tkinter import filedialog
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

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

print("📤 Select documents to upload...")
root = tk.Tk()
root.withdraw()
file_paths = filedialog.askopenfilenames(
    title="Select documents",
    filetypes=[("Documents", "*.pdf *.docx *.txt")]
)

if not file_paths:
    print("❌ No files selected. Exiting.")
    exit()

texts = []
for file_path in file_paths:
    text = extract_text(file_path)
    if text.strip():
        texts.append(text)
        print(f"✅ Loaded {os.path.basename(file_path)}")
    else:
        print(f"⚠️ Skipped {os.path.basename(file_path)} (no readable text)")

if not texts:
    print("❌ No valid documents uploaded. Exiting.")
    exit()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = []
for t in texts:
    chunks.extend(splitter.split_text(t))
print(f"\n📑 Created {len(chunks)} chunks")

embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma.from_texts(chunks, embedding_model, collection_name="genai_docs")

print("\n🤖 Loading LLM (distilgpt2)...")
hf_pipeline = pipeline("text-generation", model="distilgpt2", max_new_tokens=200, temperature=0.7)
llm = HuggingFacePipeline(pipeline=hf_pipeline)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
    chain_type="stuff"
)

print("\n🎯 Ask questions about your documents. Type 'exit' to quit.\n")
while True:
    query = input("🔎 Your query: ")
    if query.lower() in ["exit", "quit"]:
        print("👋 Exiting.")
        break

    docs = vectordb.similarity_search(query, k=2)
    print("\n📂 Retrieved Contexts:")
    for i, d in enumerate(docs, 1):
        print(f"--- Chunk {i} ---\n{d.page_content[:200]}...\n")

    answer = qa.run(query)
    print("🤖 AI Answer:\n", answer)
    print("-" * 60)
