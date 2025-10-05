import os
import pdfplumber
from docx import Document
import tkinter as tk
from tkinter import filedialog
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

# --------------------------
# Extract text
# --------------------------
def extract_text(file_path: str) -> str:
    if file_path.endswith(".pdf"):
        with pdfplumber.open(file_path) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    elif file_path.endswith(".docx"):
        doc = Document(file_path)
        return "\n".join(para.text for para in doc.paragraphs)
    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""

# --------------------------
# File picker
# --------------------------
print("📤 Select your documents (PDF/DOCX/TXT)...")
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

# --------------------------
# Split into chunks
# --------------------------
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = []
for t in texts:
    chunks.extend(splitter.split_text(t))
print(f"\n📑 Created {len(chunks)} text chunks")

# --------------------------
# Embeddings + Chroma
# --------------------------
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma.from_texts(chunks, embedding_model, collection_name="genai_docs")

# --------------------------
# Load LLM (FLAN-T5 for QA)
# --------------------------
print("\n🤖 Loading LLM (google/flan-t5-base)...")
hf_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=256,
    temperature=0.3,
)
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# --------------------------
# Create custom prompt
# --------------------------
template = """
You are a helpful assistant. 
Use the following context from the document to answer the question accurately and concisely.

Context:
{context}

Question:
{question}

Answer:
"""
QA_PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
    chain_type="stuff",
    chain_type_kwargs={"prompt": QA_PROMPT}
)

# --------------------------
# Query loop
# --------------------------
print("\n🎯 Ask questions about your uploaded documents. Type 'exit' to quit.\n")
while True:
    query = input("🔎 Your question: ")
    if query.lower() in ["exit", "quit"]:
        print("👋 Exiting.")
        break

    answer = qa.run(query)
    print("\n🤖 AI Answer:\n", answer)
    print("-" * 80)
