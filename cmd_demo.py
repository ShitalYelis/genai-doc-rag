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
# Extract text from PDF, DOCX, TXT
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
# File picker (Windows dialog)
# --------------------------
print("📤 Select documents to process...")
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
# Split text into chunks
# --------------------------
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = []
for t in texts:
    chunks.extend(splitter.split_text(t))
print(f"\n📑 Created {len(chunks)} chunks")

# --------------------------
# Embeddings + Chroma Vector DB
# --------------------------
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma.from_texts(chunks, embedding_model, collection_name="genai_docs")

# --------------------------
# LLM (FLAN-T5-Large for better factual accuracy)
# --------------------------
print("\n🤖 Loading LLM (google/flan-t5-large)...")
hf_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-large",
    max_new_tokens=512,
    temperature=0.2,
)
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# --------------------------
# Custom, strong “context-only” prompt
# --------------------------
template = """
You are a precise assistant.
You will answer **only** based on the context provided from the uploaded documents.
If the answer is not present in the context, reply clearly with:
"I could not find this information in the provided document."

Context from the document:
{context}

Question:
{question}

Answer based only on the context:
"""
QA_PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
    chain_type="stuff",
    chain_type_kwargs={"prompt": QA_PROMPT}
)

# --------------------------
# Query Loop
# --------------------------
print("\n🎯 Ask questions about your uploaded documents. Type 'exit' to quit.\n")

while True:
    query = input("🔎 Your question: ")
    if query.lower() in ["exit", "quit"]:
        print("👋 Exiting.")
        break

    # Retrieve top docs for transparency
    docs = vectordb.similarity_search(query, k=2)
    print("\n📂 Top relevant document snippets:\n")
    for i, d in enumerate(docs, 1):
        print(f"[{i}] {d.page_content[:250]}...\n")

    # Modern LangChain call
    result = qa.invoke({"query": query})

    print("\n🤖 AI Answer:\n", result["result"])
    print("-" * 100)
