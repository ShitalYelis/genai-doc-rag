# 📑 GenAI Document Q&A (RAG Demo)

This project is a **Retrieval-Augmented Generation (RAG)** demo using:
- [LangChain](https://www.langchain.com/) (framework)
- [ChromaDB](https://www.trychroma.com/) (vector database)
- [SentenceTransformers](https://www.sbert.net/) (embeddings)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/index) (LLM)
- [Streamlit](https://streamlit.io/) (web UI)

It allows you to:
- 📤 Upload **PDF/DOCX/TXT** documents
- 📑 Split & store documents in a vector database
- 🔍 Search across documents
- 🤖 Ask **GenAI-powered queries** (RAG pipeline)

---

## ⚙️ Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/yourusername/genai-doc-rag.git
cd genai-doc-rag
python -m venv venv
venv\Scripts\activate   # (Windows)
# or
source venv/bin/activate  # (Linux/Mac)

pip install -r requirements.txt
