# 🧠 RAG Knowledge Assistant

A fully local, private Retrieval-Augmented Generation (RAG) system built with LangChain, ChromaDB, Mistral (via Ollama), and Streamlit. Upload your PDF documents and ask questions — no internet, no API cost.

---

## 📋 Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Requirements](#requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)
- [Configuration](#configuration)

---

## ✨ Features

- 100% local — no data leaves your machine
- Upload multiple PDF files as your knowledge base
- Semantic search using HuggingFace embeddings (`all-MiniLM-L6-v2`)
- Powered by Mistral 7B via Ollama
- Clean Streamlit chat interface
- Source document preview with page references

---

## 🛠️ Tech Stack

| Component | Tool |
|-----------|------|
| PDF Parsing | PyMuPDF |
| Text Splitting | LangChain Text Splitters |
| Embedding Model | HuggingFace `all-MiniLM-L6-v2` (CPU) |
| Vector Database | ChromaDB |
| LLM | Mistral 7B via Ollama |
| Framework | LangChain (LCEL) |
| UI | Streamlit |

---

## 💻 Requirements

- Windows 10/11
- Python 3.10 or higher
- 8 GB RAM (minimum)
- ~6 GB free disk space (for models)
- [Ollama](https://ollama.com/download) installed

---

## 🚀 Installation

### Step 1: Install Ollama

Download and install from [ollama.com/download](https://ollama.com/download), then pull the required models:

```bash
ollama pull mistral
ollama pull nomic-embed-text
```

### Step 2: Clone / Create Project Folder

```bash
mkdir C:\rag-assistant
cd C:\rag-assistant
mkdir data vectorstore src
type nul > src\__init__.py
```

### Step 3: Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

### Step 4: Install Dependencies

```bash
pip install langchain langchain-community langchain-ollama langchain-core
pip install langchain-text-splitters chromadb pymupdf streamlit
pip install sentence-transformers
```

---

## 📁 Project Structure

```
rag-assistant/
├── data/                    ← Put your PDF files here
├── vectorstore/             ← Auto-generated ChromaDB storage
├── src/
│   ├── __init__.py
│   ├── ingest.py            ← PDF loading and indexing
│   └── chain.py             ← RAG chain setup
├── app.py                   ← Streamlit UI
├── requirements.txt
└── README.md
```

---

## 📖 Usage

### 1. Add PDFs

Copy your PDF files into the `data/` folder.

### 2. Index the PDFs

```bash
venv\Scripts\activate
python src/ingest.py
```

You should see output like:
```
✅ Loaded: your_document.pdf
📦 Total chunks: 42
💾 Vector store saved!
```

### 3. Run the App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` in your browser.

### 4. Ask Questions

- Use the sidebar to upload new PDFs and click **"Index PDFs"**
- Type your question in the chat input
- Expand **"📄 Sources"** to see which document chunks were used

---

## ⚙️ Configuration

### Change the LLM Model

In `src/chain.py`, update the model name:

```python
llm = OllamaLLM(
    model="mistral",      # Change to: llama3.2:3b, llama3.1, etc.
    temperature=0,
    num_predict=256
)
```

### Adjust Chunk Size

In `src/ingest.py`, tune chunking for your documents:

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,       # Smaller = more precise retrieval
    chunk_overlap=30,     # Overlap prevents losing context at boundaries
)
```

### Retrieve More/Fewer Chunks

In `src/chain.py`:

```python
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 6}    # Increase for more context, decrease for speed
)
```

---

## 🔧 Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: langchain.text_splitter` | Old LangChain import path | Use `langchain_text_splitters` |
| `ModuleNotFoundError: langchain.chains` | Deprecated module | Use `langchain_core.runnables` |
| `ModuleNotFoundError: langchain.prompts` | Old import path | Use `langchain_core.prompts` |
| `memory layout cannot be allocated` | Not enough RAM for two Ollama models at once | Use HuggingFace CPU embeddings instead of `nomic-embed-text` |
| `model requires more system memory` | LLM too large for available RAM | Use `llama3.2:3b` instead of `llama3.1` |
| `venv\Scripts\activate` fails in PowerShell | Execution policy restriction | Run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| Sources empty / "I don't know" answers | PDF not indexed or chunks too large | Delete `vectorstore/`, reduce `chunk_size` to 200, re-run `ingest.py` |

---

## 📦 requirements.txt

```
langchain
langchain-community
langchain-ollama
langchain-core
langchain-text-splitters
chromadb
pymupdf
streamlit
sentence-transformers
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 🧠 How It Works

```
PDF Files
   ↓ PyMuPDF loads text
Text Chunks (chunk_size=200)
   ↓ HuggingFace all-MiniLM-L6-v2 embeds
ChromaDB Vector Store
   ↓ Similarity search (top-k chunks)
Retrieved Context
   ↓ Passed to Mistral with prompt
Final Answer
```

1. **Ingestion** — PDFs are loaded, split into small chunks, embedded into vectors, and stored in ChromaDB
2. **Retrieval** — User query is embedded, top-K most similar chunks are fetched from ChromaDB
3. **Generation** — Retrieved chunks + user question are passed to Mistral, which generates a grounded answer

---

## 📄 License

This project is for personal/educational use. Models are subject to their respective licenses (Mistral, Meta LLaMA).