from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",  # মাত্র 80MB, CPU তে চলে
        model_kwargs={"device": "cpu"}
    )

def ingest_pdfs(pdf_dir="data/", persist_dir="vectorstore/"):
    documents = []

    for file in os.listdir(pdf_dir):
        if file.endswith(".pdf"):
            loader = PyMuPDFLoader(os.path.join(pdf_dir, file))
            documents.extend(loader.load())
            print(f"✅ Loaded: {file}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=30,
        separators=["\n\n", "\n", ".", "!", "?", ",", " "]
    )
    chunks = splitter.split_documents(documents)
    print(f"📦 Total chunks: {len(chunks)}")

    embeddings = get_embeddings()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    print("💾 Vector store saved!")
    return vectorstore

if __name__ == "__main__":
    ingest_pdfs()