import streamlit as st
from src.ingest import ingest_pdfs
from src.chain import load_rag_chain
import os

st.set_page_config(page_title="RAG Knowledge Assistant", page_icon="🧠")
st.title("🧠 RAG Knowledge Assistant")
st.caption("Powered by LLaMA + ChromaDB — 100% Local")

with st.sidebar:
    st.header("📚 Knowledge Base")
    uploaded_files = st.file_uploader(
        "PDF আপলোড করো",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files:
        os.makedirs("data", exist_ok=True)
        for f in uploaded_files:
            with open(f"data/{f.name}", "wb") as fp:
                fp.write(f.read())

        if st.button("🔄 Index PDFs"):
            with st.spinner("Indexing... একটু অপেক্ষা করো"):
                ingest_pdfs()
            st.success("✅ Done! এখন প্রশ্ন করো।")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if query := st.chat_input("তোমার প্রশ্ন লেখো..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("চিন্তা করছি..."):
            chain, retriever = load_rag_chain()
            source_docs = retriever.invoke(query)
            answer = chain.invoke(query)

        st.write(answer)

        with st.expander("📄 Sources"):
            for doc in source_docs:
                st.caption(f"**{doc.metadata.get('source', 'Unknown')}** — Page {doc.metadata.get('page', '?')}")
                st.text(doc.page_content[:200] + "...")

        st.session_state.messages.append({"role": "assistant", "content": answer})