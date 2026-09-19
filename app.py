"""
app.py
------
Streamlit front-end for the University Knowledge Base RAG app.

- User enters their Groq API key in the sidebar.
- App loads the pre-built FAISS index from ./faiss_index/
- Uses Groq's openai/gpt-oss-120b model to answer questions grounded
  in the six university PDFs.
"""

import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
INDEX_DIR = "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "openai/gpt-oss-120b"

# ---------------------------------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------------------------------
st.set_page_config(page_title="University Knowledge Base", page_icon="🎓", layout="centered")

st.title("🎓 University Knowledge Base Assistant")
st.caption("Ask questions about student handbooks, fees, scholarships, exams, admissions, and the academic calendar.")

# ---------------------------------------------------------------------------
# SIDEBAR — API KEY
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    st.markdown("[Get a free Groq API key →](https://console.groq.com/keys)")
    st.divider()
    st.caption("Model: `openai/gpt-oss-120b`")
    st.caption("Embeddings: `all-MiniLM-L6-v2` (local)")
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------------------------
if not api_key:
    st.info("👈 Enter your Groq API key in the sidebar to begin.")
    st.stop()

if not os.path.isdir(INDEX_DIR):
    st.error(
        f"FAISS index not found at `./{INDEX_DIR}/`. "
        "Please run `python ingest.py` first to build it from the PDFs in `knowledge_base/`."
    )
    st.stop()

# ---------------------------------------------------------------------------
# LOAD MODELS (cached across reruns)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_pipeline(groq_api_key: str):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.load_local(
        INDEX_DIR, embeddings, allow_dangerous_deserialization=True
    )
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name=LLM_MODEL,
        temperature=0.2,
    )
    return vectorstore, llm


try:
    with st.spinner("Loading knowledge base and model..."):
        vectorstore, llm = load_pipeline(api_key)
except Exception as e:
    st.error(f"Failed to load models: {e}")
    st.stop()

# ---------------------------------------------------------------------------
# BUILD RAG CHAIN
# ---------------------------------------------------------------------------
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

system_prompt = (
    "You are a helpful assistant for a university's student knowledge base. "
    "Answer the user's question using ONLY the retrieved context below. "
    "If the answer is not in the context, say you don't know and suggest the "
    "user check the relevant policy document. "
    "When the context contains tables, preserve numeric values exactly. "
    "Keep answers clear and concise, and cite the source document name "
    "and section when helpful.\n\n"
    "Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

qa_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, qa_chain)

# ---------------------------------------------------------------------------
# CHAT UI
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
if user_query := st.chat_input("Ask a question about university policies..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Searching the knowledge base..."):
            try:
                result = rag_chain.invoke({"input": user_query})
                answer = result["answer"]

                st.markdown(answer)

                with st.expander("📄 View sources"):
                    seen = set()
                    for doc in result["context"]:
                        key = (doc.metadata.get("doc"), doc.metadata.get("section"), doc.metadata.get("page"))
                        if key in seen:
                            continue
                        seen.add(key)
                        st.markdown(
                            f"**{doc.metadata.get('doc', 'Unknown')}** — "
                            f"*{doc.metadata.get('section', 'N/A')}* "
                            f"(page {doc.metadata.get('page', '?')})"
                        )
                        st.caption(doc.page_content[:400] + ("..." if len(doc.page_content) > 400 else ""))

                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                err = f"⚠️ Error: {e}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})