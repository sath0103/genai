
import os
import shutil
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
)
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "documents"
FAISS_DIR = BASE_DIR / "faiss_index"

load_dotenv(BASE_DIR / ".env")

st.set_page_config(
    page_title="Local RAG Chatbot",
    page_icon="💬",
    layout="wide",
)

st.title("💬 Knowledge Base Chatbot")
st.caption("LangChain • FAISS • OpenAI • LangSmith")

required = [
    "OPENAI_API_KEY",
    "LANGSMITH_API_KEY",
    "LANGSMITH_TRACING",
    "LANGSMITH_PROJECT",
]

missing = [key for key in required if not os.getenv(key)]

if missing:
    st.error(
        f"Missing settings in .env: {', '.join(missing)}"
    )
    st.stop()

if os.getenv("LANGSMITH_TRACING", "").lower() != "true":
    st.error("Set LANGSMITH_TRACING=true in your .env file.")
    st.stop()

DOCUMENTS_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# 2. MODELS
# --------------------------------------------------

@st.cache_resource
def get_models():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
    )

    return embeddings, llm


# --------------------------------------------------
# 3. LOAD DOCUMENTS
# --------------------------------------------------

def load_documents():
    documents = []

    for path in DOCUMENTS_DIR.iterdir():
        if not path.is_file():
            continue

        try:
            if path.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(path))

            elif path.suffix.lower() in [".txt", ".md"]:
                loader = TextLoader(
                    str(path),
                    encoding="utf-8",
                )

            else:
                continue

            loaded = loader.load()

            for doc in loaded:
                doc.metadata["source"] = path.name

            documents.extend(loaded)

        except Exception as exc:
            st.warning(f"Could not read {path.name}: {exc}")

    return documents


# --------------------------------------------------
# 4. LOCAL FAISS VECTOR DATABASE
# --------------------------------------------------

@st.cache_resource
def get_vector_store():
    embeddings, _ = get_models()

    # Reuse the saved local database
    if (FAISS_DIR / "index.faiss").exists():
        return FAISS.load_local(
            str(FAISS_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    # Build the database if it does not exist
    documents = load_documents()

    if not documents:
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        return None

    vector_store = FAISS.from_documents(
        chunks,
        embeddings,
    )

    vector_store.save_local(str(FAISS_DIR))

    return vector_store


# --------------------------------------------------
# 5. RAG PROMPT
# --------------------------------------------------

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful document question-answering assistant.

Use the retrieved context to answer the user's question.
Do not invent facts or claim that the documents say something
they do not say.

If the answer is not in the context, clearly say:
"I could not find the answer in the provided documents."

Context:
{context}"""
    ),
    (
        "human",
        "{question}",
    ),
])


# --------------------------------------------------
# 6. RETRIEVAL AND ANSWER GENERATION
# --------------------------------------------------

def ask_question(vector_store, question):
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 4}
    )

    docs = retriever.invoke(question)

    if not docs:
        return (
            "I could not find relevant information in the documents.",
            [],
            None,
        )

    context = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')}\n"
        f"{doc.page_content}"
        for doc in docs
    )

    messages = prompt.invoke({
        "context": context,
        "question": question,
    })

    _, llm = get_models()
    response = llm.invoke(messages)

    return response.content, docs, response.usage_metadata


# --------------------------------------------------
# 7. SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.header("⚙️ Settings")

    st.success("OpenAI configured")
    st.success("LangSmith tracing enabled")

    st.caption(
        f"Project: {os.getenv('LANGSMITH_PROJECT')}"
    )

    st.divider()

    st.subheader("Knowledge Base")
    st.write(f"Document folder: `{DOCUMENTS_DIR}`")
    st.write(f"Vector database: `{FAISS_DIR}`")

    if st.button("🗑️ Rebuild Vector Database", use_container_width=True):
        # Clear the cached index before rebuilding
        get_vector_store.clear()

        if FAISS_DIR.exists():
            shutil.rmtree(FAISS_DIR)

        st.session_state.messages = []

        st.rerun()

    st.caption(
        "Add PDF, TXT or Markdown files to documents/. "
        "Rebuild the database after adding or changing files."
    )

    if st.button("🧹 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# --------------------------------------------------
# 8. INITIALIZE CHAT
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.spinner("Loading local FAISS knowledge base..."):
    try:
        vector_store = get_vector_store()
    except Exception as exc:
        st.error(f"Could not load the vector database: {exc}")
        st.stop()

if vector_store is None:
    st.warning(
        "No documents found. Add a PDF, TXT or Markdown file "
        "to the documents folder, then rebuild the database."
    )
    st.stop()

st.info(
    "Ask a question about your documents. "
    "Answers are generated using retrieved document context."
)


# --------------------------------------------------
# 9. DISPLAY CHAT HISTORY
# --------------------------------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("📚 View retrieved sources"):
                for source in message["sources"]:
                    st.markdown(source)


# --------------------------------------------------
# 10. CHAT INPUT
# --------------------------------------------------

question = st.chat_input("Ask something about your documents...")

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):
            try:
                answer, docs, usage = ask_question(
                    vector_store,
                    question,
                )

                st.markdown(answer)

                sources = []

                for doc in docs:
                    source = doc.metadata.get("source", "Unknown")
                    page = doc.metadata.get("page")

                    if page is not None:
                        label = f"{source} — Page {page + 1}"
                    else:
                        label = source

                    if label not in sources:
                        sources.append(label)

                if sources:
                    with st.expander("📚 View retrieved sources"):
                        for source in sources:
                            st.markdown(f"- {source}")

                if usage:
                    st.caption(
                        f"Tokens — Input: {usage.get('input_tokens', 0)}, "
                        f"Output: {usage.get('output_tokens', 0)}"
                    )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })

            except Exception as exc:
                st.error(f"Error: {exc}")
