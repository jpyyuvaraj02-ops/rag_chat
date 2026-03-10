import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from groq import Groq

# Load environment variables
load_dotenv()

# Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Streamlit page setup
st.set_page_config(page_title="Campus AI", page_icon="🤖")

st.title("🎓 Campus Assistant Chatbot")

# -----------------------------
# Load and cache vector database
# -----------------------------
@st.cache_resource
def load_vectorstore():

    loader = TextLoader("data.txt")
    documents = loader.load()

    splitter = CharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings()

    db = FAISS.from_documents(docs, embeddings)

    return db.as_retriever()

retriever = load_vectorstore()

# -----------------------------
# Chat history
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show previous chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# -----------------------------
# User input
# -----------------------------
prompt = st.chat_input("Ask something about campus")

if prompt:

    # Show user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.write(prompt)

    # -----------------------------
    # Retrieve context (RAG)
    # -----------------------------
    docs = retriever.invoke(prompt)

    context = "\n\n".join([doc.page_content for doc in docs])

    full_prompt = f"""
You are a helpful campus assistant.

Answer only using the provided context.

Context:
{context}

Question:
{prompt}

Answer clearly.
"""

    # -----------------------------
    # LLM call
    # -----------------------------
    chat = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": full_prompt}]
    )

    answer = chat.choices[0].message.content

    # Store answer
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    # Show assistant message
    with st.chat_message("assistant"):
        st.write(answer)