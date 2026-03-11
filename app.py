import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
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

st.title("🎓 XYZ Campus AI Assistant")
st.sidebar.title("About")
st.sidebar.write("Ask questions about campus facilities, events, and services.")
uploaded_file = st.file_uploader("Upload College PDF", type="pdf")

if uploaded_file:
    with open("college.pdf", "wb") as f:
        f.write(uploaded_file.read())
    st.success("PDF uploaded successfully")

st.sidebar.title("Example Questions")
st.sidebar.write("• Where is the library?")
st.sidebar.write("• When is the hackathon?")
st.sidebar.write("• What departments are available?")
st.sidebar.write("• Where is the placement cell?")
# -----------------------------
# Load and cache vector database
# -----------------------------
@st.cache_resource
def load_vectorstore():

    documents = []

    text_loader = TextLoader("data.txt")
    documents.extend(text_loader.load())

    if os.path.exists("college.pdf"):
        pdf_loader = PyPDFLoader("college.pdf")
        documents.extend(pdf_loader.load())

    splitter = CharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings()

    db = FAISS.from_documents(docs, embeddings)

    return db.as_retriever()
    

db = load_vectorstore()
retriever = db.as_retriever()

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

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.write(prompt)

    docs = retriever.get_relevant_documents(prompt)
    context = "\n\n".join([d.page_content for d in docs[:3]])

    full_prompt = f"""
You are a helpful campus assistant for XYZ Engineering College.

Context:
{context}

Question:
{prompt}

Give a short and clear answer.
"""

    messages = []

    for m in st.session_state.messages:
        messages.append({"role": m["role"], "content": m["content"]})

    messages.append({"role": "user", "content": full_prompt})

    chat = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages
    )

    answer = chat.choices[0].message.content

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    with st.chat_message("assistant"):
        st.write(answer)