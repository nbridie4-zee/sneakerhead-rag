import streamlit as st
import os

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

from dotenv import load_dotenv
import os

load_dotenv()

HUGGINGFACEHUB_API_TOKEN=""

docs = []

for file in os.listdir("documents"):
    if file.endswith(".txt"):
        with open(f"documents/{file}", "r", encoding="utf-8") as f:
            docs.append(f.read())

splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
docs = splitter.split_text("\n".join(docs))

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.from_texts(docs, embeddings)
retriever = db.as_retriever(search_kwargs={"k": 3})

from langchain_huggingface import HuggingFaceEndpoint

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    huggingfacehub_api_token=api_key,
    temperature=0.2,
    max_new_tokens=256
)

prompt = PromptTemplate.from_template("""
Answer the question using ONLY the context.

If the answer is not in the context, say:
I don't know based on the data.

Context:
{context}

Question:
{question}

Answer:
""")

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
)

st.title("👟 Sneakerhead AI Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask something about sneakers...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    retrieved_docs = retriever.invoke(query)

    with st.expander("Retrieved Context"):
        st.write(format_docs(retrieved_docs))

    result = rag_chain.invoke(query)

    if isinstance(result, dict):
        response = result.get("result") or result.get("answer") or str(result)
    else:
        response = str(result)

    st.session_state.messages.append({"role": "assistant", "content": response})

    with st.chat_message("assistant"):
        st.markdown(response)