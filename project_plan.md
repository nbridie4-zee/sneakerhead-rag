# 👟 Sneakerhead LLM RAG Project Plan

## 🎯 Objective
Build an AI-powered sneaker assistant using Retrieval-Augmented Generation (RAG) that answers questions about sneaker brands, models, and sneaker culture.

---

## 📊 Data Source
- Local dataset stored in `data.txt`
- Contains sneaker information such as Nike Air Jordan, Adidas Yeezy, Puma RS-X, and sneaker culture descriptions

---

## 🧠 Architecture (RAG Flow)

1. User asks a question in Streamlit
2. Data is loaded from `data.txt`
3. Text is split into chunks
4. Embeddings are created using HuggingFace models
5. FAISS stores vector embeddings
6. User query is converted into embeddings
7. Similar chunks are retrieved
8. HuggingFace LLM generates final answer
9. Response is displayed in Streamlit UI

---

## 🛠 Technologies Used

- Python
- Streamlit
- LangChain
- FAISS
- HuggingFace Hub
- Sentence Transformers

---

## 🚀 Future Improvements

- Add sneaker image database
- Use real sneaker API data
- Improve chatbot UI (ChatGPT style)
- Deploy publicly on Hugging Face Spaces