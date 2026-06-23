import glob
import os
import re
from pathlib import Path
from collections import Counter

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(override=True)

MODEL = "gpt-4.1-nano"
llm = ChatOpenAI(temperature=0.3, model_name=MODEL)

KNOWLEDGE_BASE = Path(__file__).parent.parent / "documents"
TOP_K = 5


def _load_documents() -> list[Document]:
    """Load all text documents from the knowledge base."""
    documents: list[Document] = []
    
    # Load all .txt files from the documents folder
    for file_path in Path(KNOWLEDGE_BASE).glob("**/*.txt"):
        try:
            text = file_path.read_text(encoding="utf-8")
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": str(file_path),
                        "filename": file_path.name,
                        "doc_type": "sneaker",
                    },
                )
            )
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return documents


def _chunk_documents(documents: list[Document]) -> list[Document]:
    """Split documents into smaller chunks for better retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def _tokenize(text: str) -> list[str]:
    """Tokenize text for keyword matching."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _score_document(query_tokens: Counter[str], document: Document) -> tuple[int, int]:
    """Score document based on token overlap with query."""
    doc_tokens = Counter(_tokenize(document.page_content))
    shared_terms = sum(min(query_tokens[token], doc_tokens[token]) for token in query_tokens)
    return shared_terms, len(document.page_content)


# Load and chunk documents
DOCUMENTS = _chunk_documents(_load_documents())


SYSTEM_PROMPT = """
You are a knowledgeable, enthusiastic, and friendly sneaker assistant. Your goal is to help people learn about sneaker culture, history, and brands.

Key principles:
- Be informative and engaging
- Provide accurate brand and model information
- Share interesting facts about sneaker history
- Be helpful for both casual fans and serious collectors
- Acknowledge the cultural significance of sneakers

Context:
{context}

When providing advice, always:
1. Acknowledge the person's question
2. Provide clear, factual information from the context
3. Use a friendly, enthusiastic tone
4. Be specific about brands, models, and history
"""


def resolve_query(query: str, history: list) -> str:
    """Resolve query with conversation history."""
    last_user = None
    if history:
        for turn in reversed(history):
            if turn["role"] == "user":
                last_user = turn["content"]
                break
    if last_user:
        return f"{last_user}\n{query}"
    return query


def fetch_context(query: str) -> list[Document]:
    """Retrieve relevant documents from the local knowledge base."""
    query_tokens = Counter(_tokenize(query))
    
    if not DOCUMENTS:
        return []
    
    # Score all documents and get top K
    scored_docs = sorted(
        DOCUMENTS,
        key=lambda document: _score_document(query_tokens, document),
        reverse=True,
    )
    
    return scored_docs[:TOP_K]


def answer_question(query: str, history=None):
    """Generate answer using retrieved context and conversation history."""
    resolved_query = resolve_query(query, history or [])
    docs = fetch_context(resolved_query)
    context = "\n\n".join(doc.page_content for doc in docs)

    messages = [SystemMessage(content=SYSTEM_PROMPT.format(context=context))]

    if history:
        for turn in history:
            if turn["role"] == "user":
                messages.append(HumanMessage(content=turn["content"]))
            elif turn["role"] == "assistant":
                messages.append(AIMessage(content=turn["content"]))

    messages.append(HumanMessage(content=query))
    response = llm.invoke(messages)
    return response.content, docs