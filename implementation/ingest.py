import glob
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(override=True)

KNOWLEDGE_BASE = str(Path(__file__).parent.parent / "documents")
OUTPUT_FILE = Path(__file__).with_name("knowledge_base_cache.json")


def fetch_documents():
    """Load all markdown and text files from the knowledge base."""
    documents = []
    
    # Load both .md and .txt files
    for file_path in Path(KNOWLEDGE_BASE).glob("**/*.*"):
        if file_path.suffix.lower() in ['.md', '.txt']:
            try:
                text = file_path.read_text(encoding="utf-8")
                doc = Document(
                    page_content=text,
                    metadata={
                        "source": str(file_path),
                        "filename": file_path.name,
                        "doc_type": file_path.suffix.lower().replace('.', ''),
                    },
                )
                documents.append(doc)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    return documents


def create_chunks(documents):
    """Split documents into chunks for retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_documents(documents)


def embed_and_prepare(chunks):
    """Generate a lightweight cache for local retrieval."""
    return [
        {
            "id": f"chunk-{i}",
            "text": doc.page_content,
            "metadata": {
                "source": doc.metadata.get("source", ""),
                "doc_type": doc.metadata.get("doc_type", ""),
                "filename": os.path.basename(doc.metadata.get("source", "")),
            },
        }
        for i, doc in enumerate(chunks)
    ]


def write_cache(vectors):
    """Persist a cache of chunks for local startup."""
    OUTPUT_FILE.write_text(json.dumps(vectors, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    print("📚 Loading documents from knowledge base...")
    docs = fetch_documents()
    print(f"✅ Loaded {len(docs)} documents")
    
    for doc in docs:
        print(f"   - {doc.metadata.get('filename')} ({len(doc.page_content)} chars)")

    print("✂️ Creating chunks...")
    chunks = create_chunks(docs)
    print(f"✅ Created {len(chunks)} chunks")

    print("🧠 Preparing cache...")
    vectors = embed_and_prepare(chunks)
    print(f"✅ Prepared {len(vectors)} cached chunks")

    print(f"💾 Writing cache to {OUTPUT_FILE.name}...")
    write_cache(vectors)
    print(f"🚀 Ingestion complete: {len(vectors)} cached chunks written")