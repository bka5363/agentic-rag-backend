"""
One-off ingestion script.

Run this ONCE, locally (not on Render), to build/rebuild the local
qdrant_db/ folder that app.py reads from at runtime.

Usage:
    pip install -r requirements.txt pypdf
    python ingest.py path/to/your.pdf

After it finishes, commit the updated qdrant_db/ folder and push:
    git add qdrant_db
    git commit -m "Rebuild vector store with minilm embeddings"
    git push
"""

import sys
from pathlib import Path

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

COLLECTION_NAME = "agentic_rag_minilm"
EMBED_DIM = 384  # all-MiniLM-L6-v2 output size
QDRANT_PATH = "./qdrant_db"


def extract_pdf_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return "\n\n".join(pages)


def main():
    if len(sys.argv) < 2:
        print("Usage: python ingest.py path/to/document.pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]
    if not Path(pdf_path).exists():
        print(f"File not found: {pdf_path}")
        sys.exit(1)

    print(f"Extracting text from {pdf_path} ...")
    raw_text = extract_pdf_text(pdf_path)
    print(f"Extracted {len(raw_text):,} characters.")

    print("Chunking...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(raw_text)
    # Drop near-empty chunks (page breaks, stray whitespace)
    chunks = [c.strip() for c in chunks if len(c.strip()) > 40]
    print(f"Produced {len(chunks)} chunks.")

    documents = [
        Document(page_content=chunk, metadata={"source": Path(pdf_path).name, "chunk_id": i})
        for i, chunk in enumerate(chunks)
    ]

    print("Loading embedding model (sentence-transformers/all-MiniLM-L6-v2, via GCS, no HF needed)...")
    embeddings = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print(f"Opening local Qdrant store at {QDRANT_PATH} ...")
    client = QdrantClient(path=QDRANT_PATH)

    if not client.collection_exists(COLLECTION_NAME):
        print(f"Creating collection '{COLLECTION_NAME}' ({EMBED_DIM}-dim, cosine)...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists, adding to it.")

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )

    print(f"Embedding and upserting {len(documents)} chunks (this may take a minute)...")
    batch_size = 64
    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        vectorstore.add_documents(batch)
        print(f"  {min(i + batch_size, len(documents))}/{len(documents)}")

    print("Done. qdrant_db/ is ready to commit.")


if __name__ == "__main__":
    main()
