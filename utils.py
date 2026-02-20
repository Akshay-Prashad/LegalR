import os
import time

import fitz  # PyMuPDF
import numpy as np
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv(".env")
API_KEY = os.getenv("PINECONE_API_KEY")  # Your Pinecone API key

print(API_KEY)


def readPdf(file_path: str) -> str:
    print("Reading Process Started")
    doc = fitz.open(file_path)
    text = " "
    for page in doc:
        text += page.get_text()
    doc.close()

    with open("text.txt", "w", encoding="utf-8") as text_file:
        text_file.write(text)
    print("PDF Read completed")
    return text


def textSplitter() -> list:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=250, chunk_overlap=50, length_function=len
    )
    with open("text.txt", "r", encoding="utf-8") as file:
        read_file = file.read()
        text_chunks = text_splitter.split_text(read_file)
    return text_chunks


def embedText(texts: list):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, convert_to_numpy=True)
    print(f"Embedding dimension: {len(embeddings[0])}")
    print(f"Total chunks embedded: {len(embeddings)}")
    print("Embedding Successful!")
    return embeddings


def vecDB(index_name: str = "constitution-index", dimension: int = 384):

    # Initialize Pinecone client
    pc = Pinecone(api_key=API_KEY)

    # Create index if it doesn't exist
    if index_name not in pc.list_indexes().names():
        print(f"Creating new index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
        )
        # Wait for index to be ready
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)
        print("Index created and ready!")
    else:
        print(f"Using existing index: {index_name}")

    # Connect to the index
    index = pc.Index(index_name)

    def add_documents(texts: list, embeddings: list, namespace: str = "default"):

        # Prepare vectors in Pinecone format: [(id, vector, metadata), ...]
        vectors = []
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            vectors.append(
                {
                    "id": f"chunk_{i}",
                    "values": embedding.tolist()
                    if isinstance(embedding, np.ndarray)
                    else embedding,
                    "metadata": {
                        "text": text,
                        "chunk_index": i,
                        "source": "constitution_of_india",
                    },
                }
            )

        # Upsert in batches of 100 (Pinecone recommendation)
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]
            index.upsert(vectors=batch, namespace=namespace)
            print(
                f"Upserted batch {i // batch_size + 1}/{(len(vectors) - 1) // batch_size + 1}"
            )

        print(
            f"Successfully added {len(texts)} documents to Pinecone index '{index_name}'"
        )

    def search(query_embedding: list, top_k: int = 5, namespace: str = "default"):

        results = index.query(
            vector=query_embedding.tolist()
            if isinstance(query_embedding, np.ndarray)
            else query_embedding,
            top_k=top_k,
            namespace=namespace,
            include_metadata=True,
        )
        return results

    def search_by_text(
        query_text: str, model, top_k: int = 5, namespace: str = "default"
    ):
        query_embedding = model.encode([query_text])[0]
        return search(query_embedding, top_k, namespace)

    def delete_all(namespace: str = "default"):

        index.delete(delete_all=True, namespace=namespace)
        print(f"Deleted all vectors in namespace '{namespace}'")

    def get_stats():
        stats = index.describe_index_stats()
        return {
            "total_vectors": stats.total_vector_count,
            "dimension": stats.dimension,
            "namespaces": list(stats.namespaces.keys()) if stats.namespaces else [],
        }

    def fetch_by_id(vector_id: str, namespace: str = "default"):
        """Fetch a specific vector by ID."""
        return index.fetch(ids=[vector_id], namespace=namespace)

    # Return all functions as dictionary
    return {
        "add_documents": add_documents,
        "search": search,
        "search_by_text": search_by_text,
        "delete_all": delete_all,
        "get_stats": get_stats,
        "fetch_by_id": fetch_by_id,
        "index": index,
        "client": pc,
    }


if __name__ == "__main__":
    # Step 1: Read PDF
    text = readPdf("Constitution_of_india.pdf")

    # Step 2: Split into chunks
    chunks = textSplitter()
    print(f"Total chunks created: {len(chunks)}")

    # Step 3: Initialize embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Step 4: Create embeddings
    embeddings = embedText(chunks)

    # Step 5: Initialize Pinecone Vector DB
    db = vecDB(
        index_name="indian-constitution",
        dimension=384,  # Matches all-MiniLM-L6-v2
    )

    # Check current stats
    print(f"\nIndex Stats: {db['get_stats']()}")

    # Step 6: Add documents to Pinecone
    db["add_documents"](chunks, embeddings, namespace="articles")

    # Step 7: Search

    print("SEARCHING: 'fundamental rights of citizens'")

    results = db["search_by_text"](
        "fundamental rights of citizens", model, top_k=3, namespace="articles"
    )

    for match in results["matches"]:
        print(f"\nScore: {match['score']:.4f}")
        print(f"ID: {match['id']}")
        # Text is stored in metadata
        text_preview = match["metadata"]["text"][:300]
        print(f"Text: {text_preview}...")
        print("-" * 50)

    # Final stats
    print(f"\nFinal Index Stats: {db['get_stats']()}")
