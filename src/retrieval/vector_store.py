"""Embed chunks and store them in ChromaDB for semantic search."""
from pathlib import Path
from typing import Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_openai import OpenAIEmbeddings
from src.utils.config import settings
from src.retrieval.chunker import chunk_all_filings


# ChromaDB storage location
CHROMA_DIR = settings.project_root / "data" / "chroma_db"
COLLECTION_NAME = "sec_filings"

# OpenAI embedding model
# text-embedding-3-small: $0.02 per 1M tokens, 1536 dimensions
# Good balance of quality and cost
EMBEDDING_MODEL = "text-embedding-3-small"


#def get_embeddings_client() -> OpenAIEmbeddings:
#    """Create OpenAI embeddings client."""
#    return OpenAIEmbeddings(
#        model=EMBEDDING_MODEL,
#        api_key=settings.openai_api_key,
#    )

def get_embeddings_client():
    """Create embeddings client (Azure or OpenAI based on config)."""
    if settings.use_azure:
        from langchain_openai import AzureOpenAIEmbeddings
        return AzureOpenAIEmbeddings(
            azure_deployment=settings.azure_embedding_deployment,
            api_key=settings.azure_openai_api_key,
            azure_endpoint=settings.azure_openai_endpoint,
            api_version=settings.azure_openai_api_version,
        )
    else:
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            api_key=settings.openai_api_key,
        )

def get_chroma_client() -> chromadb.PersistentClient:
    """Create ChromaDB persistent client."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    
    return chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=ChromaSettings(
            anonymized_telemetry=False,  # Don't send usage data
        )
    )


def get_or_create_collection(client: chromadb.PersistentClient):
    """Get or create the SEC filings collection."""
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "SEC 10-K filings for top 10 mega-cap companies"}
    )


def embed_and_store_chunks(chunks: list[dict], batch_size: int = 100) -> None:
    """
    Embed all chunks and store in ChromaDB.
    
    Args:
        chunks: List of chunk dicts from chunker
        batch_size: How many chunks to embed at once (OpenAI API call)
    """
    print(f"🧮 Embedding {len(chunks):,} chunks...")
    print(f"   Model: {EMBEDDING_MODEL}")
    print(f"   Batch size: {batch_size}")
    
    # Initialize clients
    embeddings_client = get_embeddings_client()
    chroma_client = get_chroma_client()
    collection = get_or_create_collection(chroma_client)
    
    # Check if already embedded
    existing_count = collection.count()
    if existing_count > 0:
        print(f"⚠️  Collection already has {existing_count:,} documents.")
        response = input("Delete and re-embed? (y/N): ")
        if response.lower() == "y":
            chroma_client.delete_collection(COLLECTION_NAME)
            collection = get_or_create_collection(chroma_client)
            print("🗑️  Collection cleared.")
        else:
            print("✋ Skipping embedding. Using existing data.")
            return
    
    # Process in batches
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    
    for batch_num, i in enumerate(range(0, len(chunks), batch_size), 1):
        batch = chunks[i:i + batch_size]
        
        # Extract texts for this batch
        texts = [chunk["text"] for chunk in batch]
        ids = [chunk["chunk_id"] for chunk in batch]
        metadatas = [chunk["metadata"] for chunk in batch]
        
        # Generate embeddings via OpenAI API
        try:
            embeddings = embeddings_client.embed_documents(texts)
        except Exception as e:
            print(f"❌ Batch {batch_num} failed: {e}")
            continue
        
        # Store in ChromaDB
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        
        # Progress indicator
        print(f"  📦 Batch {batch_num}/{total_batches}: "
              f"{len(batch)} chunks embedded "
              f"(total stored: {collection.count():,})")
    
    # Final summary
    print("\n" + "=" * 50)
    print("📊 EMBEDDING SUMMARY")
    print("=" * 50)
    print(f"Total chunks embedded: {collection.count():,}")
    print(f"Storage location:      {CHROMA_DIR}")
    
    # Show per-company counts
    all_metadata = collection.get(include=["metadatas"])["metadatas"]
    from collections import Counter
    ticker_counts = Counter(m["ticker"] for m in all_metadata)
    print(f"\nEmbeddings per company:")
    for ticker, count in sorted(ticker_counts.items()):
        print(f"  {ticker:6} → {count:,} embeddings")


def search(
    query: str,
    n_results: int = 5,
    filter_ticker: Optional[str] = None
) -> list[dict]:
    """
    Search for relevant chunks given a query.
    
    Args:
        query: Natural language search query
        n_results: How many top results to return
        filter_ticker: Optional company ticker to restrict search to
    
    Returns:
        List of results with text, metadata, and distance score
    """
    embeddings_client = get_embeddings_client()
    chroma_client = get_chroma_client()
    collection = get_or_create_collection(chroma_client)
    
    # Embed the query
    query_embedding = embeddings_client.embed_query(query)
    
    # Build filter if specified
    where_filter = None
    if filter_ticker:
        where_filter = {"ticker": filter_ticker}
    
    # Search
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where_filter,
    )
    
    # Format results
    formatted = []
    for i in range(len(results["ids"][0])):
        formatted.append({
            "chunk_id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],  # Lower = more similar
        })
    
    return formatted


if __name__ == "__main__":
    # Step 1: Chunk all filings
    print("=" * 50)
    print("STEP 1: Chunking filings")
    print("=" * 50)
    chunks = chunk_all_filings()
    
    if not chunks:
        print("❌ No chunks produced. Aborting.")
        exit(1)
    
    # Step 2: Embed and store
    print("\n" + "=" * 50)
    print("STEP 2: Embedding and storing")
    print("=" * 50)
    embed_and_store_chunks(chunks)
    
    # Step 3: Test with a sample query
    print("\n" + "=" * 50)
    print("STEP 3: Testing with sample search")
    print("=" * 50)
    
    test_query = "What are the main risks related to artificial intelligence?"
    print(f"\n🔍 Query: {test_query}\n")
    
    results = search(test_query, n_results=3)
    
    for i, result in enumerate(results, 1):
        print(f"--- Result {i} ---")
        print(f"Company: {result['metadata']['ticker']}")
        print(f"Filing:  {result['metadata']['filing_type']}")
        print(f"Distance: {result['distance']:.4f}")
        print(f"Text preview: {result['text'][:300]}...")
        print()
