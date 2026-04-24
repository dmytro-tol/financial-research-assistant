"""Hybrid search combining semantic (ChromaDB) and keyword (BM25) retrieval."""
import re
from typing import Optional
from rank_bm25 import BM25Okapi
from src.retrieval.vector_store import search as semantic_search
from src.retrieval.models import RetrievedChunk
from src.retrieval.chunker import chunk_all_filings


# Cache the BM25 index in memory (building it takes a few seconds)
_bm25_cache = None
_chunks_cache = None


def tokenize(text: str) -> list[str]:
    """Simple tokenizer for BM25.
    
    Lowercase, split on non-word chars, remove very short tokens.
    """
    # Lowercase
    text = text.lower()
    # Split on non-word characters (keeps alphanumeric)
    tokens = re.findall(r'\w+', text)
    # Remove very short tokens (1-char words are usually noise)
    tokens = [t for t in tokens if len(t) > 1]
    return tokens


def get_bm25_index() -> tuple[BM25Okapi, list[dict]]:
    """Build or return cached BM25 index over all chunks."""
    global _bm25_cache, _chunks_cache
    
    if _bm25_cache is not None:
        return _bm25_cache, _chunks_cache
    
    print("🔨 Building BM25 index (one-time, ~10 seconds)...")
    
    # Load all chunks (same ones that went into ChromaDB)
    chunks = chunk_all_filings()
    
    # Tokenize each chunk's text
    tokenized_corpus = [tokenize(chunk["text"]) for chunk in chunks]
    
    # Build BM25 index
    bm25 = BM25Okapi(tokenized_corpus)
    
    # Cache for future calls
    _bm25_cache = bm25
    _chunks_cache = chunks
    
    print(f"✅ BM25 index built: {len(chunks)} chunks indexed")
    
    return bm25, chunks


def bm25_search(
    query: str,
    n_results: int = 5,
    filter_ticker: Optional[str] = None,
) -> list[RetrievedChunk]:
    """Keyword search using BM25."""
    bm25, all_chunks = get_bm25_index()
    
    # Tokenize query
    query_tokens = tokenize(query)
    
    # Get BM25 scores for all chunks
    scores = bm25.get_scores(query_tokens)
    
    # Filter by ticker if specified
    if filter_ticker:
        # Zero out scores for non-matching tickers
        for i, chunk in enumerate(all_chunks):
            if chunk["metadata"]["ticker"] != filter_ticker:
                scores[i] = 0
    
    # Get top N indices by score
    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )[:n_results]
    
    # Build result list
    results = []
    for idx in top_indices:
        if scores[idx] <= 0:  # Skip zero-score results
            continue
        chunk = all_chunks[idx]
        # Convert BM25 score to a pseudo-distance (lower = better, like ChromaDB)
        # Normalize: best score = 0 distance, worse = higher
        max_score = scores[top_indices[0]] if scores[top_indices[0]] > 0 else 1
        pseudo_distance = 1 - (scores[idx] / max_score)
        
        results.append(
            RetrievedChunk(
                text=chunk["text"],
                ticker=chunk["metadata"]["ticker"],
                filing_type=chunk["metadata"]["filing_type"],
                accession_number=chunk["metadata"]["accession_number"],
                distance=pseudo_distance,
            )
        )
    
    return results


def reciprocal_rank_fusion(
    semantic_results: list[RetrievedChunk],
    keyword_results: list[RetrievedChunk],
    k: int = 60,
    n_results: int = 5,
) -> list[RetrievedChunk]:
    """Combine two ranked lists using Reciprocal Rank Fusion.
    
    RRF is a simple but effective fusion algorithm:
    - For each item, sum 1/(rank + k) across all lists where it appears
    - Items appearing in both lists get boosted
    - k=60 is the standard constant (smooths out rank differences)
    """
    # Build a dict: chunk_id -> RRF score
    rrf_scores = {}
    chunk_lookup = {}
    
    for rank, chunk in enumerate(semantic_results):
        chunk_id = _chunk_id(chunk)
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (rank + 1 + k)
        chunk_lookup[chunk_id] = chunk
    
    for rank, chunk in enumerate(keyword_results):
        chunk_id = _chunk_id(chunk)
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (rank + 1 + k)
        if chunk_id not in chunk_lookup:
            chunk_lookup[chunk_id] = chunk
    
    # Sort by RRF score (descending)
    sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)
    
    # Build final list
    results = []
    for chunk_id in sorted_ids[:n_results]:
        chunk = chunk_lookup[chunk_id]
        # Update distance to reflect RRF score (lower = better)
        # Normalize: best RRF = 0 distance
        max_rrf = rrf_scores[sorted_ids[0]]
        chunk_copy = RetrievedChunk(
            text=chunk.text,
            ticker=chunk.ticker,
            filing_type=chunk.filing_type,
            accession_number=chunk.accession_number,
            distance=1 - (rrf_scores[chunk_id] / max_rrf),
        )
        results.append(chunk_copy)
    
    return results


def _chunk_id(chunk: RetrievedChunk) -> str:
    """Generate a unique ID for a chunk (for deduplication in RRF)."""
    return f"{chunk.ticker}_{chunk.accession_number}_{hash(chunk.text[:100])}"


def hybrid_search(
    query: str,
    n_results: int = 5,
    filter_ticker: Optional[str] = None,
    verbose: bool = False,
) -> list[RetrievedChunk]:
    """
    Hybrid search combining semantic (ChromaDB) and keyword (BM25).
    
    Returns the best results from both methods, fused with RRF.
    """
    # Get results from both methods
    # Fetch more than we need, so fusion has good candidates
    candidates_per_method = max(n_results * 2, 10)
    
    if verbose:
        print(f"  🧠 Semantic search for: {query}")
    
    # Semantic search via ChromaDB
    raw_semantic = semantic_search(
        query=query,
        n_results=candidates_per_method,
        filter_ticker=filter_ticker,
    )
    semantic_chunks = [
        RetrievedChunk(
            text=r["text"],
            ticker=r["metadata"]["ticker"],
            filing_type=r["metadata"]["filing_type"],
            accession_number=r["metadata"]["accession_number"],
            distance=r["distance"],
        )
        for r in raw_semantic
    ]
    
    if verbose:
        print(f"     → {len(semantic_chunks)} semantic results")
        print(f"  🔤 BM25 keyword search for: {query}")
    
    # Keyword search via BM25
    keyword_chunks = bm25_search(
        query=query,
        n_results=candidates_per_method,
        filter_ticker=filter_ticker,
    )
    
    if verbose:
        print(f"     → {len(keyword_chunks)} keyword results")
    
    # Fuse results
    fused = reciprocal_rank_fusion(
        semantic_results=semantic_chunks,
        keyword_results=keyword_chunks,
        n_results=n_results,
    )
    
    if verbose:
        print(f"  🔗 RRF fused to top {len(fused)} unique chunks")
    
    return fused


if __name__ == "__main__":
    # Benchmark: semantic-only vs keyword-only vs hybrid
    
    test_queries = [
        "What is Microsoft's AI strategy?",                  # Conceptual
        "dividend payment 2024",                              # Specific keyword
        "Microsoft Azure revenue",                            # Mixed
        "artificial intelligence risks",                      # Conceptual
        "total revenue fiscal 2024",                          # Exact numbers
    ]
    
    for query in test_queries:
        print("\n" + "=" * 70)
        print(f"🔍 Query: {query}")
        print("=" * 70)
        
        print("\n--- SEMANTIC ONLY ---")
        semantic_results = semantic_search(query, n_results=3)
        for i, r in enumerate(semantic_results, 1):
            print(f"{i}. {r['metadata']['ticker']}: {r['text'][:120]}...")
        
        print("\n--- KEYWORD ONLY (BM25) ---")
        keyword_results = bm25_search(query, n_results=3)
        for i, r in enumerate(keyword_results, 1):
            print(f"{i}. {r.ticker}: {r.text[:120]}...")
        
        print("\n--- HYBRID (RRF fusion) ---")
        hybrid_results = hybrid_search(query, n_results=3, verbose=True)
        for i, r in enumerate(hybrid_results, 1):
            print(f"{i}. {r.ticker}: {r.text[:120]}...")
