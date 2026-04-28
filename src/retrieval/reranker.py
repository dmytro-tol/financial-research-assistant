"""Cross-encoder re-ranking for retrieval results.

Takes initial retrieval results (top 20) and re-ranks them
using a cross-encoder that scores each query-chunk pair
specifically against the query. Returns top N most relevant.
"""
from typing import Optional
from sentence_transformers import CrossEncoder
from src.retrieval.models import RetrievedChunk


# Cross-encoder model
# ms-marco-MiniLM-L-6-v2 is fast (90MB) and performs well on retrieval
# Trained specifically on MS MARCO dataset for query-document relevance
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Cache the model in memory after first load (model loading takes 5-10 sec)
_reranker_cache = None


def get_reranker() -> CrossEncoder:
    """Load the cross-encoder model (cached after first call)."""
    global _reranker_cache
    
    if _reranker_cache is not None:
        return _reranker_cache
    
    print(f"🔨 Loading cross-encoder model: {RERANKER_MODEL}")
    print("   First time: ~5-10 seconds + 90MB download")
    
    _reranker_cache = CrossEncoder(RERANKER_MODEL)
    
    print(f"✅ Cross-encoder loaded")
    return _reranker_cache


def rerank_chunks(
    query: str,
    chunks: list[RetrievedChunk],
    top_n: int = 5,
    verbose: bool = False,
) -> list[RetrievedChunk]:
    """
    Re-rank chunks using cross-encoder scoring.
    
    Args:
        query: The original user query
        chunks: Candidate chunks from initial retrieval (typically 10-20)
        top_n: How many top chunks to return after re-ranking
        verbose: Print scores for debugging
    
    Returns:
        Top N chunks re-ordered by cross-encoder relevance score
    """
    if not chunks:
        return []
    
    # If we have fewer candidates than requested, just return them all
    if len(chunks) <= top_n:
        return chunks
    
    reranker = get_reranker()
    
    # Build pairs: [query, chunk_text] for each chunk
    pairs = [(query, chunk.text) for chunk in chunks]
    
    # Score all pairs in one batch (efficient)
    if verbose:
        print(f"  🎯 Re-ranking {len(chunks)} chunks against query...")
    
    scores = reranker.predict(pairs)
    
    # Pair each chunk with its score
    chunk_scores = list(zip(chunks, scores))
    
    # Sort by score descending (higher = more relevant)
    chunk_scores.sort(key=lambda pair: pair[1], reverse=True)
    
    if verbose:
        print(f"  📊 Score range: {scores.min():.2f} to {scores.max():.2f}")
        print(f"  ✅ Top {top_n} after re-ranking:")
        for i, (chunk, score) in enumerate(chunk_scores[:top_n], 1):
            print(f"     {i}. {chunk.ticker} (score: {score:.2f})")
    
    # Update distance to reflect re-ranker score (for downstream display)
    # Higher cross-encoder score = lower "distance" (better)
    # Normalize scores to 0-1 range
    max_score = max(scores)
    min_score = min(scores)
    score_range = max_score - min_score if max_score > min_score else 1.0
    
    reranked = []
    for chunk, score in chunk_scores[:top_n]:
        # Convert score to distance (0 = best, higher = worse)
        normalized = (score - min_score) / score_range
        new_distance = 1.0 - normalized
        
        reranked_chunk = RetrievedChunk(
            text=chunk.text,
            ticker=chunk.ticker,
            filing_type=chunk.filing_type,
            accession_number=chunk.accession_number,
            distance=new_distance,
        )
        reranked.append(reranked_chunk)
    
    return reranked


if __name__ == "__main__":
    # Test re-ranking
    from src.retrieval.hybrid_search import hybrid_search
    
    query = "What is Microsoft's AI strategy?"
    
    print(f"\n🔍 Query: {query}\n")
    print("=" * 70)
    print("BEFORE RE-RANKING (hybrid search top 10):")
    print("=" * 70)
    
    initial_chunks = hybrid_search(query, n_results=10)
    for i, chunk in enumerate(initial_chunks, 1):
        print(f"{i:2}. {chunk.ticker}: {chunk.text[:120]}...")
    
    print("\n" + "=" * 70)
    print("AFTER RE-RANKING (top 5):")
    print("=" * 70)
    
    reranked = rerank_chunks(query, initial_chunks, top_n=5, verbose=True)
    for i, chunk in enumerate(reranked, 1):
        print(f"\n{i}. {chunk.ticker} (relevance: {1-chunk.distance:.1%})")
        print(f"   {chunk.text[:200]}...")
