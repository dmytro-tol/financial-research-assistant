"""Remove near-duplicate chunks from retrieval results."""
from src.retrieval.models import RetrievedChunk


# Threshold for considering chunks duplicates
# Higher = stricter (only exact matches)
# Lower = looser (more aggressive deduplication)
SIMILARITY_THRESHOLD = 0.8


def text_similarity(a: str, b: str) -> float:
    """
    Quick similarity estimate using character n-gram overlap.
    Returns 0-1 where 1 = identical.
    
    This is fast and good enough for deduplication.
    """
    # Normalize
    a = a.lower().strip()
    b = b.lower().strip()
    
    if a == b:
        return 1.0
    
    if len(a) == 0 or len(b) == 0:
        return 0.0
    
    # Use 3-character n-grams
    def ngrams(text: str, n: int = 3) -> set:
        return {text[i:i+n] for i in range(len(text) - n + 1)}
    
    ngrams_a = ngrams(a)
    ngrams_b = ngrams(b)
    
    if not ngrams_a or not ngrams_b:
        return 0.0
    
    intersection = len(ngrams_a & ngrams_b)
    union = len(ngrams_a | ngrams_b)
    
    return intersection / union


def deduplicate_chunks(
    chunks: list[RetrievedChunk],
    threshold: float = SIMILARITY_THRESHOLD,
) -> list[RetrievedChunk]:
    """
    Remove near-duplicate chunks from retrieval results.
    
    Keeps the chunk with the best (lowest) distance score when duplicates found.
    Preserves original order for non-duplicates.
    """
    if not chunks:
        return []
    
    # Sort by relevance (best first) so we keep the best version
    sorted_chunks = sorted(chunks, key=lambda c: c.distance)
    
    kept = []
    for chunk in sorted_chunks:
        # Check if this chunk is too similar to any we've kept
        is_duplicate = False
        for kept_chunk in kept:
            similarity = text_similarity(chunk.text, kept_chunk.text)
            if similarity >= threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            kept.append(chunk)
    
    return kept


if __name__ == "__main__":
    # Quick test
    from src.agent.rag_pipeline import retrieve_chunks
    
    chunks = retrieve_chunks(
        "What are AI risks?", 
        n_results=10,
    )
    
    print(f"Before dedup: {len(chunks)} chunks")
    for i, c in enumerate(chunks, 1):
        print(f"  {i}. {c.ticker}: {c.text[:80]}...")
    
    deduped = deduplicate_chunks(chunks)
    print(f"\nAfter dedup: {len(deduped)} chunks")
    for i, c in enumerate(deduped, 1):
        print(f"  {i}. {c.ticker}: {c.text[:80]}...")
