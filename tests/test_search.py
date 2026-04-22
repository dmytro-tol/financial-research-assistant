"""Quick interactive search tester."""
from src.retrieval.vector_store import search


def test_query(query: str, filter_ticker: str = None, n: int = 3):
    """Run a query and print formatted results."""
    print(f"\n{'=' * 60}")
    print(f"🔍 Query: {query}")
    if filter_ticker:
        print(f"   Filter: {filter_ticker} only")
    print('=' * 60)
    
    results = search(query, n_results=n, filter_ticker=filter_ticker)
    
    for i, result in enumerate(results, 1):
        meta = result['metadata']
        print(f"\n--- Result {i} | {meta['ticker']} {meta['filing_type']} "
              f"(distance: {result['distance']:.3f}) ---")
        print(result['text'][:400] + "...")


if __name__ == "__main__":
    # Try different queries
    test_query("How does Microsoft make money from cloud services?")
    
    test_query("What are Apple's main sources of revenue?", filter_ticker="AAPL")
    
    test_query("Climate change and environmental risks")
    
    test_query("Competitive threats from other cloud providers")
    
    test_query("Share buyback and dividend programs", filter_ticker="MSFT")
