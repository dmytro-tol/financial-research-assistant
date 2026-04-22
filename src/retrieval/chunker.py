"""Split processed filings into chunks suitable for embedding."""
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.utils.config import settings


# Chunk configuration
# ~500 tokens ≈ 2000 characters ≈ 400 words
# Overlap helps preserve context at chunk boundaries
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200


def create_splitter() -> RecursiveCharacterTextSplitter:
    """Create the text splitter with our configuration."""
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Split on paragraphs first, then sentences, then words
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )


def chunk_filing(filing_file: Path) -> list[dict]:
    """
    Split a single processed filing into chunks with metadata.
    
    Args:
        filing_file: Path to processed text file
                     (e.g. data/processed/MSFT_10-K_0000950170-25-100235.txt)
    
    Returns:
        List of chunk dicts with text and metadata
    """
    # Parse metadata from filename: TICKER_FILINGTYPE_ACCESSION.txt
    filename = filing_file.stem  # "MSFT_10-K_0000950170-25-100235"
    parts = filename.split("_", 2)
    ticker, filing_type, accession = parts
    
    # Read content
    with open(filing_file, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Split into chunks
    splitter = create_splitter()
    chunks_text = splitter.split_text(text)
    
    # Create chunk dicts with metadata
    chunks = []
    for i, chunk_text in enumerate(chunks_text):
        chunks.append({
            "chunk_id": f"{ticker}_{filing_type}_{accession}_chunk{i:04d}",
            "text": chunk_text,
            "metadata": {
                "ticker": ticker,
                "filing_type": filing_type,
                "accession_number": accession,
                "chunk_index": i,
                "total_chunks_in_filing": len(chunks_text),
            }
        })
    
    return chunks


def chunk_all_filings() -> list[dict]:
    """Chunk all processed filings and return combined list."""
    processed_dir = settings.data_processed_dir
    
    if not processed_dir.exists():
        print("❌ No processed filings found. Run filing_parser.py first.")
        return []
    
    all_chunks = []
    files = sorted(processed_dir.glob("*.txt"))
    
    print(f"📑 Chunking {len(files)} processed filings...")
    
    for filing_file in files:
        chunks = chunk_filing(filing_file)
        all_chunks.extend(chunks)
        print(f"  ✂️  {filing_file.stem}: {len(chunks)} chunks")
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 CHUNKING SUMMARY")
    print("=" * 50)
    print(f"Filings processed:    {len(files)}")
    print(f"Total chunks created: {len(all_chunks):,}")
    
    if all_chunks:
        avg_length = sum(len(c["text"]) for c in all_chunks) // len(all_chunks)
        print(f"Average chunk length: {avg_length} characters")
        
        # Per-company breakdown
        from collections import Counter
        ticker_counts = Counter(c["metadata"]["ticker"] for c in all_chunks)
        print(f"\nChunks per company:")
        for ticker, count in sorted(ticker_counts.items()):
            print(f"  {ticker:6} → {count:,} chunks")
    
    return all_chunks


if __name__ == "__main__":
    chunks = chunk_all_filings()
    
    # Show sample chunk
    if chunks:
        print("\n" + "=" * 50)
        print("📄 SAMPLE CHUNK")
        print("=" * 50)
        sample = chunks[len(chunks) // 2]  # Middle chunk
        print(f"ID: {sample['chunk_id']}")
        print(f"Metadata: {sample['metadata']}")
        print(f"Text preview: {sample['text'][:300]}...")
