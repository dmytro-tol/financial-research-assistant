"""Parse raw SEC filings into clean text."""
import re
from pathlib import Path
from typing import Optional
from bs4 import BeautifulSoup
from src.utils.config import settings


# Key sections we want to extract from 10-K filings
# These appear as section headers in the filing
TARGET_SECTIONS = {
    "business": ["item 1.", "item 1 ", "business"],
    "risk_factors": ["item 1a.", "item 1a ", "risk factors"],
    "mda": ["item 7.", "item 7 ", "management's discussion"],
    "financial_statements": ["item 8.", "item 8 ", "financial statements"],
}


def extract_filing_text(filing_path: Path) -> str:
    """
    Read a raw SEC filing and extract the main document text.
    
    SEC filings are multi-document SGML files containing the 10-K,
    exhibits, and other materials. We extract just the main 10-K.
    
    Args:
        filing_path: Path to filing directory or full-submission.txt
    
    Returns:
        Cleaned text content of the main document
    """
    # Accept either directory or direct file path
    if filing_path.is_dir():
        submission_file = filing_path / "full-submission.txt"
    else:
        submission_file = filing_path
    
    if not submission_file.exists():
        raise FileNotFoundError(f"No submission file at {submission_file}")
    
    # Read raw content
    with open(submission_file, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()
    
    # Split into individual documents (SEC filings contain multiple)
    # Each document is wrapped in <DOCUMENT>...</DOCUMENT>
    documents = re.findall(
        r"<DOCUMENT>(.+?)</DOCUMENT>",
        raw,
        re.DOTALL
    )
    
    if not documents:
        # Fallback: treat entire file as one document
        documents = [raw]
    
    # Find the main 10-K document (TYPE tag at the top)
    main_doc = None
    for doc in documents:
        # Check document type
        type_match = re.search(r"<TYPE>([^\n<]+)", doc)
        if type_match:
            doc_type = type_match.group(1).strip()
            if doc_type in ("10-K", "10-Q", "8-K"):
                main_doc = doc
                break
    
    if main_doc is None:
        # Use the longest document as fallback
        main_doc = max(documents, key=len)
    
    # Extract HTML content (strip SGML headers)
    html_match = re.search(r"<TEXT>(.+?)</TEXT>", main_doc, re.DOTALL)
    if html_match:
        html_content = html_match.group(1)
    else:
        html_content = main_doc
    
    # Parse HTML and extract text
    soup = BeautifulSoup(html_content, "lxml")
    
    # Remove script and style tags
    for tag in soup(["script", "style"]):
        tag.decompose()
    
    # Get clean text
    text = soup.get_text(separator="\n")
    
    # Clean up whitespace
    text = clean_text(text)
    
    return text


def clean_text(text: str) -> str:
    """Normalize whitespace and remove noise."""
    # Replace multiple newlines with single newline
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    # Replace multiple spaces with single space (within lines)
    text = re.sub(r" {2,}", " ", text)
    
    # Remove lines that are just special characters or very short
    lines = text.split("\n")
    cleaned_lines = [
        line.strip() for line in lines
        if len(line.strip()) > 2  # Keep lines with meaningful content
    ]
    
    return "\n".join(cleaned_lines)


def extract_sections(text: str) -> dict[str, str]:
    """
    Extract key sections from 10-K text.
    
    Returns dict with section_name -> section_text.
    Returns what it could find — not all sections always present.
    """
    sections = {}
    text_lower = text.lower()
    
    # Find positions of all section markers
    section_positions = []
    for section_key, markers in TARGET_SECTIONS.items():
        for marker in markers:
            idx = text_lower.find(marker)
            if idx > 0:
                section_positions.append((idx, section_key))
                break
    
    # Sort by position in document
    section_positions.sort()
    
    # Extract text between consecutive section markers
    for i, (start_idx, section_key) in enumerate(section_positions):
        if i + 1 < len(section_positions):
            end_idx = section_positions[i + 1][0]
        else:
            end_idx = len(text)
        
        section_text = text[start_idx:end_idx].strip()
        
        # Only include if section has meaningful content
        if len(section_text) > 500:
            sections[section_key] = section_text
    
    return sections


def parse_filing(filing_dir: Path) -> dict:
    """
    Parse a single filing directory into structured data.
    
    Returns dict with:
        - full_text: cleaned text of the entire document
        - sections: dict of extracted sections
        - metadata: ticker, filing_type, accession_number
    """
    # Parse metadata from path
    # Expected: data/raw/sec-edgar-filings/TICKER/FILING_TYPE/ACCESSION/
    parts = filing_dir.parts
    ticker = parts[-3]
    filing_type = parts[-2]
    accession = parts[-1]
    
    print(f"  📄 Parsing {ticker} {filing_type} {accession}...")
    
    full_text = extract_filing_text(filing_dir)
    sections = extract_sections(full_text)
    
    return {
        "ticker": ticker,
        "filing_type": filing_type,
        "accession_number": accession,
        "full_text": full_text,
        "sections": sections,
        "text_length": len(full_text),
        "sections_found": list(sections.keys()),
    }


def parse_all_filings() -> list[dict]:
    """Parse all downloaded filings and save cleaned text."""
    raw_dir = settings.data_raw_dir / "sec-edgar-filings"
    processed_dir = settings.data_processed_dir
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    if not raw_dir.exists():
        print("❌ No raw filings found. Run sec_downloader.py first.")
        return []
    
    results = []
    
    # Iterate: TICKER/FILING_TYPE/ACCESSION/
    for ticker_dir in sorted(raw_dir.iterdir()):
        if not ticker_dir.is_dir():
            continue
        
        for filing_type_dir in ticker_dir.iterdir():
            if not filing_type_dir.is_dir():
                continue
            
            for filing_dir in filing_type_dir.iterdir():
                if not filing_dir.is_dir():
                    continue
                
                try:
                    parsed = parse_filing(filing_dir)
                    
                    # Save processed text to disk
                    output_file = (
                        processed_dir /
                        f"{parsed['ticker']}_{parsed['filing_type']}_{parsed['accession_number']}.txt"
                    )
                    with open(output_file, "w", encoding="utf-8") as f:
                        f.write(parsed["full_text"])
                    
                    results.append(parsed)
                    
                except Exception as e:
                    print(f"  ❌ Failed {filing_dir}: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 PARSING SUMMARY")
    print("=" * 50)
    print(f"Total filings parsed: {len(results)}")
    
    if results:
        total_chars = sum(r["text_length"] for r in results)
        print(f"Total text extracted: {total_chars:,} characters")
        print(f"Average per filing:   {total_chars // len(results):,} characters")
        print(f"\nSections found across filings:")
        from collections import Counter
        section_counter = Counter()
        for r in results:
            section_counter.update(r["sections_found"])
        for section, count in section_counter.most_common():
            print(f"  {section:25} {count} filings")
    
    return results


if __name__ == "__main__":
    results = parse_all_filings()
