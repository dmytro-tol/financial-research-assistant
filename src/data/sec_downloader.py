"""Download SEC filings for a given company."""
from pathlib import Path
from sec_edgar_downloader import Downloader
from src.utils.config import settings


# The 10 companies we're tracking
COMPANIES = {
    "MSFT": "Microsoft",
    "AAPL": "Apple",
    "GOOGL": "Alphabet",
    "NVDA": "Nvidia",
    "META": "Meta",
    "AMZN": "Amazon",
    "TSLA": "Tesla",
    "JPM": "JPMorgan Chase",
    "V": "Visa",
    "UNH": "UnitedHealth",
}


def get_downloader() -> Downloader:
    """Create a configured SEC downloader instance."""
    # SEC requires a User-Agent string identifying the requester
    # Format: "Company Name email@example.com"
    name_and_email = settings.sec_user_agent
    company_name, email = name_and_email.rsplit(" ", 1)
    
    return Downloader(
        company_name=company_name,
        email_address=email,
        download_folder=str(settings.data_raw_dir)
    )


def download_filing(
    ticker: str,
    filing_type: str = "10-K",
    limit: int = 1
) -> list[Path]:
    """
    Download filings for a specific company.
    
    Args:
        ticker: Company ticker (e.g. "MSFT")
        filing_type: "10-K" (annual), "10-Q" (quarterly), "8-K" (material events)
        limit: How many most recent filings to download
    
    Returns:
        List of paths to downloaded filing directories
    """
    if ticker not in COMPANIES:
        raise ValueError(f"Unknown ticker: {ticker}. Known: {list(COMPANIES.keys())}")
    
    downloader = get_downloader()
    
    print(f"📥 Downloading {limit} {filing_type} filing(s) for {ticker} ({COMPANIES[ticker]})...")
    
    downloader.get(filing_type, ticker, limit=limit)
    
    # Downloaded files are in: data/raw/sec-edgar-filings/{TICKER}/{FILING_TYPE}/
    ticker_dir = settings.data_raw_dir / "sec-edgar-filings" / ticker / filing_type
    
    if not ticker_dir.exists():
        print(f"⚠️  No filings found for {ticker}")
        return []
    
    # Each filing is in its own subdirectory
    filing_dirs = sorted(ticker_dir.iterdir(), reverse=True)
    print(f"✅ Downloaded {len(filing_dirs)} filing(s) to {ticker_dir}")
    
    return filing_dirs


def download_all_companies(
    filing_type: str = "10-K",
    limit: int = 3
) -> dict[str, list[Path]]:
    """
    Download filings for all tracked companies.
    
    Args:
        filing_type: "10-K", "10-Q", or "8-K"
        limit: How many most recent filings per company
    
    Returns:
        Dict mapping ticker to list of filing directory paths
    """
    results = {}
    
    for ticker in COMPANIES.keys():
        try:
            filings = download_filing(ticker, filing_type, limit)
            results[ticker] = filings
        except Exception as e:
            print(f"❌ Failed {ticker}: {e}")
            results[ticker] = []
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 DOWNLOAD SUMMARY")
    print("=" * 50)
    total = 0
    for ticker, filings in results.items():
        count = len(filings)
        total += count
        status = "✅" if count > 0 else "❌"
        print(f"{status} {ticker:6} {COMPANIES[ticker]:20} → {count} filings")
    print(f"\nTotal: {total} filings downloaded")
    
    return results


if __name__ == "__main__":
    # Download last 3 years of 10-Ks for all 10 companies
    results = download_all_companies(filing_type="10-K", limit=3)
