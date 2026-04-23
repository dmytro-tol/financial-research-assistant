"""Shared data models for retrieval."""
from dataclasses import dataclass


@dataclass
class RetrievedChunk:
    """A single chunk retrieved from the vector store."""
    text: str
    ticker: str
    filing_type: str
    accession_number: str
    distance: float
    
    def to_context_string(self) -> str:
        """Format chunk for inclusion in LLM context."""
        return (
            f"[Source: {self.ticker} {self.filing_type} "
            f"(Accession: {self.accession_number})]\n"
            f"{self.text}\n"
        )
