"""End-to-end RAG pipeline: question → retrieval → LLM answer with citations."""
from dataclasses import dataclass
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from src.utils.config import settings
from src.retrieval.vector_store import search


# LLM configuration
# gpt-4o-mini: cheap ($0.15/1M input, $0.60/1M output), good quality
# Temperature 0 = deterministic (same question → same answer)
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0

# Retrieval configuration
DEFAULT_N_RESULTS = 5  # How many chunks to retrieve per query


# System prompt — instructs the LLM how to behave
SYSTEM_PROMPT = """You are a financial research assistant specializing in SEC filings analysis for large US corporations.

You answer questions based ONLY on the context provided from SEC 10-K filings. You do not use your general knowledge about companies — only information from the retrieved filings.

CRITICAL RULES:
1. Answer based ONLY on the provided context
2. If the context doesn't contain the answer, say "I don't have enough information in the retrieved filings to answer this."
3. Always cite your sources using the format [ticker, filing_type, accession_number]
4. Quote directly from filings when the exact wording matters
5. Be specific with numbers, dates, and facts when they appear in context
6. If comparing companies, use evidence from each company's filings
7. Distinguish between what the filing explicitly states vs what you're inferring

Your tone is that of a professional financial analyst: precise, factual, and objective."""


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


@dataclass
class RAGResponse:
    """Result of a RAG query."""
    question: str
    answer: str
    retrieved_chunks: list[RetrievedChunk]
    
    def print_formatted(self) -> None:
        """Pretty-print the response."""
        print("\n" + "=" * 60)
        print(f"❓ Question: {self.question}")
        print("=" * 60)
        print(f"\n💡 Answer:\n{self.answer}\n")
        print(f"📚 Sources used ({len(self.retrieved_chunks)} chunks):")
        for i, chunk in enumerate(self.retrieved_chunks, 1):
            print(f"  {i}. {chunk.ticker} {chunk.filing_type} "
                  f"(relevance: {1 - chunk.distance:.2%})")


def get_llm() -> ChatOpenAI:
    """Create the LLM client."""
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        api_key=settings.openai_api_key,
    )


def retrieve_chunks(
    query: str,
    n_results: int = DEFAULT_N_RESULTS,
    filter_ticker: Optional[str] = None,
) -> list[RetrievedChunk]:
    """
    Retrieve relevant chunks for a query.
    
    Wraps the vector_store.search() function with structured output.
    """
    raw_results = search(
        query=query,
        n_results=n_results,
        filter_ticker=filter_ticker,
    )
    
    return [
        RetrievedChunk(
            text=r["text"],
            ticker=r["metadata"]["ticker"],
            filing_type=r["metadata"]["filing_type"],
            accession_number=r["metadata"]["accession_number"],
            distance=r["distance"],
        )
        for r in raw_results
    ]


def build_context_string(chunks: list[RetrievedChunk]) -> str:
    """Combine retrieved chunks into a single context string for the LLM."""
    if not chunks:
        return "[No relevant context retrieved]"
    
    context_parts = [chunk.to_context_string() for chunk in chunks]
    return "\n---\n".join(context_parts)


def answer_question(
    question: str,
    n_results: int = DEFAULT_N_RESULTS,
    filter_ticker: Optional[str] = None,
    verbose: bool = False,
) -> RAGResponse:
    """
    Answer a question using the RAG pipeline.
    
    Args:
        question: The user's question
        n_results: How many chunks to retrieve
        filter_ticker: Optional company filter (e.g. "MSFT")
        verbose: Print intermediate steps
    
    Returns:
        RAGResponse with answer and retrieved chunks
    """
    # Step 1: Retrieve relevant chunks
    if verbose:
        print(f"🔍 Retrieving chunks for: {question}")
    
    chunks = retrieve_chunks(
        query=question,
        n_results=n_results,
        filter_ticker=filter_ticker,
    )
    
    if verbose:
        print(f"   Found {len(chunks)} chunks")
        for i, chunk in enumerate(chunks[:3], 1):
            print(f"   {i}. {chunk.ticker} ({chunk.distance:.3f}): "
                  f"{chunk.text[:100]}...")
    
    # Step 2: Build context from chunks
    context = build_context_string(chunks)
    
    # Step 3: Build the user message
    user_message = f"""Context from SEC filings:

{context}

---

Question: {question}

Answer based only on the context above. Cite sources as [TICKER, FILING_TYPE, ACCESSION]."""
    
    # Step 4: Call the LLM
    if verbose:
        print(f"\n🤖 Calling {LLM_MODEL}...")
    
    llm = get_llm()
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ]
    
    response = llm.invoke(messages)
    answer = response.content
    
    if verbose:
        print(f"   Response length: {len(answer)} characters\n")
    
    return RAGResponse(
        question=question,
        answer=answer,
        retrieved_chunks=chunks,
    )


if __name__ == "__main__":
    # Test with three example questions
    test_questions = [
        "How does Microsoft describe its AI strategy?",
        "What are Apple's main sources of revenue?",
        "What risks does Meta identify related to artificial intelligence?",
    ]
    
    for question in test_questions:
        response = answer_question(question, verbose=True)
        response.print_formatted()
