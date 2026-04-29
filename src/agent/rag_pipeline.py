"""End-to-end RAG research agent with conversation memory and query decomposition."""
from dataclasses import dataclass, field
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from pydantic import BaseModel, Field
from src.utils.config import settings
from src.retrieval.vector_store import search
from src.retrieval.dedup import deduplicate_chunks
from src.retrieval.models import RetrievedChunk


# LLM configuration
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0
DEFAULT_N_RESULTS = 5
MAX_CONVERSATION_HISTORY = 10


# System prompts
ANSWER_SYSTEM_PROMPT = """You are a financial research assistant specializing in SEC filings analysis for large US corporations.

You answer questions based ONLY on the context provided from SEC 10-K filings. You do not use your general knowledge about companies — only information from the retrieved filings.

CRITICAL RULES:
1. Answer based ONLY on the provided context
2. If the context doesn't contain the answer, say "I don't have enough information in the retrieved filings to answer this."
3. Always cite your sources using the format [ticker, filing_type, accession_number]
4. Quote directly from filings when the exact wording matters
5. Be specific with numbers, dates, and facts when they appear in context
6. If comparing companies, use evidence from each company's filings
7. Distinguish between what the filing explicitly states vs what you're inferring
8. When referring to previous conversation turns, say "as we discussed" or "building on earlier question"

Your tone is that of a professional financial analyst: precise, factual, and objective."""


DECOMPOSE_SYSTEM_PROMPT = """You are a query planner for a financial research system. Your job is to break down complex user questions into smaller, specific sub-queries that can each be answered by searching SEC filings.

The system has filings for these 10 companies:
MSFT (Microsoft), AAPL (Apple), GOOGL (Alphabet), NVDA (Nvidia), META (Meta),
AMZN (Amazon), TSLA (Tesla), JPM (JPMorgan), V (Visa), UNH (UnitedHealth)

Rules:
1. Simple factual questions → return 1 sub-query
2. Comparison questions → return 1 sub-query per entity being compared
3. Multi-faceted questions → return 1 sub-query per distinct facet
4. Aggregation questions ("which companies...") → return 1 sub-query per likely company (up to 4)
5. Maximum 4 sub-queries total
6. Each sub-query should be SPECIFIC and include entity names, not pronouns
7. Use conversation history to resolve pronouns (it, that, one, they) to specific entity names

Examples:
"Compare X and Y's approach to Z" → ["X approach to Z", "Y approach to Z"]
"Which one is bigger?" (with context about X and Y) → ["X size", "Y size"]
"Which tech companies mention climate risk?" → ["Microsoft climate risk", "Google climate risk", "Apple climate risk", "Meta climate risk"]

Return JSON:
{
    "reasoning": "brief explanation",
    "sub_queries": ["specific query 1", "specific query 2", ...]
}"""


# Pydantic models
class QueryPlan(BaseModel):
    """Decomposition plan for a user question."""
    reasoning: str = Field(description="Why this decomposition")
    sub_queries: list[str] = Field(description="List of sub-queries to execute")


@dataclass
class ConversationTurn:
    """A single Q&A turn in the conversation."""
    question: str
    answer: str
    chunks_used: list[RetrievedChunk]


@dataclass
class RAGResponse:
    """Result of a RAG query."""
    question: str
    answer: str
    retrieved_chunks: list[RetrievedChunk]
    sub_queries: list[str] = field(default_factory=list)
    
    def print_formatted(self) -> None:
        """Pretty-print the response."""
        print("\n" + "=" * 60)
        print(f"❓ Question: {self.question}")
        print("=" * 60)
        
        if self.sub_queries and len(self.sub_queries) > 1:
            print(f"\n🧠 Broke into {len(self.sub_queries)} sub-queries:")
            for i, sq in enumerate(self.sub_queries, 1):
                print(f"   {i}. {sq}")
        
        print(f"\n💡 Answer:\n{self.answer}\n")
        print(f"📚 Sources used ({len(self.retrieved_chunks)} unique chunks):")
        for i, chunk in enumerate(self.retrieved_chunks, 1):
            # Clamp relevance to 0-100% range (distances can exceed 1.0 for dissimilar chunks)
            relevance = max(0.0, min(1.0, 1 - chunk.distance))
            print(f"  {i}. {chunk.ticker} {chunk.filing_type} "
                  f"(relevance: {relevance:.1%})")


# === CORE FUNCTIONS ===


#def get_llm(model: str = LLM_MODEL) -> ChatOpenAI:
#    """Create an LLM client."""
#    return ChatOpenAI(
#        model=model,
#        temperature=LLM_TEMPERATURE,
#        api_key=settings.openai_api_key,
#    )

def get_llm(model: str = LLM_MODEL):
    """Create an LLM client (Azure or OpenAI based on config)."""
    if settings.use_azure:
        from langchain_openai import AzureChatOpenAI
        return AzureChatOpenAI(
            azure_deployment=settings.azure_llm_deployment,
            api_key=settings.azure_openai_api_key,
            azure_endpoint=settings.azure_openai_endpoint,
            api_version=settings.azure_openai_api_version,
            temperature=LLM_TEMPERATURE,
        )
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            temperature=LLM_TEMPERATURE,
            api_key=settings.openai_api_key,
        )

def retrieve_chunks(
    query: str,
    n_results: int = DEFAULT_N_RESULTS,
    filter_ticker: Optional[str] = None,
) -> list[RetrievedChunk]:
    """Retrieve relevant chunks for a query."""
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


def decompose_query(
    question: str,
    history: Optional[list] = None,
    verbose: bool = False,
) -> QueryPlan:
    """Break a complex question into sub-queries using LLM.
    
    Uses conversation history to resolve pronouns ("it", "that", "which one")
    to specific entities mentioned in previous turns.
    """
    llm = get_llm()
    structured_llm = llm.with_structured_output(QueryPlan)
    
    # Build context from recent conversation history
    history_context = ""
    if history:
        recent = history[-3:]  # Last 3 turns
        history_parts = []
        for turn in recent:
            # Include question and first 500 chars of answer for context
            answer_preview = turn.answer[:500] + "..." if len(turn.answer) > 500 else turn.answer
            history_parts.append(
                f"Previous Q: {turn.question}\n"
                f"Previous A: {answer_preview}"
            )
        history_context = "\n\n".join(history_parts)
    
    # Build the decomposition request
    if history_context:
        user_message = f"""Previous conversation:
{history_context}

---

New question: {question}

Break down this question. CRITICAL: resolve pronouns like "it", "that", "one", "they" 
using the previous conversation. Replace them with specific entity names from context.

For example, if previous question was about Microsoft and Google, and new question is 
"which one invests more?", your sub-queries should be specifically about Microsoft and Google,
NOT generic entities."""
    else:
        user_message = f"Break down this question:\n\n{question}"
    
    messages = [
        SystemMessage(content=DECOMPOSE_SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ]
    
    plan = structured_llm.invoke(messages)
    
    if verbose:
        print(f"📋 Query plan: {plan.reasoning}")
        print(f"   Sub-queries: {plan.sub_queries}")
    
    return plan


def build_context_string(chunks: list[RetrievedChunk]) -> str:
    """Combine chunks into context string for LLM."""
    if not chunks:
        return "[No relevant context retrieved]"
    
    return "\n---\n".join(chunk.to_context_string() for chunk in chunks)


def build_conversation_messages(
    history: list[ConversationTurn],
    current_question: str,
    context: str,
) -> list:
    """Build message list including conversation history."""
    messages = [SystemMessage(content=ANSWER_SYSTEM_PROMPT)]
    
    for turn in history[-MAX_CONVERSATION_HISTORY:]:
        messages.append(HumanMessage(content=turn.question))
        messages.append(AIMessage(content=turn.answer))
    
    current_prompt = f"""Context from SEC filings:

{context}

---

Question: {current_question}

Answer based only on the context above. Cite sources as [TICKER, FILING_TYPE, ACCESSION]."""
    
    messages.append(HumanMessage(content=current_prompt))
    
    return messages


# === MAIN AGENT CLASS ===


class ResearchAgent:
    """Stateful research agent with conversation memory and query decomposition."""
    
    def __init__(self, use_decomposition: bool = True):
        self.history: list[ConversationTurn] = []
        self.use_decomposition = use_decomposition
    
    def reset(self) -> None:
        self.history = []
    
    def ask(
        self,
        question: str,
        filter_ticker: Optional[str] = None,
        n_results: int = DEFAULT_N_RESULTS,
        verbose: bool = False,
    ) -> RAGResponse:
        """Ask a question using decomposition + retrieval + LLM synthesis."""
        if verbose:
            print(f"\n🔍 Processing: {question}")
            if self.history:
                print(f"   (Turn {len(self.history) + 1} in conversation)")
        
        if self.use_decomposition:
            plan = decompose_query(
                question=question,
                history=self.history,
                verbose=verbose,
            )
            sub_queries = plan.sub_queries
        else:
            sub_queries = [question]
        
        all_chunks = []
        for sub_q in sub_queries:
            chunks = retrieve_chunks(
                query=sub_q,
                n_results=n_results,
                filter_ticker=filter_ticker,
            )
            all_chunks.extend(chunks)
            
            if verbose:
                print(f"   📄 '{sub_q[:50]}...': {len(chunks)} chunks")
        
        unique_chunks = deduplicate_chunks(all_chunks)
        
        if verbose:
            print(f"   🧹 After dedup: {len(unique_chunks)} unique chunks "
                  f"(removed {len(all_chunks) - len(unique_chunks)})")
        
        context = build_context_string(unique_chunks)
        messages = build_conversation_messages(
            history=self.history,
            current_question=question,
            context=context,
        )
        
        if verbose:
            print(f"   🤖 Calling {LLM_MODEL} with {len(messages)} messages...")
        
        llm = get_llm()
        response = llm.invoke(messages)
        answer = response.content
        
        turn = ConversationTurn(
            question=question,
            answer=answer,
            chunks_used=unique_chunks,
        )
        self.history.append(turn)
        
        return RAGResponse(
            question=question,
            answer=answer,
            retrieved_chunks=unique_chunks,
            sub_queries=sub_queries,
        )
    
    def ask_aggregation(
        self,
        question: str,
        topic: str,
        relevance_threshold: float = 0.2,
        verbose: bool = False,
    ) -> RAGResponse:
        """
        Search ALL 10 companies for a topic.
        
        Use this for "which companies..." questions where you want 
        complete coverage instead of just 4 sub-queries.
        
        Args:
            question: The original user question
            topic: The topic to search for in each company 
                   (e.g. "climate change material risk")
            relevance_threshold: Minimum relevance (0-1) to keep chunks
            verbose: Print intermediate steps
        """
        from src.data.sec_downloader import COMPANIES
        
        if verbose:
            print(f"\n🔍 Aggregation search: {question}")
            print(f"   Searching {len(COMPANIES)} companies for: {topic}")
        
        all_chunks = []
        company_relevance = {}  # Track best relevance per company
        
        for ticker in COMPANIES.keys():
            chunks = retrieve_chunks(
                query=topic,
                n_results=2,  # Top 2 per company
                filter_ticker=ticker,
            )
            all_chunks.extend(chunks)
            
            if chunks:
                best_relevance = max(0, 1 - chunks[0].distance)
                company_relevance[ticker] = best_relevance
                
                if verbose:
                    print(f"   📄 {ticker}: {len(chunks)} chunks "
                          f"(best relevance: {best_relevance:.1%})")
            elif verbose:
                print(f"   📄 {ticker}: no chunks found")
        
        # Keep only chunks with reasonable relevance
        filtered_chunks = [
            c for c in all_chunks 
            if (1 - c.distance) >= relevance_threshold
        ]
        
        if verbose:
            print(f"   🧹 Filtered to {len(filtered_chunks)} chunks above "
                  f"{relevance_threshold:.0%} relevance "
                  f"(from {len(all_chunks)} total)")
        
        unique_chunks = deduplicate_chunks(filtered_chunks)
        
        context = build_context_string(unique_chunks)
        messages = build_conversation_messages(
            history=self.history,
            current_question=question,
            context=context,
        )
        
        if verbose:
            print(f"   🤖 Calling {LLM_MODEL}...")
        
        llm = get_llm()
        response = llm.invoke(messages)
        answer = response.content
        
        turn = ConversationTurn(
            question=question,
            answer=answer,
            chunks_used=unique_chunks,
        )
        self.history.append(turn)
        
        return RAGResponse(
            question=question,
            answer=answer,
            retrieved_chunks=unique_chunks,
            sub_queries=[f"{t}: {topic}" for t in COMPANIES.keys()],
        )


def answer_question(
    question: str,
    n_results: int = DEFAULT_N_RESULTS,
    filter_ticker: Optional[str] = None,
    verbose: bool = False,
) -> RAGResponse:
    """Single-shot question answering (no conversation memory)."""
    agent = ResearchAgent(use_decomposition=True)
    return agent.ask(
        question=question,
        filter_ticker=filter_ticker,
        n_results=n_results,
        verbose=verbose,
    )


if __name__ == "__main__":
    print("=" * 60)
    print("TEST 1: Comparative question with decomposition")
    print("=" * 60)
    
    agent = ResearchAgent()
    
    response = agent.ask(
        "Compare Microsoft's and Google's approaches to AI strategy",
        verbose=True,
    )
    response.print_formatted()
    
    print("\n\n" + "=" * 60)
    print("TEST 2: Follow-up question (uses conversation memory)")
    print("=" * 60)
    
    followup = agent.ask(
        "Which one invests more in infrastructure?",
        verbose=True,
    )
    followup.print_formatted()
    
    print("\n\n" + "=" * 60)
    print("TEST 3: Aggregation — search all 10 companies")
    print("=" * 60)
    
    agent3 = ResearchAgent()
    
    response3 = agent3.ask_aggregation(
        question="Which companies identify climate change as a material risk?",
        topic="climate change material risk",
        verbose=True,
    )
    response3.print_formatted()
