"""LangGraph-based research agent with self-reflection and retry logic."""
from typing import TypedDict, Optional, Literal
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from src.utils.config import settings
from src.retrieval.dedup import deduplicate_chunks
from src.retrieval.models import RetrievedChunk
from src.agent.rag_pipeline import decompose_query

# === CONFIGURATION ===

LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0
MAX_RETRIES = 2
RELEVANCE_THRESHOLD = 0.2  # Lowered from 0.4 — our scores run lower
QUALITY_THRESHOLD = 0.7


# === STATE DEFINITION ===

class AgentState(TypedDict):
    """Shared state that flows through the graph.
    
    Each node reads from and writes to this state.
    """
    # Input
    question: str
    filter_ticker: Optional[str]
    
    # Planning
    sub_queries: list[str]
    reasoning: str
    
    # Retrieval
    retrieved_chunks: list[RetrievedChunk]
    chunks_relevant: bool
    
    # Synthesis
    answer: str
    
    # Quality
    quality_score: float
    quality_feedback: str
    is_good_enough: bool
    
    # Control
    retry_count: int
    max_retries: int


# === STRUCTURED OUTPUT MODELS ===

class QueryPlan(BaseModel):
    """Decomposed sub-queries for a user question."""
    reasoning: str = Field(description="Why this decomposition")
    sub_queries: list[str] = Field(description="Specific sub-queries")


class ChunkGrading(BaseModel):
    """Grade of whether retrieved chunks are relevant."""
    relevant: bool = Field(description="Are these chunks relevant to the question?")
    reasoning: str = Field(description="Why or why not")


class QualityAssessment(BaseModel):
    """Assessment of answer quality."""
    score: float = Field(description="Quality score 0-1", ge=0, le=1)
    is_good_enough: bool = Field(description="Should we accept this answer?")
    feedback: str = Field(description="What's good or missing")


# === HELPER FUNCTIONS ===

def get_llm() -> ChatOpenAI:
    """Create an LLM client."""
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        api_key=settings.openai_api_key,
    )


def format_chunks_for_prompt(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks for inclusion in LLM prompt."""
    if not chunks:
        return "[No relevant context retrieved]"
    
    return "\n---\n".join(chunk.to_context_string() for chunk in chunks)


# === GRAPH NODES ===

def decompose_node(state: AgentState) -> dict:
    """Node 1: Break the question into searchable sub-queries."""
    print(f"🧠 [decompose] Breaking down: {state['question']}")
    
    llm = get_llm()
    structured_llm = llm.with_structured_output(QueryPlan)
    
    system = """You are a query planner for a financial research system.
Break down complex questions into specific sub-queries.

The system has filings for: MSFT, AAPL, GOOGL, NVDA, META, AMZN, TSLA, JPM, V, UNH.

Rules:
- Simple factual questions → 1 sub-query
- Comparison questions → 1 sub-query per entity
- Aggregation questions → 1 sub-query per relevant entity (max 4)
- Each sub-query must be SPECIFIC (include entity names)
- Maximum 4 sub-queries"""
    
    messages = [
        SystemMessage(content=system),
        HumanMessage(content=f"Break down this question:\n\n{state['question']}"),
    ]
    
    # If this is a retry, tell the LLM
    if state.get("retry_count", 0) > 0:
        messages[-1] = HumanMessage(
            content=f"""The previous retrieval didn't find enough relevant information.
Please broaden the search or try different keywords.

Original question: {state['question']}

Break down with different/broader sub-queries than before:"""
        )
    
    plan = structured_llm.invoke(messages)
    
    print(f"   Sub-queries: {plan.sub_queries}")
    
    return {
        "sub_queries": plan.sub_queries,
        "reasoning": plan.reasoning,
    }

def retrieve_node(state: AgentState) -> dict:
    """Node 2: Retrieve chunks using hybrid search (semantic + keyword).
    
    Note: We tested cross-encoder re-ranking but it hurt evaluation metrics
    on this dataset. Cross-encoders trained on web search don't transfer well
    to dense financial filings where most chunks contain similar terminology.
    Hybrid search alone produces our best results.
    """
    from src.retrieval.hybrid_search import hybrid_search
    
    print(f"🔍 [retrieve] Hybrid search for {len(state['sub_queries'])} sub-queries...")
    
    all_chunks = []
    for sub_q in state["sub_queries"]:
        chunks = hybrid_search(
            query=sub_q,
            n_results=5,
            filter_ticker=state.get("filter_ticker"),
            verbose=False,
        )
        all_chunks.extend(chunks)
    
    unique_chunks = deduplicate_chunks(all_chunks)
    
    print(f"   Retrieved {len(all_chunks)} total, {len(unique_chunks)} unique")
    
    return {
        "retrieved_chunks": unique_chunks,
    }

def grade_chunks_node(state: AgentState) -> dict:
    """Node 3: Grade whether retrieved chunks are relevant enough."""
    print(f"⚖️  [grade] Evaluating {len(state['retrieved_chunks'])} chunks...")
    
    chunks = state["retrieved_chunks"]
    
    if not chunks:
        print("   ❌ No chunks retrieved — not relevant")
        return {"chunks_relevant": False}
    
    # Simple heuristic first: average relevance score
    avg_relevance = sum(
        max(0, 1 - c.distance) for c in chunks[:5]
    ) / min(5, len(chunks))
    
    print(f"   Average top-5 relevance: {avg_relevance:.1%}")
    
    # If average relevance too low, mark as not relevant
    if avg_relevance < RELEVANCE_THRESHOLD:
        print(f"   ❌ Below threshold ({RELEVANCE_THRESHOLD:.0%})")
        return {"chunks_relevant": False}
    
    # Use LLM to double-check for borderline cases
    if avg_relevance < RELEVANCE_THRESHOLD + 0.1:
        llm = get_llm()
        structured_llm = llm.with_structured_output(ChunkGrading)
        
        top_chunks_preview = "\n\n".join(
            f"Chunk {i+1} ({c.ticker}): {c.text[:300]}..."
            for i, c in enumerate(chunks[:3])
        )
        
        messages = [
            SystemMessage(content="You evaluate whether retrieved text chunks are relevant to a question."),
            HumanMessage(content=f"""Question: {state['question']}

Top retrieved chunks:
{top_chunks_preview}

Are these chunks actually relevant to answering the question?"""),
        ]
        
        grading = structured_llm.invoke(messages)
        print(f"   LLM grade: {'✅ relevant' if grading.relevant else '❌ not relevant'}")
        print(f"   Reasoning: {grading.reasoning}")
        
        return {"chunks_relevant": grading.relevant}
    
    print(f"   ✅ Relevant")
    return {"chunks_relevant": True}


def synthesize_node(state: AgentState) -> dict:
    """Node 4: Generate the answer from retrieved chunks."""
    print(f"✍️  [synthesize] Generating answer...")
    
    system = """You are a financial research assistant specializing in SEC filings.

You answer questions based ONLY on the context provided. Do not use general knowledge.

Rules:
1. Answer based ONLY on the provided context
2. If context is insufficient, say "I don't have enough information"
3. Cite sources as [TICKER, FILING_TYPE, ACCESSION]
4. Quote directly when specific wording matters
5. Be precise with numbers and dates from the context"""
    
    context = format_chunks_for_prompt(state["retrieved_chunks"])
    
    messages = [
        SystemMessage(content=system),
        HumanMessage(content=f"""Context from SEC filings:

{context}

---

Question: {state['question']}

Answer based only on the context above. Cite sources as [TICKER, FILING_TYPE, ACCESSION]."""),
    ]
    
    llm = get_llm()
    response = llm.invoke(messages)
    
    print(f"   Answer length: {len(response.content)} chars")
    
    return {"answer": response.content}


def reflect_node(state: AgentState) -> dict:
    """Node 5: Evaluate answer quality."""
    print(f"🔍 [reflect] Evaluating answer quality...")
    
    llm = get_llm()
    structured_llm = llm.with_structured_output(QualityAssessment)
    
    system = """You evaluate the quality of research assistant answers.

Good answers have:
- Specific facts, numbers, or quotes from the context
- Proper source citations [TICKER, FILING, ACCESSION]
- Directly address the question
- Acknowledge limitations when context is incomplete

IMPORTANT: An honest refusal is a GOOD answer when context is genuinely weak.
If the answer says "I don't have enough information" and the retrieved context 
truly doesn't address the question, score it HIGH (0.8+) and mark good_enough=True.
Hallucinating an answer would be worse than refusing.

Bad answers:
- Vague generalizations not grounded in context
- Missing citations when claiming facts
- Going off-topic
- Making claims not in the context"""
    
    messages = [
        SystemMessage(content=system),
        HumanMessage(content=f"""Question: {state['question']}

Retrieved context chunks: {len(state['retrieved_chunks'])}

Generated answer:
{state['answer']}

Evaluate the answer quality."""),
    ]
    
    assessment = structured_llm.invoke(messages)
    
    print(f"   Quality score: {assessment.score:.1%}")
    print(f"   Good enough: {assessment.is_good_enough}")
    print(f"   Feedback: {assessment.feedback}")
    
    return {
        "quality_score": assessment.score,
        "quality_feedback": assessment.feedback,
        "is_good_enough": assessment.is_good_enough,
    }


# === CONDITIONAL ROUTING ===

def route_after_grading(state: AgentState) -> Literal["synthesize", "retry", "fail"]:
    """Decide: do we have good chunks, or do we retry?"""
    if state["chunks_relevant"]:
        return "synthesize"
    
    if state["retry_count"] < state["max_retries"]:
        print(f"   🔄 Retrying (attempt {state['retry_count'] + 1}/{state['max_retries']})")
        return "retry"
    
    print(f"   ⚠️  Max retries reached, proceeding with what we have")
    return "fail"


def route_after_reflection(state: AgentState) -> Literal["accept", "retry", "fail"]:
    """Decide: is answer good enough, or retry?"""
    if state["is_good_enough"]:
        return "accept"
    
    # Only retry if we haven't hit max AND we weren't already in fallback
    # (if retry_count >= max_retries, we came from fallback)
    if state["retry_count"] < state["max_retries"]:
        print(f"   🔄 Answer not good enough, retrying...")
        return "retry"
    
    print(f"   ⚠️  Max retries reached, accepting answer")
    return "fail"


def retry_node(state: AgentState) -> dict:
    """Increment retry counter."""
    new_count = state["retry_count"] + 1
    print(f"🔁 [retry] Attempt {new_count}/{state['max_retries']}")
    return {"retry_count": new_count}


def fallback_answer_node(state: AgentState) -> dict:
    """Generate a fallback answer when retries failed."""
    print(f"🛟 [fallback] Generating best-effort answer")
    
    # Still try to synthesize with what we have
    if state["retrieved_chunks"]:
        return synthesize_node(state)
    
    return {
        "answer": (
            "I don't have enough information in the retrieved filings to answer "
            f"your question: '{state['question']}'. The knowledge base covers "
            "SEC 10-K filings for 10 mega-cap companies (MSFT, AAPL, GOOGL, "
            "NVDA, META, AMZN, TSLA, JPM, V, UNH). Your question may be about "
            "a topic not well-represented in these filings."
        )
    }


# === BUILD THE GRAPH ===

def build_graph():
    """Construct the LangGraph StateGraph."""
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("decompose", decompose_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_chunks_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("reflect", reflect_node)
    graph.add_node("retry", retry_node)
    graph.add_node("fallback", fallback_answer_node)
    
    # Entry point
    graph.set_entry_point("decompose")
    
    # Linear edges
    graph.add_edge("decompose", "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_edge("retry", "decompose")
    graph.add_edge("synthesize", "reflect")
    graph.add_edge("fallback", "reflect")
    
    # Conditional edges
    graph.add_conditional_edges(
        "grade",
        route_after_grading,
        {
            "synthesize": "synthesize",
            "retry": "retry",
            "fail": "fallback",
        }
    )
    
    graph.add_conditional_edges(
        "reflect",
        route_after_reflection,
        {
            "accept": END,
            "retry": "retry",
            "fail": END,  # Accept what we have if retries exhausted
        }
    )
    
    return graph.compile()


# === MAIN INTERFACE ===

class GraphResearchAgent:
    """Research agent built with LangGraph."""
    
    def __init__(self):
        self.graph = build_graph()
    
    def ask(
        self,
        question: str,
        filter_ticker: Optional[str] = None,
        max_retries: int = MAX_RETRIES,
    ) -> AgentState:
        """Ask a question and get the full agent state back."""
        print("\n" + "=" * 60)
        print(f"🎯 Question: {question}")
        print("=" * 60 + "\n")
        
        initial_state = {
            "question": question,
            "filter_ticker": filter_ticker,
            "sub_queries": [],
            "reasoning": "",
            "retrieved_chunks": [],
            "chunks_relevant": False,
            "answer": "",
            "quality_score": 0.0,
            "quality_feedback": "",
            "is_good_enough": False,
            "retry_count": 0,
            "max_retries": max_retries,
        }
        
        final_state = self.graph.invoke(initial_state)
        
        print("\n" + "=" * 60)
        print(f"✅ DONE (retries used: {final_state['retry_count']})")
        print("=" * 60)
        
        return final_state
    
    def ask_simple(
        self,
        question: str,
        filter_ticker: Optional[str] = None,
    ) -> str:
        """Simpler interface that just returns the answer string."""
        state = self.ask(question, filter_ticker)
        return state["answer"]

    def ask_streaming(
        self,
        question: str,
        filter_ticker: Optional[str] = None,
    ):
        """Ask a question with streaming output.
        
        Yields status updates and answer tokens as they're generated.
        
        Yields:
            tuple: (event_type, content)
              event_type: 'status' | 'token' | 'final_state'
              content: status message OR token text OR final state dict
        """
        from src.retrieval.hybrid_search import hybrid_search
        from langchain_openai import ChatOpenAI
        from langchain.schema import SystemMessage, HumanMessage
        
        # === DECOMPOSE ===
        yield ("status", "🧠 Planning approach...")
        
        plan = decompose_query(question=question, history=None, verbose=False)
        sub_queries = plan.sub_queries
        
        if len(sub_queries) > 1:
            yield ("status", f"📋 Broke into {len(sub_queries)} sub-queries")
        
        # === RETRIEVE ===
        yield ("status", "🔍 Searching SEC filings...")
        
        all_chunks = []
        for sub_q in sub_queries:
            chunks = hybrid_search(
                query=sub_q,
                n_results=5,
                filter_ticker=filter_ticker,
                verbose=False,
            )
            all_chunks.extend(chunks)
        
        unique_chunks = deduplicate_chunks(all_chunks)
        
        yield ("status", f"📚 Found {len(unique_chunks)} relevant passages")
        
        # === SYNTHESIZE WITH STREAMING ===
        yield ("status", "✍️ Generating answer...")
        
        # Build the messages
        system_msg = """You are a financial research assistant specializing in SEC filings.

You answer questions based ONLY on the context provided. Do not use general knowledge.

Rules:
1. Answer based ONLY on the provided context
2. If context is insufficient, say "I don't have enough information"
3. Cite sources as [TICKER, FILING_TYPE, ACCESSION]
4. Quote directly when specific wording matters
5. Be precise with numbers and dates from the context"""
        
        context = "\n---\n".join(c.to_context_string() for c in unique_chunks)
        
        user_msg = f"""Context from SEC filings:

{context}

---

Question: {question}

Answer based only on the context above. Cite sources as [TICKER, FILING_TYPE, ACCESSION]."""
        
        # Create streaming LLM
        streaming_llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            api_key=settings.openai_api_key,
            streaming=True,
        )
        
        messages = [
            SystemMessage(content=system_msg),
            HumanMessage(content=user_msg),
        ]
        
        # Stream tokens
        full_answer = ""
        for chunk in streaming_llm.stream(messages):
            token = chunk.content
            if token:
                full_answer += token
                yield ("token", token)
        
        # Yield final state with metadata
        final_state = {
            "question": question,
            "answer": full_answer,
            "retrieved_chunks": unique_chunks,
            "sub_queries": sub_queries,
        }
        yield ("final_state", final_state)

if __name__ == "__main__":
    agent = GraphResearchAgent()
    
    # Test 1: Simple question (should pass reflection on first try)
    print("\n\n🧪 TEST 1: Simple factual question")
    result1 = agent.ask("What is Microsoft's primary cloud business?")
    print("\n📝 ANSWER:")
    print(result1["answer"])
    print(f"\n📊 Quality: {result1['quality_score']:.1%}")
    print(f"📚 Chunks used: {len(result1['retrieved_chunks'])}")
    
    # Test 2: Comparative question
    print("\n\n🧪 TEST 2: Comparative question")
    result2 = agent.ask("Compare Apple and Google's revenue breakdown")
    print("\n📝 ANSWER:")
    print(result2["answer"])
    print(f"\n📊 Quality: {result2['quality_score']:.1%}")
    
    # Test 3: Potentially weak question (tests retry logic)
    print("\n\n🧪 TEST 3: Question that may trigger retry")
    result3 = agent.ask("What blockchain investments do these companies have?")
    print("\n📝 ANSWER:")
    print(result3["answer"])
    print(f"\n📊 Quality: {result3['quality_score']:.1%}")
    print(f"🔁 Retries used: {result3['retry_count']}")
