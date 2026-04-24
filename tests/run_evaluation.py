"""Run the research assistant against the test dataset and compute metrics."""
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable
from src.agent.graph_agent import GraphResearchAgent
from src.agent.rag_pipeline import ResearchAgent
from tests.eval_dataset import ALL_QUESTIONS, EvalQuestion


@dataclass
class EvalResult:
    """Result of evaluating one question."""
    question: str
    category: str
    answer: str
    retrieved_tickers: list[str]
    
    # Metrics
    retrieval_precision: float  # 0-1: did we find chunks from expected companies?
    keyword_coverage: float     # 0-1: did answer contain expected keywords?
    answered: bool              # did the agent produce a substantive answer?
    refused_correctly: bool     # for unanswerable: did it say "don't know"?
    
    # Meta
    duration_seconds: float
    chunks_retrieved: int


def evaluate_retrieval(
    expected_tickers: list[str],
    actual_tickers: list[str],
) -> float:
    """How many expected tickers appear in retrieved chunks?"""
    if not expected_tickers:
        return 1.0  # No expectation = pass
    
    found = sum(1 for t in expected_tickers if t in actual_tickers)
    return found / len(expected_tickers)


def evaluate_keywords(answer: str, expected_keywords: list[str]) -> float:
    """How many expected keywords appear in the answer (case-insensitive)?"""
    if not expected_keywords:
        return 1.0
    
    answer_lower = answer.lower()
    found = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return found / len(expected_keywords)


def is_refusal(answer: str) -> bool:
    """Detect if the agent refused to answer (for unanswerable questions)."""
    refusal_phrases = [
        "i don't have enough information",
        "i don't have information",
        "not available in the retrieved",
        "cannot answer",
        "unable to answer",
        "no information",
        "not in the retrieved",
    ]
    answer_lower = answer.lower()
    return any(phrase in answer_lower for phrase in refusal_phrases)


def is_substantive(answer: str) -> bool:
    """Did the agent produce a real answer (not just a refusal)?"""
    if is_refusal(answer):
        return False
    # Substantive answers have at least 50 characters of content
    return len(answer.strip()) >= 50


def evaluate_question(
    question: EvalQuestion,
    agent_fn: Callable,
) -> EvalResult:
    """Run one question through the agent and compute metrics."""
    start = time.time()
    
    try:
        response = agent_fn(question.question)
        answer = response["answer"]
        chunks = response["retrieved_chunks"]
    except Exception as e:
        answer = f"ERROR: {e}"
        chunks = []
    
    duration = time.time() - start
    
    # Extract tickers from retrieved chunks
    actual_tickers = list(set(c.ticker for c in chunks))
    
    # Compute metrics
    retrieval_precision = evaluate_retrieval(
        question.expected_tickers,
        actual_tickers,
    )
    keyword_coverage = evaluate_keywords(answer, question.expected_keywords)
    answered = is_substantive(answer)
    refused_correctly = (not question.answerable) and is_refusal(answer)
    
    return EvalResult(
        question=question.question,
        category=question.category,
        answer=answer[:300] + ("..." if len(answer) > 300 else ""),
        retrieved_tickers=sorted(actual_tickers),
        retrieval_precision=retrieval_precision,
        keyword_coverage=keyword_coverage,
        answered=answered,
        refused_correctly=refused_correctly,
        duration_seconds=duration,
        chunks_retrieved=len(chunks),
    )


def run_evaluation(agent_name: str = "graph") -> list[EvalResult]:
    """Run the full evaluation suite."""
    print(f"🧪 Starting evaluation with {agent_name} agent")
    print(f"📋 {len(ALL_QUESTIONS)} questions to evaluate")
    print("=" * 70)
    
    # Pick the agent
    if agent_name == "graph":
        graph_agent = GraphResearchAgent()
        def agent_fn(q):
            state = graph_agent.ask(q)
            return {
                "answer": state["answer"],
                "retrieved_chunks": state["retrieved_chunks"],
            }
    else:
        rag_agent = ResearchAgent(use_decomposition=True)
        def agent_fn(q):
            response = rag_agent.ask(q, verbose=False)
            return {
                "answer": response.answer,
                "retrieved_chunks": response.retrieved_chunks,
            }
    
    results = []
    for i, question in enumerate(ALL_QUESTIONS, 1):
        print(f"\n[{i}/{len(ALL_QUESTIONS)}] {question.category}: {question.question[:60]}...")
        result = evaluate_question(question, agent_fn)
        results.append(result)
        print(f"   Retrieval: {result.retrieval_precision:.0%} | "
              f"Keywords: {result.keyword_coverage:.0%} | "
              f"Time: {result.duration_seconds:.1f}s")
    
    return results


def print_report(results: list[EvalResult]) -> None:
    """Print a summary report of evaluation results."""
    print("\n\n" + "=" * 70)
    print("📊 EVALUATION REPORT")
    print("=" * 70)
    
    # Overall stats
    total = len(results)
    avg_retrieval = sum(r.retrieval_precision for r in results) / total
    avg_keyword = sum(r.keyword_coverage for r in results) / total
    avg_duration = sum(r.duration_seconds for r in results) / total
    total_duration = sum(r.duration_seconds for r in results)
    
    print(f"\n🎯 Overall Metrics:")
    print(f"   Questions evaluated:   {total}")
    print(f"   Avg retrieval precision: {avg_retrieval:.1%}")
    print(f"   Avg keyword coverage:    {avg_keyword:.1%}")
    print(f"   Avg response time:       {avg_duration:.1f}s")
    print(f"   Total duration:          {total_duration:.0f}s")
    
    # By category
    categories = {}
    for r in results:
        categories.setdefault(r.category, []).append(r)
    
    print(f"\n📋 By Category:")
    for cat, cat_results in categories.items():
        n = len(cat_results)
        avg_r = sum(r.retrieval_precision for r in cat_results) / n
        avg_k = sum(r.keyword_coverage for r in cat_results) / n
        print(f"   {cat:15} ({n:2} questions)  "
              f"retrieval: {avg_r:.0%}  keywords: {avg_k:.0%}")
    
    # Unanswerable handling
    unanswerable = [r for r in results if r.category == "unanswerable"]
    if unanswerable:
        refused_count = sum(1 for r in unanswerable if r.refused_correctly)
        print(f"\n🛡️  Hallucination resistance:")
        print(f"   Unanswerable questions: {len(unanswerable)}")
        print(f"   Correctly refused:      {refused_count}/{len(unanswerable)} "
              f"({refused_count/len(unanswerable):.0%})")
    
    # Problem questions
    print(f"\n⚠️  Questions needing attention (retrieval < 50% or keywords < 50%):")
    problems = [
        r for r in results 
        if (r.retrieval_precision < 0.5 or r.keyword_coverage < 0.5)
        and r.category != "unanswerable"
    ]
    if problems:
        for r in problems:
            print(f"   ❌ {r.category}: {r.question[:60]}")
            print(f"      Expected tickers retrieved: {r.retrieval_precision:.0%}, "
                  f"keywords: {r.keyword_coverage:.0%}")
    else:
        print(f"   ✅ None! All answerable questions passed thresholds.")
    
    # Performance summary
    print(f"\n⚡ Performance:")
    fastest = min(results, key=lambda r: r.duration_seconds)
    slowest = max(results, key=lambda r: r.duration_seconds)
    print(f"   Fastest: {fastest.duration_seconds:.1f}s — {fastest.question[:50]}")
    print(f"   Slowest: {slowest.duration_seconds:.1f}s — {slowest.question[:50]}")


def save_results(results: list[EvalResult], path: str = "tests/eval_results.json"):
    """Save detailed results to JSON for later analysis."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        "total_questions": len(results),
        "results": [asdict(r) for r in results],
    }
    
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"\n💾 Detailed results saved to {output_path}")


if __name__ == "__main__":
    # Default: run with graph agent (our best)
    # To compare: change to "rag" and re-run
    results = run_evaluation(agent_name="graph")
    print_report(results)
    save_results(results)
