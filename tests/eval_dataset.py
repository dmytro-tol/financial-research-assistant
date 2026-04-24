"""Test dataset for evaluating the research assistant.

Each question has:
- expected_tickers: which companies should appear in retrieved chunks
- expected_keywords: specific terms that should be in a correct answer
- answerable: whether the question CAN be answered from our corpus
- category: type of question for breakdown analysis
"""
from dataclasses import dataclass, field


@dataclass
class EvalQuestion:
    """A single evaluation question with expected results."""
    question: str
    category: str  # 'factual', 'comparative', 'aggregation', 'unanswerable'
    expected_tickers: list[str] = field(default_factory=list)
    expected_keywords: list[str] = field(default_factory=list)
    answerable: bool = True
    notes: str = ""


# === FACTUAL QUESTIONS (single company, specific facts) ===

FACTUAL_QUESTIONS = [
    EvalQuestion(
        question="What is Microsoft's primary cloud platform?",
        category="factual",
        expected_tickers=["MSFT"],
        expected_keywords=["Azure"],
    ),
    EvalQuestion(
        question="What are Apple's main product categories?",
        category="factual",
        expected_tickers=["AAPL"],
        expected_keywords=["iPhone", "Mac", "iPad", "Services"],
    ),
    EvalQuestion(
        question="What was Apple's total net sales in fiscal 2024?",
        category="factual",
        expected_tickers=["AAPL"],
        expected_keywords=["391"],  # Should mention $391 billion
    ),
    EvalQuestion(
        question="What does Nvidia's data center segment include?",
        category="factual",
        expected_tickers=["NVDA"],
        expected_keywords=["GPU", "data center"],
    ),
    EvalQuestion(
        question="What is Meta's Reality Labs segment?",
        category="factual",
        expected_tickers=["META"],
        expected_keywords=["Reality Labs", "Quest", "VR"],
    ),
    EvalQuestion(
        question="What is JPMorgan's primary business focus?",
        category="factual",
        expected_tickers=["JPM"],
        expected_keywords=["banking", "financial"],
    ),
    EvalQuestion(
        question="What are Visa's main revenue sources?",
        category="factual",
        expected_tickers=["V"],
        expected_keywords=["payment", "transaction"],
    ),
    EvalQuestion(
        question="What services does UnitedHealth provide?",
        category="factual",
        expected_tickers=["UNH"],
        expected_keywords=["health", "insurance", "care"],
    ),
    EvalQuestion(
        question="What was Google's cloud revenue in 2023?",
        category="factual",
        expected_tickers=["GOOGL"],
        expected_keywords=["33", "Cloud"],  # Should find ~$33B
    ),
    EvalQuestion(
        question="What is Tesla's autonomous driving technology called?",
        category="factual",
        expected_tickers=["TSLA"],
        expected_keywords=["Autopilot", "FSD"],
    ),
]


# === COMPARATIVE QUESTIONS (multiple companies) ===

COMPARATIVE_QUESTIONS = [
    EvalQuestion(
        question="Compare Microsoft and Google's AI strategies",
        category="comparative",
        expected_tickers=["MSFT", "GOOGL"],
        expected_keywords=["Azure", "OpenAI"],
    ),
    EvalQuestion(
        question="How do Apple and Google differ in revenue composition?",
        category="comparative",
        expected_tickers=["AAPL", "GOOGL"],
        expected_keywords=["iPhone", "advertising", "Services"],
    ),
    EvalQuestion(
        question="Compare Amazon and Microsoft's cloud offerings",
        category="comparative",
        expected_tickers=["AMZN", "MSFT"],
        expected_keywords=["AWS", "Azure"],
    ),
    EvalQuestion(
        question="How do Nvidia and Meta approach AI investment?",
        category="comparative",
        expected_tickers=["NVDA", "META"],
        expected_keywords=["AI"],
    ),
]


# === AGGREGATION QUESTIONS (across many companies) ===

AGGREGATION_QUESTIONS = [
    EvalQuestion(
        question="Which companies rely heavily on advertising revenue?",
        category="aggregation",
        expected_tickers=["META", "GOOGL"],
        expected_keywords=["advertising"],
    ),
    EvalQuestion(
        question="Which companies mention supply chain risks?",
        category="aggregation",
        expected_tickers=["AAPL", "NVDA", "TSLA"],  # Hardware companies
        expected_keywords=["supply chain"],
    ),
    EvalQuestion(
        question="Which companies discuss AI regulatory risks?",
        category="aggregation",
        expected_tickers=["MSFT", "GOOGL", "META"],
        expected_keywords=["AI", "regulatory"],
    ),
]


# === UNANSWERABLE QUESTIONS (test hallucination handling) ===

UNANSWERABLE_QUESTIONS = [
    EvalQuestion(
        question="What is Tesla's quarterly dividend payment?",
        category="unanswerable",
        answerable=False,
        notes="Tesla doesn't pay dividends — system should say so",
    ),
    EvalQuestion(
        question="What was Facebook's revenue in 1995?",
        category="unanswerable",
        answerable=False,
        notes="Facebook didn't exist in 1995",
    ),
    EvalQuestion(
        question="What are Siemens' main product lines?",
        category="unanswerable",
        answerable=False,
        notes="Siemens not in our corpus (10 US mega-caps only)",
    ),
    EvalQuestion(
        question="What is Microsoft's detailed quantum computing roadmap for 2030?",
        category="unanswerable",
        answerable=False,
        notes="Too specific future-looking, unlikely in 10-K",
    ),
]


# Combined dataset
ALL_QUESTIONS = (
    FACTUAL_QUESTIONS
    + COMPARATIVE_QUESTIONS
    + AGGREGATION_QUESTIONS
    + UNANSWERABLE_QUESTIONS
)


if __name__ == "__main__":
    print(f"📋 Total questions: {len(ALL_QUESTIONS)}")
    print(f"   Factual:       {len(FACTUAL_QUESTIONS)}")
    print(f"   Comparative:   {len(COMPARATIVE_QUESTIONS)}")
    print(f"   Aggregation:   {len(AGGREGATION_QUESTIONS)}")
    print(f"   Unanswerable:  {len(UNANSWERABLE_QUESTIONS)}")
    
    print("\nSample questions:")
    for q in ALL_QUESTIONS[:5]:
        print(f"  [{q.category}] {q.question}")
