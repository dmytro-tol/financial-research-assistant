"""My custom questions for the RAG system."""
from src.agent.rag_pipeline import answer_question

questions = [
    "How has Nvidia's revenue changed over the last 3 years?",
    "Which companies planning to layoffs because of AI",
]

for q in questions:
    answer_question(q).print_formatted()
