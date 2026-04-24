import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
from src.agent.rag_pipeline import ResearchAgent
from src.data.sec_downloader import COMPANIES


# === PAGE CONFIGURATION ===

st.set_page_config(
    page_title="Financial Research Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# === CUSTOM STYLING ===

st.markdown("""
<style>
    /* Make chat messages look professional */
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    /* Source citation cards */
    .source-card {
        background-color: #f0f2f6;
        border-left: 3px solid #0066cc;
        padding: 0.5rem 1rem;
        margin: 0.25rem 0;
        border-radius: 0.25rem;
        font-size: 0.85rem;
    }
    
    /* Sub-query display */
    .sub-query {
        background-color: #e8f4f8;
        padding: 0.3rem 0.6rem;
        border-radius: 0.3rem;
        margin: 0.2rem 0;
        font-size: 0.8rem;
        color: #0066cc;
    }
    
    /* Header styling */
    h1 {
        color: #0a1929;
    }
    
    /* Stats boxes */
    [data-testid="stMetricValue"] {
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)


# === SESSION STATE INITIALIZATION ===

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "agent" not in st.session_state:
        st.session_state.agent = ResearchAgent(use_decomposition=True)
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "query_count" not in st.session_state:
        st.session_state.query_count = 0


# === SIDEBAR ===

def render_sidebar():
    """Render the sidebar with filters and stats."""
    with st.sidebar:
        st.title("📊 Research Assistant")
        st.caption("SEC filings AI analyst")
        
        st.divider()
        
        # Company filter
        st.subheader("🔍 Filter by Company")
        
        # Build dropdown options
        company_options = ["All companies"] + [
            f"{ticker} - {name}" for ticker, name in COMPANIES.items()
        ]
        
        selected = st.selectbox(
            "Focus search on:",
            options=company_options,
            index=0,
            help="Limit retrieval to specific company filings",
        )
        
        # Extract ticker if specific company chosen
        filter_ticker = None
        if selected != "All companies":
            filter_ticker = selected.split(" - ")[0]
        
        st.divider()
        
        # Corpus info
        st.subheader("📚 Knowledge Base")
        st.metric("Companies", len(COMPANIES))
        st.metric("10-K Filings", 30)
        st.metric("Text Chunks", "8,654")
        
        st.divider()
        
        # Session stats
        st.subheader("📈 Session")
        st.metric("Questions asked", st.session_state.query_count)
        
        # Reset button
        if st.button("🔄 New Conversation", use_container_width=True):
            st.session_state.agent.reset()
            st.session_state.messages = []
            st.session_state.query_count = 0
            st.rerun()
        
        st.divider()
        
        # Info
        with st.expander("ℹ️ About"):
            st.markdown("""
            This assistant analyzes SEC 10-K filings for 10 major US companies 
            using retrieval-augmented generation (RAG).
            
            **Tech stack:**
            - OpenAI GPT-4o-mini
            - ChromaDB vector store
            - LangChain orchestration
            - Pydantic structured outputs
            
            **Companies covered:** MSFT, AAPL, GOOGL, NVDA, META, AMZN, TSLA, JPM, V, UNH
            """)
        
        return filter_ticker


# === MAIN CONTENT ===

def render_header():
    """Render the main header."""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.title("Financial Deep Research Assistant")
        st.caption(
            "Ask me anything about the last 3 years of SEC 10-K filings for "
            "10 major US corporations. I'll find relevant passages and cite my sources."
        )


def render_example_questions():
    """Show example questions to help users get started."""
    if not st.session_state.messages:
        st.subheader("💡 Try asking:")
        
        examples = [
            "How does Microsoft describe its AI strategy?",
            "Compare Apple and Google's revenue sources",
            "What are the main risks Meta identifies with AI?",
            "How has Nvidia's data center business evolved?",
            "What does JPMorgan say about regulatory compliance?",
        ]
        
        cols = st.columns(len(examples))
        
        for i, (col, example) in enumerate(zip(cols, examples)):
            with col:
                if st.button(
                    example,
                    key=f"example_{i}",
                    use_container_width=True,
                ):
                    st.session_state.pending_question = example
                    st.rerun()


def render_message(message):
    """Render a single message with its metadata."""
    role = message["role"]
    
    with st.chat_message(role):
        st.markdown(message["content"])
        
        # Show sub-queries if this was a decomposed question
        if role == "assistant" and message.get("sub_queries"):
            if len(message["sub_queries"]) > 1:
                with st.expander(f"🧠 Broke into {len(message['sub_queries'])} sub-queries"):
                    for i, sq in enumerate(message["sub_queries"], 1):
                        st.markdown(
                            f'<div class="sub-query">{i}. {sq}</div>',
                            unsafe_allow_html=True,
                        )
        
        # Show sources if available
        if role == "assistant" and message.get("sources"):
            with st.expander(f"📚 Sources ({len(message['sources'])} chunks)"):
                for i, source in enumerate(message["sources"], 1):
                    relevance = max(0, min(1, 1 - source["distance"]))
                    st.markdown(
                        f"""<div class="source-card">
                        <strong>{i}. {source['ticker']} {source['filing_type']}</strong>
                        — {source['accession_number']}
                        <br><small>Relevance: {relevance:.1%}</small>
                        <br><em>{source['preview']}...</em>
                        </div>""",
                        unsafe_allow_html=True,
                    )


def render_conversation():
    """Render all conversation messages."""
    for message in st.session_state.messages:
        render_message(message)


def handle_user_input(user_input: str, filter_ticker: str = None):
    """Process a user question and add response to conversation."""
    # Add user message to display
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
    })
    
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching filings and analyzing..."):
            try:
                response = st.session_state.agent.ask(
                    question=user_input,
                    filter_ticker=filter_ticker,
                    verbose=False,
                )
                
                # Show answer
                st.markdown(response.answer)
                
                # Prepare source data for storage
                sources = [
                    {
                        "ticker": chunk.ticker,
                        "filing_type": chunk.filing_type,
                        "accession_number": chunk.accession_number,
                        "distance": chunk.distance,
                        "preview": chunk.text[:200],
                    }
                    for chunk in response.retrieved_chunks
                ]
                
                # Show sub-queries if decomposed
                if len(response.sub_queries) > 1:
                    with st.expander(f"🧠 Broke into {len(response.sub_queries)} sub-queries"):
                        for i, sq in enumerate(response.sub_queries, 1):
                            st.markdown(
                                f'<div class="sub-query">{i}. {sq}</div>',
                                unsafe_allow_html=True,
                            )
                
                # Show sources
                with st.expander(f"📚 Sources ({len(sources)} chunks)"):
                    for i, source in enumerate(sources, 1):
                        relevance = max(0, min(1, 1 - source["distance"]))
                        st.markdown(
                            f"""<div class="source-card">
                            <strong>{i}. {source['ticker']} {source['filing_type']}</strong>
                            — {source['accession_number']}
                            <br><small>Relevance: {relevance:.1%}</small>
                            <br><em>{source['preview']}...</em>
                            </div>""",
                            unsafe_allow_html=True,
                        )
                
                # Save to message history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response.answer,
                    "sources": sources,
                    "sub_queries": response.sub_queries,
                })
                
                st.session_state.query_count += 1
                
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                })


# === MAIN APP ===

def main():
    """Main application entry point."""
    initialize_session_state()
    
    filter_ticker = render_sidebar()
    render_header()
    
    st.divider()
    
    # Show examples if no conversation yet
    render_example_questions()
    
    # Render conversation history
    render_conversation()
    
    # Handle example button clicks
    if "pending_question" in st.session_state:
        pending = st.session_state.pop("pending_question")
        handle_user_input(pending, filter_ticker)
        st.rerun()
    
    # Chat input at bottom
    user_input = st.chat_input("Ask about any of the 10 companies...")
    
    if user_input:
        handle_user_input(user_input, filter_ticker)
        st.rerun()


if __name__ == "__main__":
    main()
