import streamlit as st
import json
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Handle API key for deployment
def get_api_key():
    """Get API key from environment or Streamlit secrets"""
    try:
        # Try Streamlit secrets first (for Streamlit Cloud)
        return st.secrets["GOOGLE_API_KEY"]
    except:
        # Try environment variable (for other deployments)
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            st.warning("⚠️ Google API key not found. Using local mode only.")
        return api_key

# Set API key if available
api_key = get_api_key()
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key

# Try to import our modules with better error handling
try:
    from rag_retriever import MedicalRAGRetriever
except ImportError as e:
    st.error(f"❌ Error importing modules: {str(e)}")
    st.info("💡 This might be due to missing dependencies. Check the requirements.txt file.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Medical Guidelines RAG System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling and fix white section issues
st.markdown("""
<style>
    /* Fix main content area */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 100%;
    }
    
    /* Remove default Streamlit margins that cause white sections */
    .element-container {
        margin-bottom: 1rem !important;
    }
    
    /* Header styling */
    .main-header {
        font-size: 2.5rem;
        color: #1e3a8a;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #3b82f6;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #f0f9ff 0%, #e0f2fe 100%);
        border-radius: 0.5rem;
    }
    
    /* Query input styling */
    .query-section {
        background-color: #f8fafc;
        padding: 1.5rem;
        border-radius: 0.75rem;
        border: 2px solid #e2e8f0;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Answer box styling */
    .answer-box {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        padding: 1.5rem;
        border-radius: 0.75rem;
        border: 2px solid #3b82f6;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(59, 130, 246, 0.1);
    }
    
    /* Source box styling */
    .source-box {
        background: linear-gradient(135deg, #fefce8 0%, #fef3c7 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #eab308;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(234, 179, 8, 0.1);
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
        padding: 1rem;
        border-radius: 0.75rem;
        text-align: center;
        border: 1px solid #cbd5e1;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    /* Sidebar improvements */
    .css-1d391kg {
        background-color: #f8fafc;
        border-right: 2px solid #e2e8f0;
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 0.5rem;
        border: 2px solid #3b82f6;
        background-color: #3b82f6;
        color: white;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #1e3a8a;
        border-color: #1e3a8a;
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(30, 58, 138, 0.3);
    }
    
    /* Fix text area styling */
    .stTextArea > div > div > textarea {
        border-radius: 0.5rem;
        border: 2px solid #e2e8f0;
    }
    
    /* Remove extra whitespace */
    .block-container > div {
        gap: 1rem;
    }
    
    /* Improve expander styling */
    .streamlit-expanderHeader {
        background-color: #f1f5f9;
        border-radius: 0.5rem;
        border: 1px solid #cbd5e1;
    }
    
    /* Footer styling */
    .footer {
        background-color: #f8fafc;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-top: 3px solid #3b82f6;
        margin-top: 2rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

def initialize_retriever():
    """Initialize the RAG retriever with error handling"""
    try:
        retriever = MedicalRAGRetriever()
        return retriever, None
    except Exception as e:
        error_msg = f"""
        ❌ **Error initializing RAG system:** {str(e)}
        
        **To fix this:**
        1. Run `python embeddings_generator.py` first
        2. Ensure all required files exist:
           - medical_guidelines_faiss_index.faiss
           - medical_guidelines_metadata.json
        3. Check that section_chunks.txt is in the same directory
        """
        return None, error_msg

def main():
    """Main Streamlit application"""
    
    # Header with improved styling
    st.markdown("""
    <div class="main-header">
        🏥 Medical Guidelines RAG System
        <div style="font-size: 1rem; color: #64748b; margin-top: 0.5rem;">
            Intelligent Medical Information Retrieval & AI-Powered Responses
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        # Retrieval settings
        st.markdown("**📊 Search Parameters**")
        top_k = st.slider("Number of sections to retrieve", 1, 10, 3)
        similarity_threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.5, 0.1)
        
        st.markdown("---")
        st.markdown("**🤖 AI Assistant**")
        use_gemini = st.checkbox("Enable AI-powered responses", value=True)
        
        if use_gemini:
            st.info("� Using advanced AI for enhanced responses")
        else:
            st.info("📝 Using local rule-based responses")
        
        st.markdown("---")
        st.markdown("### 📋 Sample Queries")
        sample_queries = [
            "What is the recommended treatment for type 2 diabetes?",
            "How should blood glucose be monitored?",
            "What are the dietary recommendations for diabetes?",
            "What complications can arise from diabetes?",
            "How should insulin therapy be managed?",
            "What are the diagnostic criteria for diabetes mellitus?"
        ]
        
        for query in sample_queries:
            if st.button(f"💡 {query[:40]}...", key=query):
                st.session_state.selected_query = query
    
    # Initialize retriever
    retriever, error_msg = initialize_retriever()
    
    if error_msg:
        st.error(error_msg)
        st.stop()
    
    # Main interface with improved layout
    # Remove white space issues
    st.markdown('<div style="margin: 0; padding: 0;">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1], gap="large")
    
    with col1:
        # Query section with better styling
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); padding: 1.5rem; border-radius: 0.75rem; border: 2px solid #3b82f6; margin-bottom: 1.5rem; box-shadow: 0 4px 6px rgba(59, 130, 246, 0.1);">
            <h3 style="margin: 0 0 0.5rem 0; color: #1e3a8a;">🔍 Ask a Medical Question</h3>
            <p style="color: #64748b; margin: 0;">Enter your question about diabetes management and clinical guidelines</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Query input with better styling
        query_input = st.text_area(
            label="Medical Question Input",  # Non-empty label for accessibility
            value=st.session_state.get('selected_query', ''),
            height=120,
            placeholder="💬 e.g., What are the treatment options for diabetes complications?\n\n📝 Try typing questions about:\n• Treatment protocols\n• Diagnostic criteria\n• Medication guidelines\n• Dietary recommendations",
            label_visibility="hidden"  # Hide the label but keep it for accessibility
        )
        # Action buttons in columns
        btn_col1, btn_col2, btn_col3 = st.columns([2, 1, 1])
        
        with btn_col1:
            search_clicked = st.button("🔍 Search Medical Guidelines", type="primary", use_container_width=True)
        
        with btn_col2:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.selected_query = ""
                st.rerun()
        
        with btn_col3:
            if st.button("📋 Examples", use_container_width=True):
                st.session_state.show_examples = not st.session_state.get('show_examples', False)
        # Process search when button clicked
        if search_clicked:
            if query_input.strip():
                with st.spinner("🧠 Analyzing medical guidelines..."):
                    try:
                        # Perform RAG query
                        result = retriever.query(
                            query_input,
                            top_k=top_k,
                            use_gemini=use_gemini,
                            similarity_threshold=similarity_threshold
                        )
                        
                        # Store result in session state
                        st.session_state.last_result = result
                        st.session_state.last_query = query_input
                        st.success("✅ Search completed successfully!")
                        
                    except Exception as e:
                        st.error(f"❌ Error processing query: {str(e)}")
                        st.info("💡 Try simplifying your question or check your internet connection.")
            else:
                st.warning("⚠️ Please enter a medical question to search.")
    
    with col2:
        # System status with better design
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%); padding: 1.5rem; border-radius: 0.75rem; border: 2px solid #cbd5e1; margin-bottom: 1rem;">
            <h3 style="margin: 0 0 1rem 0; color: #1e293b;">📊 System Status</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if retriever:
            # Status indicators
            st.markdown("**🏥 Medical Database**")
            col2_1, col2_2 = st.columns(2)
            
            with col2_1:
                st.metric(
                    label="📚 Sections",
                    value=len(retriever.sections),
                    help="Number of medical guideline sections indexed"
                )
            
            with col2_2:
                st.metric(
                    label="🚀 Status",
                    value="Ready",
                    help="System is ready to process queries"
                )
            
            # Configuration display
            st.markdown("**⚙️ Current Settings**")
            
            # Display current settings in a nice format
            settings_info = f"""
            **📊 Search Config:**
            - Sections: {top_k}
            - Threshold: {similarity_threshold}
            
            **🤖 AI Mode:**
            - {"🚀 Gemini AI Enabled" if use_gemini else "📝 Local Mode"}
            """
            st.markdown(settings_info)
            
            # Quick stats
            if 'last_result' in st.session_state:
                st.markdown("**� Last Query Stats**")
                last_result = st.session_state.last_result
                
                st.metric(
                    label="Sources Found",
                    value=len(last_result.get('sources', [])),
                    help="Number of relevant sections retrieved"
                )
        
        else:
            st.error("❌ System not initialized")
        
        # Help section
        with st.expander("ℹ️ How to Use", expanded=False):
            st.markdown("""
            **🔍 Search Tips:**
            1. Ask specific medical questions
            2. Focus on diabetes-related topics
            3. Use clinical terminology when possible
            
            **⚙️ Settings:**
            - **Sections**: More = comprehensive answers
            - **Threshold**: Higher = more precise results
            - **AI Mode**: Natural language responses
            
            **📋 Example Topics:**
            • Treatment protocols
            • Diagnostic criteria  
            • Medication guidelines
            • Dietary management
            • Complications
            """)
        
        # Quick actions
        st.markdown("**🚀 Quick Actions**")
        if st.button("� Reset Settings", use_container_width=True):
            st.session_state.clear()
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close the no-margin div
    
    # Display results with improved styling
    if 'last_result' in st.session_state:
        result = st.session_state.last_result
        query = st.session_state.last_query
        
        st.markdown("---")
        
        # Results header
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); padding: 1rem; border-radius: 0.75rem; border-left: 4px solid #3b82f6; margin: 1rem 0;">
            <h2 style="margin: 0; color: #1e3a8a;">📋 Query Results</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Query display with copy functionality
        col_query, col_copy = st.columns([4, 1])
        with col_query:
            st.markdown(f"""
            <div style="background-color: #f8fafc; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #6366f1; margin-bottom: 1rem;">
                <strong>🔍 Your Question:</strong><br>
                <em style="color: #374151;">{query}</em>
            </div>
            """, unsafe_allow_html=True)
        
        # Answer display with better formatting
        st.markdown("### 💡 AI Response")
        
        # Fix f-string syntax by extracting the newline replacement
        answer_content = result["answer"].replace('\n', '<br>')
        answer_html = f"""
        <div class="answer-box">
            <div style="margin-bottom: 0.5rem;">
                <strong style="color: #1e3a8a;">Medical Guidelines Response:</strong>
            </div>
            <div style="line-height: 1.6; color: #374151;">
                {answer_content}
            </div>
        </div>
        """
        st.markdown(answer_html, unsafe_allow_html=True)
        
        # Sources with enhanced display
        if result['sources']:
            st.markdown("### 📚 Source References")
            st.markdown("*Click on each source to see the relevant section content*")
            
            # Create tabs for sources if there are many
            if len(result['sources']) > 3:
                source_tabs = st.tabs([f"📄 Source {i+1}" for i in range(len(result['sources']))])
                
                for i, (source, tab) in enumerate(zip(result['sources'], source_tabs)):
                    with tab:
                        display_source_card(source, i+1)
            else:
                for i, source in enumerate(result['sources'], 1):
                    display_source_card(source, i)
        
        else:
            st.warning("⚠️ No relevant sources found. Try adjusting your search parameters.")
        
        # Enhanced metadata section
        with st.expander("🔧 Detailed Query Information", expanded=False):
            col_meta1, col_meta2, col_meta3, col_meta4 = st.columns(4)
            
            with col_meta1:
                st.metric(
                    label="📊 Sections Retrieved",
                    value=len(result['sources']),
                    help="Number of relevant sections found"
                )
            
            with col_meta2:
                st.metric(
                    label="🎯 Top-K Setting",
                    value=top_k,
                    help="Maximum sections to retrieve"
                )
            
            with col_meta3:
                st.metric(
                    label="📏 Similarity Threshold",
                    value=f"{similarity_threshold:.1f}",
                    help="Minimum relevance required"
                )
            
            with col_meta4:
                avg_score = sum(s['similarity_score'] for s in result['sources']) / len(result['sources']) if result['sources'] else 0
                st.metric(
                    label="📈 Avg Relevance",
                    value=f"{avg_score:.2f}",
                    help="Average similarity score"
                )
            
            # Additional metadata
            st.markdown("**🔍 Search Analysis:**")
            search_quality = "Excellent" if avg_score > 0.7 else "Good" if avg_score > 0.5 else "Moderate" if avg_score > 0.3 else "Low"
            st.info(f"Search Quality: {search_quality} | AI Mode: {'Gemini' if use_gemini else 'Local'}")

def display_source_card(source, index):
    """Display a source card with enhanced styling"""
    score_color = "#059669" if source['similarity_score'] > 0.7 else "#d97706" if source['similarity_score'] > 0.5 else "#dc2626"
    
    # Fix the f-string by extracting the content preview separately
    content_preview = source['content_preview'][:300]
    relevance_text = "Excellent" if source['similarity_score'] > 0.7 else "Good" if source['similarity_score'] > 0.5 else "Moderate"
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #fefce8 0%, #fef3c7 100%); padding: 1rem; border-radius: 0.75rem; border-left: 4px solid #eab308; margin: 0.5rem 0; box-shadow: 0 2px 4px rgba(234, 179, 8, 0.1);">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
            <h4 style="margin: 0; color: #92400e;">📄 Source {index}: {source['title']}</h4>
            <span style="background-color: {score_color}; color: white; padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-size: 0.875rem; font-weight: 600;">
                {source['similarity_score']:.2f}
            </span>
        </div>
        <div style="color: #78716c; font-size: 0.9rem; margin-bottom: 0.5rem;">
            <strong>Relevance:</strong> {relevance_text}
        </div>
        <div style="background-color: #fffbeb; padding: 0.75rem; border-radius: 0.5rem; border: 1px solid #fed7aa;">
            <strong>Content Preview:</strong><br>
            <span style="color: #374151; line-height: 1.5;">{content_preview}...</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    # Add medical disclaimer at the end
    st.markdown("---")
    st.info("⚠️ **Medical Disclaimer:** This system is for educational and research purposes only. Always consult qualified healthcare professionals for medical decisions.")
main()
