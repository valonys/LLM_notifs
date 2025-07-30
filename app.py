import streamlit as st
import os
import time
import json
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Generator
import asyncio

# Core imports
from core.vector_store import EnhancedVectorStore
from core.document_processor import AdvancedDocumentProcessor
from core.model_manager import ModelManager
from core.query_router import QueryRouter
from core.pivot_analyzer import PivotAnalyzer
from core.database_manager import DatabaseManager
from utils.cache_manager import CacheManager
from utils.error_handler import ErrorHandler
from utils.config import Config
from prompts.industrial_prompts import INDUSTRIAL_PROMPTS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize components
@st.cache_resource
def initialize_components():
    """Initialize all core components with caching"""
    try:
        config = Config()
        cache_manager = CacheManager()
        error_handler = ErrorHandler()
        
        # Initialize database manager
        try:
            database_manager = DatabaseManager(config)
        except Exception as e:
            logger.warning(f"Database manager initialization failed: {str(e)}")
            database_manager = None
        
        vector_store = EnhancedVectorStore(config, database_manager)
        document_processor = AdvancedDocumentProcessor(config)
        model_manager = ModelManager(config)
        query_router = QueryRouter(config)
        pivot_analyzer = PivotAnalyzer(config)
        
        return {
            'config': config,
            'cache_manager': cache_manager,
            'error_handler': error_handler,
            'database_manager': database_manager,
            'vector_store': vector_store,
            'document_processor': document_processor,
            'model_manager': model_manager,
            'query_router': query_router,
            'pivot_analyzer': pivot_analyzer
        }
    except Exception as e:
        logger.error(f"Failed to initialize components: {str(e)}")
        st.error(f"Initialization failed: {str(e)}")
        return None

# Page configuration
st.set_page_config(
    page_title="DigiTwin Enhanced RAG System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI
st.markdown("""
    <style>
    @import url('https://fonts.cdnfonts.com/css/tw-cen-mt');
    * {
        font-family: 'Tw Cen MT', sans-serif !important;
    }
    .main-header {
        background: linear-gradient(90deg, #1f77b4, #2ca02c);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #1f77b4;
    }
    .analysis-section {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize session state variables"""
    defaults = {
        'components': None,
        'chat_history': [],
        'current_model': 'EE Smartest Agent',
        'current_prompt_type': 'Daily Report Summarization',
        'processed_documents': None,
        'pivot_data': None,
        'analysis_results': {},
        'query_history': [],
        'selected_analysis_type': 'Document Analysis'
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

# Load components
if st.session_state.components is None:
    with st.spinner("Initializing DigiTwin Enhanced RAG System..."):
        st.session_state.components = initialize_components()

if st.session_state.components is None:
    st.error("Failed to initialize the application. Please refresh the page.")
    st.stop()

# Main application
def main():
    """Main application flow"""
    
    # Header
    st.markdown("""
        <div class="main-header">
            <h1 style="color: white; text-align: center; margin: 0;">
                🔍 DigiTwin Enhanced RAG System
            </h1>
            <p style="color: white; text-align: center; margin: 0.5rem 0 0 0;">
                Advanced Industrial Inspection Analysis with AI
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.image("https://raw.githubusercontent.com/achilela/vila_fofoka_analysis/991f4c6e4e1dc7a8e24876ca5aae5228bcdb4dba/Ataliba_Avatar.jpg", width=100)
        st.title("⚙️ Configuration")
        
        # Analysis type selection
        analysis_type = st.selectbox(
            "Select Analysis Type",
            ["Document Analysis", "Pivot Table Analysis", "Hybrid Analysis"],
            index=0,
            key="analysis_type_selector"
        )
        st.session_state.selected_analysis_type = analysis_type
        
        # Model selection
        available_models = [
            "EE Smartest Agent",
            "EdJa-Valonys", 
            "JI Divine Agent",
            "OpenAI GPT-4",
            "Domain-Specific Industrial Model"
        ]
        
        selected_model = st.selectbox(
            "Select AI Model",
            available_models,
            index=0,
            key="model_selector"
        )
        st.session_state.current_model = selected_model
        
        # Prompt type selection
        prompt_types = list(INDUSTRIAL_PROMPTS.keys())
        selected_prompt = st.selectbox(
            "Select Analysis Focus",
            prompt_types,
            index=0,
            key="prompt_selector"
        )
        st.session_state.current_prompt_type = selected_prompt
        
        # Advanced settings
        with st.expander("🔧 Advanced Settings"):
            chunk_strategy = st.selectbox(
                "Chunking Strategy",
                ["Hierarchical", "Semantic", "Hybrid", "Fixed-Size"],
                index=0
            )
            
            embedding_model = st.selectbox(
                "Embedding Model",
                ["all-MiniLM-L6-v2", "industrial-bert-base", "multi-qa-MiniLM", "domain-specific"],
                index=0
            )
            
            retrieval_mode = st.selectbox(
                "Retrieval Mode",
                ["Hybrid (Dense + Sparse)", "Dense Only", "Sparse Only"],
                index=0
            )
            
            max_tokens = st.slider("Max Response Tokens", 100, 4000, 2000)
            temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # File upload section
        st.markdown("### 📁 Document & Data Upload")
        
        uploaded_files = st.file_uploader(
            "Upload Documents and Data Files",
            type=['pdf', 'xlsx', 'xls', 'docx', 'txt', 'csv'],
            accept_multiple_files=True,
            key="file_uploader"
        )
        
        if uploaded_files:
            process_uploaded_files(uploaded_files)
        
        # Query interface
        st.markdown("### 💬 Query Interface")
        
        # Query input
        user_query = st.text_area(
            "Enter your query about the industrial data:",
            height=100,
            placeholder="e.g., 'Analyze safety violations from last month' or 'Show me pivot table of notifications by FPSO'"
        )
        
        # Query buttons
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            if st.button("🔍 Analyze", type="primary"):
                if user_query:
                    handle_query(user_query)
                else:
                    st.warning("Please enter a query first.")
        
        with col_b:
            if st.button("📊 Generate Insights"):
                generate_automated_insights()
        
        with col_c:
            if st.button("🔄 Clear History"):
                st.session_state.chat_history = []
                st.session_state.query_history = []
                st.rerun()
    
    with col2:
        # System status
        display_system_status()
        
        # Quick actions
        st.markdown("### ⚡ Quick Actions")
        
        if st.button("📈 Trend Analysis"):
            perform_trend_analysis()
        
        if st.button("⚠️ Safety Review"):
            perform_safety_review()
        
        if st.button("📋 Compliance Check"):
            perform_compliance_check()
        
        if st.button("🔧 Equipment Status"):
            perform_equipment_analysis()
    
    # Results display
    display_results()
    
    # Chat history
    display_chat_history()

def process_uploaded_files(uploaded_files):
    """Process uploaded files with enhanced capabilities"""
    components = st.session_state.components
    
    with st.spinner("Processing uploaded files..."):
        try:
            # Process documents
            documents = components['document_processor'].process_files(uploaded_files)
            
            if documents:
                # Enhanced vector store creation
                st.session_state.processed_documents = components['vector_store'].create_enhanced_store(documents)
                
                # Process any Excel files for pivot analysis
                excel_files = [f for f in uploaded_files if f.name.endswith(('.xlsx', '.xls'))]
                if excel_files:
                    pivot_data = components['pivot_analyzer'].process_excel_files(excel_files)
                    st.session_state.pivot_data = pivot_data
                
                st.success(f"Successfully processed {len(uploaded_files)} files!")
                
                # Display file summary
                with st.expander("📋 File Processing Summary"):
                    for i, file in enumerate(uploaded_files):
                        st.write(f"✅ {file.name} - {file.type}")
            else:
                st.error("Failed to process uploaded files.")
                
        except Exception as e:
            logger.error(f"File processing error: {str(e)}")
            st.error(f"Error processing files: {str(e)}")

def handle_query(query: str):
    """Handle user queries with intelligent routing"""
    components = st.session_state.components
    
    with st.spinner("Analyzing your query..."):
        try:
            # Route query to appropriate handler
            query_type = components['query_router'].classify_query(query)
            
            # Add to query history
            st.session_state.query_history.append({
                'timestamp': datetime.now(),
                'query': query,
                'type': query_type
            })
            
            # Generate response based on query type
            if query_type == 'pivot_table':
                response = handle_pivot_query(query)
            elif query_type == 'document':
                response = handle_document_query(query)
            else:
                response = handle_hybrid_query(query)
            
            # Add to chat history
            st.session_state.chat_history.append({
                'timestamp': datetime.now(),
                'user': query,
                'assistant': response,
                'type': query_type
            })
            
            # Display response
            with st.container():
                st.markdown("### 🤖 AI Response")
                st.markdown(response)
                
        except Exception as e:
            logger.error(f"Query handling error: {str(e)}")
            st.error(f"Error processing query: {str(e)}")

def handle_pivot_query(query: str) -> str:
    """Handle pivot table specific queries"""
    components = st.session_state.components
    
    if st.session_state.pivot_data is None:
        return "No pivot table data available. Please upload Excel files with tabular data first."
    
    try:
        # Generate SQL-like query from natural language
        sql_query = components['pivot_analyzer'].natural_language_to_sql(query)
        
        # Execute query on pivot data
        results = components['pivot_analyzer'].execute_query(sql_query, st.session_state.pivot_data)
        
        # Generate natural language response
        response = components['model_manager'].generate_pivot_response(
            query, results.data.to_dict() if hasattr(results.data, 'to_dict') else results.data, st.session_state.current_model
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Pivot query error: {str(e)}")
        return f"Error processing pivot query: {str(e)}"

def handle_document_query(query: str) -> str:
    """Handle document-based queries"""
    components = st.session_state.components
    
    if st.session_state.processed_documents is None:
        return "No documents available. Please upload documents first."
    
    try:
        # Retrieve relevant documents
        relevant_docs = components['vector_store'].similarity_search(
            query, 
            k=5,
            retrieval_mode="hybrid"
        )
        
        # Generate response using retrieved context
        response = components['model_manager'].generate_document_response(
            query, 
            relevant_docs, 
            st.session_state.current_model,
            st.session_state.current_prompt_type
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Document query error: {str(e)}")
        return f"Error processing document query: {str(e)}"

def handle_hybrid_query(query: str) -> str:
    """Handle queries that require both document and pivot data"""
    components = st.session_state.components
    
    try:
        doc_response = ""
        pivot_response = ""
        
        # Get document context if available
        if st.session_state.processed_documents:
            relevant_docs = components['vector_store'].similarity_search(query, k=3)
            doc_response = components['model_manager'].generate_document_response(
                query, relevant_docs, st.session_state.current_model, st.session_state.current_prompt_type
            )
        
        # Get pivot context if available
        if st.session_state.pivot_data:
            pivot_results = components['pivot_analyzer'].analyze_for_query(query)
            pivot_response = components['model_manager'].generate_pivot_response(
                query, pivot_results, st.session_state.current_model
            )
        
        # Combine responses
        combined_response = components['model_manager'].combine_responses(
            query, doc_response, pivot_response
        )
        
        return combined_response
        
    except Exception as e:
        logger.error(f"Hybrid query error: {str(e)}")
        return f"Error processing hybrid query: {str(e)}"

def generate_automated_insights():
    """Generate automated insights from available data"""
    components = st.session_state.components
    
    with st.spinner("Generating automated insights..."):
        try:
            insights = []
            
            # Document insights
            if st.session_state.processed_documents:
                doc_insights = components['vector_store'].generate_document_insights()
                insights.extend(doc_insights)
            
            # Pivot insights
            if st.session_state.pivot_data:
                pivot_insights = components['pivot_analyzer'].generate_automated_insights()
                insights.extend(pivot_insights)
            
            if insights:
                st.markdown("### 🔍 Automated Insights")
                for insight in insights:
                    with st.container():
                        st.markdown(f"**{insight['category']}**: {insight['description']}")
                        if insight.get('recommendation'):
                            st.info(f"💡 Recommendation: {insight['recommendation']}")
            else:
                st.info("No data available for insight generation. Please upload files first.")
                
        except Exception as e:
            logger.error(f"Insight generation error: {str(e)}")
            st.error(f"Error generating insights: {str(e)}")

def perform_trend_analysis():
    """Perform trend analysis on available data"""
    if st.session_state.pivot_data is None:
        st.warning("No data available for trend analysis. Please upload Excel files first.")
        return
    
    components = st.session_state.components
    
    with st.spinner("Performing trend analysis..."):
        try:
            trends = components['pivot_analyzer'].analyze_trends()
            
            st.markdown("### 📈 Trend Analysis Results")
            
            for trend in trends:
                with st.container():
                    st.markdown(f"**{trend['metric']}**")
                    st.write(f"Direction: {trend['direction']}")
                    st.write(f"Magnitude: {trend['magnitude']}")
                    if trend.get('insight'):
                        st.info(trend['insight'])
                        
        except Exception as e:
            logger.error(f"Trend analysis error: {str(e)}")
            st.error(f"Error performing trend analysis: {str(e)}")

def perform_safety_review():
    """Perform safety-focused analysis"""
    query = "Analyze all safety violations, incidents, and compliance issues. Provide risk assessment and recommendations."
    handle_query(query)

def perform_compliance_check():
    """Perform compliance analysis"""
    query = "Review compliance status across all operations. Identify gaps and regulatory requirements."
    handle_query(query)

def perform_equipment_analysis():
    """Perform equipment performance analysis"""
    query = "Analyze equipment performance, maintenance status, and reliability metrics."
    handle_query(query)

def display_system_status():
    """Display system status and metrics"""
    st.markdown("### 📊 System Status")
    
    # Status indicators
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.session_state.processed_documents:
            st.success("Documents: Ready")
        else:
            st.warning("Documents: None")
    
    with col2:
        if st.session_state.pivot_data:
            st.success("Data: Ready")
        else:
            st.warning("Data: None")
    
    with col3:
        # Database status
        components = st.session_state.components
        if components.get('database_manager'):
            try:
                health = components['database_manager'].health_check()
                if health.get('status') == 'healthy':
                    st.success("Database: Connected")
                else:
                    st.error("Database: Error")
            except:
                st.error("Database: Offline")
        else:
            st.warning("Database: Not configured")
    
    # Database statistics
    if components.get('database_manager'):
        with st.expander("🗄️ Database Statistics"):
            try:
                stats = components['database_manager'].get_database_stats()
                if stats:
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Documents", stats.get('documents_count', 0))
                        st.metric("Chunks", stats.get('document_chunks_count', 0))
                        st.metric("Queries", stats.get('query_history_count', 0))
                    with col_b:
                        st.metric("Safety Incidents", stats.get('safety_incidents_count', 0))
                        st.metric("Equipment Records", stats.get('equipment_count', 0))
                        st.metric("Cache Entries", stats.get('cache_entries_count', 0))
                    
                    if stats.get('database_size'):
                        st.info(f"Database Size: {stats['database_size']}")
                else:
                    st.info("Database statistics unavailable")
            except Exception as e:
                st.error(f"Database error: {str(e)}")
    
    # Query statistics
    if st.session_state.query_history:
        st.markdown(f"**Queries Processed:** {len(st.session_state.query_history)}")
        
        # Query type distribution
        query_types = [q['type'] for q in st.session_state.query_history]
        type_counts = pd.Series(query_types).value_counts()
        st.bar_chart(type_counts)

def display_results():
    """Display analysis results and visualizations"""
    if st.session_state.analysis_results:
        st.markdown("### 📊 Analysis Results")
        
        for category, results in st.session_state.analysis_results.items():
            with st.expander(f"📈 {category}"):
                if isinstance(results, dict):
                    for key, value in results.items():
                        st.write(f"**{key}:** {value}")
                else:
                    st.write(results)

def display_chat_history():
    """Display chat history with enhanced formatting"""
    if st.session_state.chat_history:
        st.markdown("### 💬 Query History")
        
        for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):  # Show last 5
            with st.expander(f"Query {len(st.session_state.chat_history) - i}: {chat['user'][:50]}..."):
                st.markdown(f"**Query Type:** {chat['type']}")
                st.markdown(f"**Timestamp:** {chat['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                st.markdown(f"**User:** {chat['user']}")
                st.markdown(f"**Assistant:** {chat['assistant']}")

if __name__ == "__main__":
    main()
