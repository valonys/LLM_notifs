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

# Custom CSS for modern UI
st.markdown("""
    <style>
    .main-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
    }
    .chat-container {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        min-height: 600px;
    }
    .message-user {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 5px 20px;
        margin: 1rem 0;
        margin-left: 20%;
    }
    .message-bot {
        background: #f8f9fa;
        color: #333;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 20px 5px;
        margin: 1rem 0;
        margin-right: 20%;
        border-left: 4px solid #667eea;
    }
    .sidebar-header {
        text-align: center;
        padding: 1rem 0;
        border-bottom: 1px solid #e0e0e0;
        margin-bottom: 1rem;
    }
    .query-input {
        border: 2px solid #e0e0e0;
        border-radius: 25px;
        padding: 1rem 1.5rem;
        font-size: 16px;
        width: 100%;
        outline: none;
        transition: border-color 0.3s;
    }
    .query-input:focus {
        border-color: #667eea;
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

# Sidebar Configuration
def setup_sidebar():
    """Setup sidebar with file upload and configuration"""
    with st.sidebar:
        # Logo and Header
        try:
            st.image("assets/ValonyLabs_Logo.png", width=200)
        except:
            st.markdown("### 🔧 DigiTwin")
        
        st.markdown("---")
        
        # File Upload Section
        st.markdown("### 📁 Document Upload")
        uploaded_files = st.file_uploader(
            "Upload documents for analysis",
            type=['pdf', 'docx', 'xlsx', 'csv', 'txt'],
            accept_multiple_files=True,
            help="Supported formats: PDF, DOCX, XLSX, CSV, TXT"
        )
        
        # Model Configuration
        st.markdown("### ⚙️ Configuration")
        available_models = [
            'EE Smartest Agent', 'EdJa-Valonys', 'JI Divine Agent', 
            'OpenAI GPT-4', 'Local Model'
        ]
        selected_model = st.selectbox(
            "AI Model", 
            available_models,
            index=0
        )
        
        # Analysis Type
        analysis_type = st.selectbox(
            "Analysis Type",
            ['Document Analysis', 'Safety Analysis', 'Compliance Check', 'Data Pivot']
        )
        
        # Process uploaded files
        if uploaded_files:
            process_uploaded_files(uploaded_files)
        
        return selected_model, analysis_type

# Main application
def main():
    """Main application flow"""
    
    # Setup sidebar first
    selected_model, analysis_type = setup_sidebar()
    
    # Main content area with modern chat interface
    st.markdown("""
        <div class="main-container">
            <h1 style="color: white; text-align: center; margin: 0; font-size: 2.5rem;">
                DigiTwin Enhanced RAG System
            </h1>
            <p style="color: white; text-align: center; margin: 0.5rem 0 0 0; font-size: 1.2rem;">
                Advanced Industrial Inspection Analysis with AI
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Chat Interface
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Display chat history
    display_chat_history()
    
    # Query Input at the bottom
    col1, col2 = st.columns([6, 1])
    with col1:
        user_query = st.text_input(
            "Ask me anything about your documents or request data analysis...",
            placeholder="e.g., 'Analyze safety incidents by location' or 'Summarize equipment maintenance reports'",
            key="main_query_input",
            label_visibility="collapsed"
        )
    with col2:
        send_query = st.button("🚀 Send", type="primary", use_container_width=True)
    
    # Process query
    if send_query and user_query:
        process_user_query(user_query, selected_model, analysis_type)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Chat and file processing functions
def display_chat_history():
    """Display chat history with modern styling"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if not st.session_state.chat_history:
        st.markdown("""
            <div style="text-align: center; padding: 3rem; color: #666;">
                <h3>👋 Welcome to DigiTwin</h3>
                <p>Upload documents and start asking questions about your industrial data!</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"""
                    <div class="message-user">
                        <strong>You:</strong><br>{message["content"]}
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="message-bot">
                        <strong>🤖 DigiTwin:</strong><br>{message["content"]}
                    </div>
                """, unsafe_allow_html=True)

def process_uploaded_files(uploaded_files):
    """Process uploaded files"""
    if not uploaded_files:
        return
    
    with st.spinner("Processing uploaded files..."):
        try:
            components = st.session_state.components
            if not components:
                st.error("System components not initialized")
                return
            
            # Process files
            processed_docs = components['document_processor'].process_files(uploaded_files)
            
            if processed_docs:
                # Create enhanced vector store
                vector_store = components['vector_store'].create_enhanced_store(processed_docs)
                st.session_state.processed_documents = processed_docs
                st.success(f"Successfully processed {len(uploaded_files)} files!")
            else:
                st.error("Failed to process uploaded files")
                
        except Exception as e:
            st.error(f"File processing error: {str(e)}")

def process_user_query(query, model, analysis_type):
    """Process user query and generate response"""
    try:
        # Add user message to chat
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        st.session_state.chat_history.append({"role": "user", "content": query})
        
        # Process query based on analysis type
        components = st.session_state.components
        if not components:
            st.error("System components not initialized")
            return
        
        with st.spinner("Generating response..."):
            # Route query
            query_type = components['query_router'].classify_query(query)
            
            if query_type == 'pivot_table' and st.session_state.get('pivot_data'):
                # Handle pivot table queries
                result = components['pivot_analyzer'].analyze_query(
                    query, st.session_state.pivot_data
                )
                response = f"**Analysis Results:**\n\n{result.summary}\n\n**Key Insights:**\n" + "\n".join([f"• {insight}" for insight in result.insights])
            else:
                # Handle document analysis queries
                if st.session_state.get('processed_documents'):
                    # Use RAG system
                    response = "Based on your uploaded documents, I can help analyze the data. However, the AI models need proper configuration to provide detailed responses."
                else:
                    response = "Please upload documents first so I can analyze them for you."
            
            # Add bot response to chat
            st.session_state.chat_history.append({"role": "bot", "content": response})
            
            # Refresh the page to show new messages
            st.rerun()
            
    except Exception as e:
        st.error(f"Query processing error: {str(e)}")
        logger.error(f"Query processing failed: {str(e)}")

# Run the main application
if __name__ == "__main__":
    main()
