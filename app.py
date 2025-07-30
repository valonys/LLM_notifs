import streamlit as st
import os
import time
import json
import logging
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd

# Load environment variables
load_dotenv()

# Configure logging early
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import our enhanced components
try:
    from utils.config import Config
    from utils.cache_manager import CacheManager  
    from utils.error_handler import ErrorHandler
    from core.database_manager import DatabaseManager
    from core.document_processor import AdvancedDocumentProcessor as DocumentProcessor
    from core.vector_store import VectorStore
    from core.model_manager import ModelManager
    from core.query_router import QueryRouter
    from core.pivot_analyzer import PivotAnalyzer
    from core.embedding_manager import EmbeddingManager
    from prompts.industrial_prompts import INDUSTRIAL_PROMPTS
except ImportError as e:
    logger.warning(f"Some components not available: {str(e)}")
    # Create fallback classes
    class FallbackComponent:
        def __init__(self):
            pass
        def __getattr__(self, name):
            return lambda *args, **kwargs: None
    
    Config = FallbackComponent
    CacheManager = FallbackComponent
    ErrorHandler = FallbackComponent
    DatabaseManager = FallbackComponent
    class DocumentProcessor(FallbackComponent):
        def __init__(self, config=None):
            super().__init__()
    VectorStore = FallbackComponent
    ModelManager = FallbackComponent
    QueryRouter = FallbackComponent
    PivotAnalyzer = FallbackComponent
    EmbeddingManager = FallbackComponent
    INDUSTRIAL_PROMPTS = {
        "Daily Report Summarization": "You are DigiTwin, an expert inspector. Analyze and summarize the provided reports.",
        "Safety Violation Analysis": "You are DigiTwin, a safety expert. Analyze safety violations and provide recommendations.",
        "Equipment Performance Review": "You are DigiTwin, an equipment specialist. Review equipment performance and provide insights.",
        "Compliance Assessment": "You are DigiTwin, a compliance expert. Assess regulatory compliance status.",
        "Risk Management Analysis": "You are DigiTwin, a risk specialist. Analyze operational risks and provide mitigation strategies.",
        "Pivot Table Analysis": "You are DigiTwin, a data analyst. Analyze pivot table data and provide operational insights."
    }



# --- UI CONFIG & STYLE (Retained from original app_rag.py) ---
st.set_page_config(page_title="DigiTwin RAG Forecast", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.cdnfonts.com/css/tw-cen-mt');
    * {
        font-family: 'Tw Cen MT', sans-serif !important;
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"]::before {
        content: "▶";
        font-size: 1.3rem;
        margin-right: 0.4rem;
    }
    /* Logo container removed - logo now in sidebar */
    </style>
""", unsafe_allow_html=True)

st.title("📊 DigiTwin - The Insp Nerdzx")

# --- Constants (from original app_rag.py) ---
USER_AVATAR = "https://raw.githubusercontent.com/achilela/vila_fofoka_analysis/9904d9a0d445ab0488cf7395cb863cce7621d897/USER_AVATAR.png"
BOT_AVATAR = "https://raw.githubusercontent.com/achilela/vila_fofoka_analysis/991f4c6e4e1dc7a8e24876ca5aae5228bcdb4dba/Ataliba_Avatar.jpg"

# --- Enhanced System State Management ---
@st.cache_resource
def initialize_components():
    """Initialize all system components with error handling"""
    try:
        logger.info("Initializing DigiTwin components...")
        
        # Initialize components
        components = {}
        
        # Database manager
        components['database_manager'] = DatabaseManager()
        
        # Cache manager
        components['cache_manager'] = CacheManager()
        
        # Embedding manager
        components['embedding_manager'] = EmbeddingManager()
        
        # Vector store
        components['vector_store'] = VectorStore()
        
        # Document processor
        config = Config()
        components['document_processor'] = DocumentProcessor(config)
        
        # Model manager
        components['model_manager'] = ModelManager()
        
        # Query router
        components['query_router'] = QueryRouter()
        
        # Pivot analyzer
        components['pivot_analyzer'] = PivotAnalyzer()
        
        logger.info("All components initialized successfully")
        return components
        
    except Exception as e:
        logger.error(f"Component initialization failed: {str(e)}")
        try:
            error_handler = ErrorHandler()
            error_handler.handle_error(e, "Component initialization failed")
        except:
            pass
        return {}

# Initialize components
if 'components' not in st.session_state:
    st.session_state.components = initialize_components()

# --- State Management (from original app_rag.py) ---
class AppState:
    @staticmethod
    def initialize():
        state_defaults = {
            "vectorstore": None,
            "chat_history": [],
            "model_intro_done": False,
            "current_model": None,
            "current_prompt": None,
            "last_processed": None,
            "processed_documents": None,
            "pivot_data": None,
            "pivot_summary": "",
            "query_history": [],
            "analysis_results": {}
        }
        for key, val in state_defaults.items():
            if key not in st.session_state:
                st.session_state[key] = val

AppState.initialize()

# --- Enhanced Response Generation with Multiple Models ---
def generate_response(prompt):
    """Enhanced response generation with multiple AI models"""
    try:
        components = st.session_state.components
        if not components:
            yield "⚠️ System components not initialized"
            return
        
        # Create enhanced context
        context = ""
        if st.session_state.vectorstore:
            try:
                # Retrieve relevant documents
                relevant_docs = components['vector_store'].similarity_search(prompt, k=5)
                context = "\n\n".join([doc.page_content for doc in relevant_docs])
            except Exception as e:
                logger.warning(f"Context retrieval failed: {str(e)}")
        
        # Add pivot table context if available
        if st.session_state.pivot_data:
            try:
                pivot_context = components['pivot_analyzer'].get_relevant_data(prompt)
                if pivot_context:
                    context += f"\n\nRelevant Data:\n{pivot_context}"
            except Exception as e:
                logger.warning(f"Pivot context retrieval failed: {str(e)}")
        
        # Select appropriate prompt
        system_prompt = INDUSTRIAL_PROMPTS.get(st.session_state.current_prompt, INDUSTRIAL_PROMPTS["Daily Report Summarization"])
        
        # Generate response using model manager
        model_alias = st.session_state.current_model
        
        # Model mapping
        model_map = {
            "EE Smartest Agent": "together",
            "JI Divine Agent": "deepseek", 
            "EdJa-Valonys": "cerebras",
            "XAI Inspector": "xai",
            "Valonys Llama": "local"
        }
        
        model_provider = model_map.get(model_alias, "xai")
        
        try:
            # Generate streaming response
            response_text = ""
            for chunk in components['model_manager'].generate_streaming_response(
                prompt, context, system_prompt, model_provider
            ):
                response_text += chunk
                yield f"<span style='font-family:Tw Cen MT'>{chunk}</span>"
                
        except Exception as model_error:
            logger.error(f"Model generation failed: {str(model_error)}")
            # Fallback response
            fallback_response = f"Based on your query '{prompt}', I can provide analysis once the AI models are properly configured. The system has processed your documents and is ready to provide insights."
            
            for word in fallback_response.split():
                yield f"<span style='font-family:Tw Cen MT'>{word} </span>"
                time.sleep(0.02)
                
    except Exception as e:
        error_msg = f"<span style='color:red'>⚠️ Error: {str(e)}</span>"
        logger.error(f"Response generation failed: {str(e)}")
        yield error_msg

# --- Enhanced Document Processing ---
def process_uploaded_files(uploaded_files, file_type):
    """Process uploaded files with enhanced capabilities"""
    try:
        components = st.session_state.components
        if not components:
            st.sidebar.error("System components not initialized")
            return
        
        with st.spinner(f"Processing {file_type} files..."):
            if file_type == "PDF":
                # Process PDFs
                processed_docs = components['document_processor'].process_files(uploaded_files)
                if processed_docs:
                    # Create enhanced vector store
                    st.session_state.vectorstore = components['vector_store'].create_enhanced_store(processed_docs)
                    st.session_state.processed_documents = processed_docs
                    st.sidebar.success(f"{len(processed_docs)} reports indexed with enhanced features.")
                else:
                    st.sidebar.error("Failed to process PDF files")
            else:
                # Process Excel files
                excel_docs = components['document_processor'].process_files(uploaded_files)
                if excel_docs:
                    st.session_state.vectorstore = components['vector_store'].create_enhanced_store(excel_docs)
                    st.session_state.processed_documents = excel_docs
                    
                    # Create pivot tables for analysis
                    try:
                        pivot_data = components['pivot_analyzer'].process_excel_files(uploaded_files)
                        if pivot_data:
                            st.session_state.pivot_data = pivot_data
                            
                            # Generate pivot summary
                            pivot_summary = components['pivot_analyzer'].generate_summary(pivot_data)
                            st.session_state.pivot_summary = pivot_summary
                            
                            st.sidebar.success(f"{len(excel_docs)} notification chunks indexed. Pivot tables created.")
                        else:
                            st.sidebar.success(f"{len(excel_docs)} Excel documents processed.")
                    except Exception as pivot_error:
                        logger.warning(f"Pivot table creation failed: {str(pivot_error)}")
                        st.sidebar.success(f"{len(excel_docs)} Excel documents processed (pivot analysis unavailable).")
                else:
                    st.sidebar.error("Failed to process Excel files")
                    
    except Exception as e:
        logger.error(f"File processing error: {str(e)}")
        st.sidebar.error(f"Processing error: {str(e)}")
        try:
            error_handler = ErrorHandler()
            error_handler.handle_error(e, "File processing failed")
        except:
            pass

# --- Main UI Components (Original Layout from app_rag.py) ---
with st.sidebar:
    # Logo at the top of sidebar (using ValonyLabs logo)
    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 20px;">
            <img src="https://github.com/valonys/DigiTwin/blob/29dd50da95bec35a5abdca4bdda1967f0e5efff6/ValonyLabs_Logo.png?raw=true" width="120" style="margin-bottom: 10px;">
            <h3 style="margin: 0; color: #1f77b4;">DigiTwin</h3>
            <p style="margin: 0; font-size: 0.8em; color: #666;">The Insp Nerdzx</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("---")  # Separator line
    
    # Model selection with enhanced options
    model_alias = st.selectbox("Choose your AI Agent", [
        "EE Smartest Agent", "JI Divine Agent", "EdJa-Valonys", "XAI Inspector", "Valonys Llama"
    ])
    
    # File type selection
    file_type = st.radio("Select file type", ["PDF", "Excel"])
    
    # File uploader
    if file_type == "PDF":
        uploaded_files = st.file_uploader("📄 Upload up to 10 PDF reports", type=["pdf"], accept_multiple_files=True)
    else:
        uploaded_files = st.file_uploader("📊 Upload Excel file", type=["xlsx", "xls"], accept_multiple_files=False)
    
    # Enhanced prompt selection
    prompt_type = st.selectbox("Select the Task Type", list(INDUSTRIAL_PROMPTS.keys()))

# --- Pivot Table Display (from original app_rag.py) ---
if 'pivot_summary' in st.session_state and st.session_state.pivot_summary:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Data Analysis Summary")
    st.sidebar.markdown(st.session_state.pivot_summary)

# --- Enhanced Document Processing ---
if uploaded_files:
    process_uploaded_files(uploaded_files, file_type)

# --- CHAT INTERFACE (Original from app_rag.py) ---

# Agent Introduction Logic
if not st.session_state.model_intro_done or \
   st.session_state.current_model != model_alias or \
   st.session_state.current_prompt != prompt_type:
    
    agent_intros = {
        "EE Smartest Agent": "💡 EE Agent Activated — Pragmatic & Smart",
        "JI Divine Agent": "✨ JI Agent Activated — DeepSeek Reasoning", 
        "EdJa-Valonys": "⚡ EdJa Agent Activated — Cerebras Speed",
        "XAI Inspector": "🔍 XAI Inspector — Qwen Custom Fine-tune",
        "Valonys Llama": "🦙 Valonys Llama — LLaMA3-Based Reasoning"
    }
    
    intro_message = agent_intros.get(model_alias, "🤖 AI Agent Activated")
    st.session_state.chat_history.append({
        "role": "assistant", 
        "content": intro_message,
        "timestamp": time.time()
    })
    st.session_state.model_intro_done = True
    st.session_state.current_model = model_alias
    st.session_state.current_prompt = prompt_type
    logger.info(f"Switched to model: {model_alias} with prompt: {prompt_type}")

# Display Chat History with Performance Metrics (Original Style)
for msg in st.session_state.chat_history:
    with st.chat_message(
        msg["role"], 
        avatar=USER_AVATAR if msg["role"] == "user" else BOT_AVATAR
    ):
        # Add subtle timestamp for production debugging
        timestamp = ""
        if "timestamp" in msg:
            timestamp = f"<small style='color:#888;float:right;'>\
            {time.strftime('%H:%M:%S', time.localtime(msg['timestamp']))}</small>"
        
        st.markdown(f"{msg['content']}{timestamp}", unsafe_allow_html=True)

# Chat Input with Enhanced Features (Original Style)
if prompt := st.chat_input("Ask a summary or forecast about the reports..."):
    # Validate input before processing
    if len(prompt.strip()) < 3:
        st.warning("Please enter a more detailed question")
        st.stop()
    
    # Add user message to history with timestamp
    st.session_state.chat_history.append({
        "role": "user",
        "content": prompt,
        "timestamp": time.time()
    })
    
    # Display user message
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)
    
    # Generate and stream response
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        response_placeholder = st.empty()
        full_response = ""
        start_time = time.time()
        
        try:
            for chunk in generate_response(prompt):
                full_response += chunk
                response_placeholder.markdown(
                    f"{full_response}▌", 
                    unsafe_allow_html=True
                )
            
            # Final render
            response_placeholder.markdown(full_response, unsafe_allow_html=True)
            
            # Log performance metrics
            duration = time.time() - start_time
            logger.info(f"Generated response in {duration:.2f}s for prompt: {prompt[:50]}...")
            
            # Add to history with metadata
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": full_response,
                "timestamp": time.time(),
                "metadata": {
                    "response_time": duration,
                    "model": st.session_state.current_model,
                    "prompt_type": st.session_state.current_prompt
                }
            })
            
        except Exception as e:
            error_msg = f"<span style='color:red'>⚠️ System Error: Please try again later</span>"
            response_placeholder.markdown(error_msg, unsafe_allow_html=True)
            logger.error(f"Response generation failed: {str(e)}")
            try:
                error_handler = ErrorHandler()
                error_handler.handle_error(e, "Response generation failed")
            except:
                pass

if __name__ == "__main__":
    logger.info("DigiTwin application started with enhanced capabilities")