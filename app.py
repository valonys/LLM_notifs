import streamlit as st
import os
import time
import json
import logging
from dotenv import load_dotenv
try:
    from PyPDF2 import PdfReader
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("Warning: PyPDF2 not available")

try:
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain.schema import Document as LCDocument
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("Warning: LangChain components not available")

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: Transformers not available")

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI not available")

try:
    from cerebras.cloud.sdk import Cerebras
    CEREBRAS_AVAILABLE = True
except ImportError:
    CEREBRAS_AVAILABLE = False
    print("Warning: Cerebras SDK not available")

try:
    from vector_store import VectorStore, SimpleTextStore
    VECTOR_STORE_AVAILABLE = True
except ImportError:
    VECTOR_STORE_AVAILABLE = False
    print("Warning: Vector store not available")

try:
    from cachetools import TTLCache
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False
    print("Warning: Cache tools not available")

try:
    import sentry_sdk
    SENTRY_AVAILABLE = True
except ImportError:
    SENTRY_AVAILABLE = False

try:
    from prometheus_client import start_http_server, Counter  
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# Try to import torch, but handle the case where it's not available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Using CPU for embeddings.")

# Simple fallback embedding class for when sentence-transformers is not available
class SimpleEmbeddings:
    def __init__(self, model_name="simple"):
        self.model_name = model_name
    
    def embed_documents(self, texts):
        # Simple hash-based embeddings as fallback
        import hashlib
        embeddings = []
        for text in texts:
            # Create a simple hash-based embedding
            hash_obj = hashlib.md5(text.encode())
            embedding = [int(hash_obj.hexdigest()[i:i+2], 16) / 255.0 for i in range(0, 32, 2)]
            # Pad to 384 dimensions (same as MiniLM)
            while len(embedding) < 384:
                embedding.extend(embedding[:min(384-len(embedding), len(embedding))])
            embeddings.append(embedding[:384])
        return embeddings
    
    def embed_query(self, text):
        return self.embed_documents([text])[0]

# --- Configuration & Monitoring ---
load_dotenv()

# Initialize monitoring
REQUEST_COUNTER = None
if SENTRY_AVAILABLE:
    sentry_dsn = os.getenv('SENTRY_DSN')
    if sentry_dsn:
        sentry_sdk.init(sentry_dsn)

# Start Prometheus metrics server on a different port to avoid conflicts
if PROMETHEUS_AVAILABLE:
    try:
        start_http_server(8001)  # Changed from 8000 to 8001
        REQUEST_COUNTER = Counter('app_requests', 'Total API requests')
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"Warning: Prometheus metrics server port 8001 is in use. Metrics will not be available.")
            REQUEST_COUNTER = None
        else:
            raise e

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

# Initialize caching
if CACHE_AVAILABLE:
    response_cache = TTLCache(maxsize=1000, ttl=3600)  # 1 hour cache
else:
    response_cache = {}  # Simple dict fallback

# --- UI CONFIG & STYLE (Retained from original) ---
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

# Logo moved to sidebar

st.title("📊 DigiTwin - The Insp Nerdzx")

# --- Constants ---
USER_AVATAR = "https://raw.githubusercontent.com/achilela/vila_fofoka_analysis/9904d9a0d445ab0488cf7395cb863cce7621d897/USER_AVATAR.png"
BOT_AVATAR = "https://raw.githubusercontent.com/achilela/vila_fofoka_analysis/991f4c6e4e1dc7a8e24876ca5aae5228bcdb4dba/Ataliba_Avatar.jpg"

# --- Enhanced System Prompts ---
PROMPTS = {
    "Daily Report Summarization": """You are DigiTwin, an expert inspector with deep knowledge of industrial processes, safety protocols, and regulatory compliance. Your role is to analyze daily inspection reports and provide comprehensive summaries that highlight:

1. **Critical Findings**: Any safety violations, equipment malfunctions, or compliance issues that require immediate attention
2. **Trend Analysis**: Patterns or recurring issues that may indicate systemic problems
3. **Recommendations**: Actionable steps to address identified issues and improve safety/compliance
4. **Risk Assessment**: Evaluation of potential risks and their severity levels
5. **Compliance Status**: Overall compliance with relevant regulations and standards

Please provide clear, professional summaries that can be used by management for decision-making and by field teams for immediate action.""",
    
    "Safety Violation Analysis": """You are DigiTwin, a safety expert specializing in industrial safety analysis. Your task is to identify and analyze safety violations from inspection reports. Focus on:

1. **Violation Classification**: Categorize violations by severity (Critical, Major, Minor)
2. **Root Cause Analysis**: Identify underlying causes and contributing factors
3. **Corrective Actions**: Recommend specific corrective and preventive actions
4. **Regulatory References**: Cite relevant safety standards and regulations
5. **Timeline Assessment**: Evaluate urgency and establish priority timelines for remediation

Ensure all analysis is thorough, factual, and actionable for safety management teams.""",

    "Equipment Performance Review": """You are DigiTwin, a maintenance and reliability engineer with expertise in industrial equipment performance analysis. Your mission is to evaluate equipment performance data and provide insights on:

1. **Performance Metrics**: Analysis of key performance indicators and operational efficiency
2. **Maintenance Needs**: Identification of required maintenance activities and schedules
3. **Reliability Assessment**: Evaluation of equipment reliability and failure patterns
4. **Optimization Opportunities**: Recommendations for performance improvements
5. **Cost-Benefit Analysis**: Economic evaluation of maintenance and upgrade options

Provide technical, data-driven recommendations that support optimal equipment performance and cost-effectiveness.""",

    "Compliance Assessment": """You are DigiTwin, a compliance specialist with extensive knowledge of industrial regulations, standards, and best practices. Your objective is to assess compliance status and provide guidance on:

1. **Regulatory Adherence**: Evaluation against applicable regulations and standards
2. **Gap Analysis**: Identification of compliance gaps and non-conformities
3. **Remediation Plans**: Development of action plans to address compliance issues
4. **Documentation Review**: Assessment of required documentation and record-keeping
5. **Audit Readiness**: Preparation recommendations for regulatory audits

Ensure all assessments are accurate, comprehensive, and aligned with current regulatory requirements.""",

    "Custom Analysis": """You are DigiTwin, an industrial intelligence specialist capable of performing comprehensive analysis across multiple domains including safety, maintenance, compliance, and operational efficiency. Adapt your analysis based on the specific context and requirements of the provided data, focusing on actionable insights and recommendations that support informed decision-making."""
}

# --- Model Configuration ---
@st.cache_data
def get_model_configs():
    return {
        "XAI Grok Beta": {
            "provider": "xai",
            "model_name": "grok-beta",
            "temperature": 0.7,
            "max_tokens": 4000
        },
        "XAI Grok 2 Latest": {
            "provider": "xai", 
            "model_name": "grok-2-latest",
            "temperature": 0.7,
            "max_tokens": 4000
        },
        "XAI Grok 2 1212": {
            "provider": "xai",
            "model_name": "grok-2-1212", 
            "temperature": 0.7,
            "max_tokens": 4000
        },
        "XAI Grok 2 Vision": {
            "provider": "xai",
            "model_name": "grok-2-vision-1212",
            "temperature": 0.7,
            "max_tokens": 4000
        },
        "DeepSeek Chat": {
            "provider": "deepseek",
            "model_name": "deepseek-chat",
            "temperature": 0.7,
            "max_tokens": 4000
        },
        "Cerebras Llama 3.1 70B": {
            "provider": "cerebras",
            "model_name": "llama3.1-70b",
            "temperature": 0.7,
            "max_tokens": 4000
        },
        "Cerebras Llama 3.1 8B": {
            "provider": "cerebras", 
            "model_name": "llama3.1-8b",
            "temperature": 0.7,
            "max_tokens": 4000
        }
    }

# --- Database & Vector Store Initialization ---
@st.cache_resource
def initialize_vector_store():
    """Initialize vector store with database integration"""
    if VECTOR_STORE_AVAILABLE:
        return VectorStore()
    else:
        # Simple fallback when vector store is not available
        class FallbackVectorStore:
            def get_cached_files(self):
                return []
            def load_cached_file(self, filename):
                return None
            def add_documents(self, docs):
                pass
            def process_excel_to_documents(self, files):
                return []
            def create_notification_pivot_tables(self):
                return {}
            def get_relevant_documents(self, query, k=5):
                return []
        return FallbackVectorStore()

vector_store = initialize_vector_store()

# --- API Functions ---
def get_api_key(provider):
    """Get API key for the specified provider"""
    key_mapping = {
        "xai": "XAI_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY", 
        "cerebras": "CEREBRAS_API_KEY"
    }
    return os.getenv(key_mapping.get(provider))

def create_client(provider, api_key):
    """Create API client for the specified provider"""
    if provider == "xai" and OPENAI_AVAILABLE:
        return openai.OpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1"
        )
    elif provider == "deepseek" and OPENAI_AVAILABLE:
        return openai.OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
    elif provider == "cerebras" and CEREBRAS_AVAILABLE:
        return Cerebras(api_key=api_key)
    else:
        raise ValueError(f"Unsupported provider: {provider} or required libraries not available")

def stream_response(client, provider, model_name, messages, temperature, max_tokens):
    """Stream response from the specified model"""
    try:
        if provider == "cerebras":
            stream = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
        else:  # XAI and DeepSeek use OpenAI-compatible interface
            stream = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
        
        for chunk in stream:
            if hasattr(chunk, 'choices') and chunk.choices:
                delta = chunk.choices[0].delta
                if hasattr(delta, 'content') and delta.content:
                    yield delta.content
                    
    except Exception as e:
        error_msg = f"<span style='color:red'>⚠️ Error: {str(e)}</span>"
        logger.error(f"Streaming failed for {provider}/{model_name}: {str(e)}")
        yield error_msg

# --- Enhanced Document Processing ---
class DocumentProcessor:
    @staticmethod
    @st.cache_resource(show_spinner=False)
    def process_pdfs(_files):
        if not PDF_AVAILABLE:
            st.error("PDF processing not available. Please install PyPDF2.")
            return []
        if not LANGCHAIN_AVAILABLE:
            st.error("LangChain not available for document processing.")
            return []
        
        try:
            parsed_docs = []
            for f in _files:
                with st.spinner(f"Processing {f.name}..."):
                    reader = PdfReader(f)
                    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
                    parsed_docs.append(LCDocument(page_content=text, metadata={"name": f.name}))
            return parsed_docs
        except Exception as e:
            logger.error(f"PDF processing failed: {str(e)}")
            raise

    @staticmethod  
    @st.cache_resource(show_spinner=False)
    def process_text_files(_files):
        if not LANGCHAIN_AVAILABLE:
            st.error("LangChain not available for document processing.")
            return []
            
        try:
            parsed_docs = []
            for f in _files:
                with st.spinner(f"Processing {f.name}..."):
                    text = str(f.read(), "utf-8")
                    parsed_docs.append(LCDocument(page_content=text, metadata={"name": f.name}))
            return parsed_docs
        except Exception as e:
            logger.error(f"Text file processing failed: {str(e)}")
            raise

# --- Sidebar Configuration ---
with st.sidebar:
    # Add logo at the top of sidebar
    st.image("https://raw.githubusercontent.com/achilela/vila_fofoka_analysis/4d7f7962ad5b1b0b89e3065cd9d8d0a398b09b8b/ValonyLabs_Logo.png", width=200)
    
    st.header("⚙️ Configuration")
    
    # Model selection with display name on canvas
    model_configs = get_model_configs()
    selected_model = st.selectbox(
        "🤖 Select Model:", 
        options=list(model_configs.keys()),
        index=0
    )
    
    # Analysis type selection
    selected_prompt = st.selectbox(
        "📋 Analysis Type:",
        options=list(PROMPTS.keys()),
        index=0
    )
    
    # Advanced settings in expander
    with st.expander("🔧 Advanced Settings"):
        temperature = st.slider("Temperature", 0.1, 1.0, 0.7, 0.1)
        max_tokens = st.slider("Max Tokens", 100, 8000, 4000, 100)
    
    # Cached files dropdown for database integration
    st.header("📁 Cached Files")
    cached_files = vector_store.get_cached_files()
    if cached_files:
        selected_cached_file = st.selectbox(
            "Load cached file:",
            options=["None"] + [f["filename"] for f in cached_files],
            key="cached_file_selector"
        )
        
        if selected_cached_file != "None":
            if st.button("Load Selected File", key="load_cached"):
                # Load cached file from database
                cached_data = vector_store.load_cached_file(selected_cached_file)
                if cached_data:
                    st.success(f"Loaded {selected_cached_file} from cache")
                    st.session_state['cached_file_loaded'] = selected_cached_file
                else:
                    st.error("Failed to load cached file")
    else:
        st.info("No cached files available")

# Display selected model name on main canvas
st.info(f"🤖 **Active Model**: {selected_model}")

# --- File Upload Section ---
st.header("📄 Document Upload")

file_types = ["PDF", "Text Files", "Excel Files"] 
selected_file_type = st.selectbox("Select file type:", file_types)

uploaded_files = None
if selected_file_type == "PDF":
    uploaded_files = st.file_uploader(
        "Upload PDF files", 
        type=["pdf"], 
        accept_multiple_files=True
    )
elif selected_file_type == "Text Files":
    uploaded_files = st.file_uploader(
        "Upload text files", 
        type=["txt"], 
        accept_multiple_files=True
    )
elif selected_file_type == "Excel Files":
    uploaded_files = st.file_uploader(
        "Upload Excel files", 
        type=["xlsx", "xls"], 
        accept_multiple_files=True
    )

# Process uploaded files
if uploaded_files:
    try:
        if selected_file_type == "PDF":
            documents = DocumentProcessor.process_pdfs(uploaded_files)
            vector_store.add_documents(documents)
            st.success(f"✅ Processed {len(documents)} PDF documents")
            
        elif selected_file_type == "Text Files":
            documents = DocumentProcessor.process_text_files(uploaded_files)
            vector_store.add_documents(documents) 
            st.success(f"✅ Processed {len(documents)} text documents")
            
        elif selected_file_type == "Excel Files":
            # Use vector_store method for Excel processing with database caching
            documents = vector_store.process_excel_to_documents(uploaded_files)
            st.success(f"✅ Processed {len(uploaded_files)} Excel files with database caching")
            
            # Show pivot table creation option
            if st.button("🔄 Create Notification Pivot Tables"):
                with st.spinner("Creating pivot tables..."):
                    pivot_results = vector_store.create_notification_pivot_tables()
                    if pivot_results:
                        st.success("✅ Pivot tables created and cached in database")
                        
                        # Display some pivot results
                        for table_name, data in list(pivot_results.items())[:3]:  # Show first 3 tables
                            st.subheader(f"📊 {table_name}")
                            if isinstance(data, dict) and 'data' in data:
                                st.json(data['data'])
                            else:
                                st.write(data)
                    else:
                        st.warning("No pivot tables were created")
        
    except Exception as e:
        st.error(f"❌ Error processing files: {str(e)}")
        logger.error(f"File processing error: {str(e)}")

# --- Chat Interface ---
st.header("💬 Analysis Chat")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about your documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)
    
    # Generate AI response
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        response_placeholder = st.empty()
        
        try:
            # Get model configuration
            config = model_configs[selected_model]
            api_key = get_api_key(config["provider"])
            
            if not api_key:
                st.error(f"❌ API key not found for {config['provider']}. Please set the appropriate environment variable.")
            else:
                # Create client and prepare messages
                client = create_client(config["provider"], api_key)
                
                # Get relevant context from vector store
                if hasattr(vector_store, 'get_relevant_documents'):
                    context_docs = vector_store.get_relevant_documents(prompt, k=5)
                    context = "\n\n".join([doc.page_content[:1000] for doc in context_docs])
                else:
                    context = "No specific context available."
                
                # Prepare messages with system prompt and context
                messages = [
                    {"role": "system", "content": PROMPTS[selected_prompt]},
                    {"role": "user", "content": f"Context from documents:\n{context}\n\nUser question: {prompt}"}
                ]
                
                # Stream response
                full_response = ""
                for chunk in stream_response(
                    client, 
                    config["provider"], 
                    config["model_name"], 
                    messages, 
                    temperature, 
                    max_tokens
                ):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
                
                response_placeholder.markdown(full_response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
        except Exception as e:
            error_msg = f"❌ Error generating response: {str(e)}"
            st.error(error_msg)
            logger.error(f"Chat error: {str(e)}")

# Add footer with ValonyLabs branding
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; font-size: 14px;'>"
    "DigiTwin - Industrial Intelligence System | Powered by ValonyLabs"
    "</div>", 
    unsafe_allow_html=True
)

logger.info("DigiTwin application started successfully")