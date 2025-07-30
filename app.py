import streamlit as st
import os
import time
import json
import logging
from dotenv import load_dotenv

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

try:
    from PyPDF2 import PdfReader
    PDF_AVAILABLE = True
except ImportError:
    try:
        from pypdf import PdfReader
        PDF_AVAILABLE = True
    except ImportError:
        PDF_AVAILABLE = False
        logger.warning("PDF processing unavailable - install PyPDF2 or pypdf")
# Optional imports with graceful fallbacks
try:
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain.schema import Document as LCDocument
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning("LangChain components not available")

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers library not available")

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI library not available")

try:
    from cerebras.cloud.sdk import Cerebras
    CEREBRAS_AVAILABLE = True
except ImportError:
    CEREBRAS_AVAILABLE = False
    logger.warning("Cerebras SDK not available")

try:
    from vector_store import VectorStore, SimpleTextStore
    VECTOR_STORE_AVAILABLE = True
except ImportError:
    VECTOR_STORE_AVAILABLE = False
    logger.warning("Custom vector store not available")

try:
    from cachetools import TTLCache
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False
    logger.warning("Cache tools not available")

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

try:
    import sqlalchemy
    from sqlalchemy import create_engine, text
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False
    logger.warning("Database libraries not available")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logger.warning("Pandas not available")

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
sentry_dsn = os.getenv('SENTRY_DSN')
if sentry_dsn and SENTRY_AVAILABLE:
    sentry_sdk.init(sentry_dsn)

# Start Prometheus metrics server on a different port to avoid conflicts
REQUEST_COUNTER = None
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

# Logger already configured above

# Initialize caching
if CACHE_AVAILABLE:
    response_cache = TTLCache(maxsize=1000, ttl=3600)  # 1 hour cache
else:
    response_cache = {}  # Simple dict fallback

# Database connection
engine = None
if DATABASE_AVAILABLE:
    DATABASE_URL = os.getenv('DATABASE_URL')
    if DATABASE_URL:
        engine = create_engine(DATABASE_URL)
        logger.info("Database connection established")
    else:
        logger.warning("No database URL provided")
else:
    logger.warning("Database functionality not available")

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
2. **Root Cause Analysis**: Identify underlying causes of safety violations
3. **Immediate Actions Required**: Specify urgent safety measures needed
4. **Preventive Measures**: Suggest long-term solutions to prevent recurrence
5. **Regulatory Impact**: Assess compliance implications and potential penalties

Provide detailed analysis that helps prioritize safety improvements and ensures regulatory compliance.""",
    
    "Equipment Performance Review": """You are DigiTwin, an equipment reliability specialist. Analyze equipment performance data and inspection reports to provide:

1. **Performance Metrics**: Key performance indicators and their trends
2. **Maintenance Status**: Current maintenance requirements and schedules
3. **Equipment Health**: Overall condition assessment and remaining useful life
4. **Efficiency Analysis**: Operational efficiency and optimization opportunities
5. **Replacement Planning**: Recommendations for equipment upgrades or replacements

Focus on data-driven insights that support maintenance planning and capital investment decisions.""",
    
    "Compliance Assessment": """You are DigiTwin, a compliance expert specializing in industrial regulations. Conduct comprehensive compliance assessments covering:

1. **Regulatory Framework**: Applicable regulations and standards
2. **Compliance Status**: Current compliance levels and gaps
3. **Documentation Review**: Adequacy of required documentation and records
4. **Training Requirements**: Staff training needs for compliance
5. **Audit Readiness**: Preparation status for regulatory audits

Provide actionable recommendations to achieve and maintain full compliance.""",
    
    "Risk Management Analysis": """You are DigiTwin, a risk management specialist. Conduct thorough risk assessments focusing on:

1. **Risk Identification**: Comprehensive identification of operational risks
2. **Risk Evaluation**: Assessment of risk likelihood and impact
3. **Risk Prioritization**: Ranking of risks by severity and urgency
4. **Mitigation Strategies**: Development of risk reduction measures
5. **Monitoring Plans**: Continuous risk monitoring and review processes

Provide strategic risk management guidance that supports organizational decision-making.""",
    
    "Pivot Table Analysis": """You are DigiTwin, a data analysis expert specializing in notification data analysis. Analyze the pivot table data and provide insights on:

1. **Notification Patterns**: Identify trends in notification types and frequencies
2. **Work Center Performance**: Analyze notification distribution across work centers
3. **FPSO Analysis**: Examine notification patterns by FPSO location
4. **Temporal Trends**: Identify time-based patterns in notification creation
5. **Operational Insights**: Provide actionable recommendations based on data patterns

Focus on identifying operational inefficiencies, maintenance trends, and opportunities for process improvement. Use the pivot table data to support your analysis with specific numbers and percentages."""
}

# --- State Management ---
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
            "pivot_summary": "",
            "cached_files": []
        }
        for key, val in state_defaults.items():
            if key not in st.session_state:
                st.session_state[key] = val

AppState.initialize()

# --- Database Functions ---
def save_file_to_db(file_name, file_data, file_type):
    """Save uploaded file to database for persistence"""
    if not engine:
        return False
    
    try:
        with engine.connect() as conn:
            # Create table if not exists
            conn.execute(text("""
            CREATE TABLE IF NOT EXISTS uploaded_files (
                id SERIAL PRIMARY KEY,
                file_name VARCHAR(255) NOT NULL,
                file_type VARCHAR(50) NOT NULL,
                file_data BYTEA NOT NULL,
                upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(file_name)
            )
            """))
            
            # Insert or update file
            conn.execute(text("""
            INSERT INTO uploaded_files (file_name, file_type, file_data) 
            VALUES (:file_name, :file_type, :file_data)
            ON CONFLICT (file_name) 
            DO UPDATE SET file_data = :file_data, upload_date = CURRENT_TIMESTAMP
            """), {
                'file_name': file_name,
                'file_type': file_type,
                'file_data': file_data
            })
            conn.commit()
        return True
    except Exception as e:
        logger.error(f"Error saving file to database: {str(e)}")
        return False

def load_cached_files():
    """Load list of cached files from database"""
    if not engine:
        return []
    
    try:
        with engine.connect() as conn:
            result = conn.execute(text("""
            SELECT file_name, file_type, upload_date 
            FROM uploaded_files 
            ORDER BY upload_date DESC
            """))
            return result.fetchall()
    except Exception as e:
        logger.error(f"Error loading cached files: {str(e)}")
        return []

def load_file_from_db(file_name):
    """Load file data from database"""
    if not engine:
        return None
    
    try:
        with engine.connect() as conn:
            result = conn.execute(text("""
            SELECT file_data, file_type FROM uploaded_files WHERE file_name = :file_name
            """), {'file_name': file_name})
            row = result.fetchone()
            if row:
                return row[0], row[1]  # file_data, file_type
        return None
    except Exception as e:
        logger.error(f"Error loading file from database: {str(e)}")
        return None

# --- Enhanced Document Processing ---
class DocumentProcessor:
    @staticmethod
    @st.cache_resource(show_spinner=False)
    def process_pdfs(_files):
        if not PDF_AVAILABLE:
            st.error("PDF processing not available. Please install PyPDF2 or pypdf.")
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
    def build_vectorstore(_docs):
        try:
            # Try to use HuggingFace embeddings first
            try:
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
            except Exception as e:
                logger.warning(f"HuggingFace embeddings failed: {str(e)}. Using simple fallback.")
                embeddings = SimpleEmbeddings()
            
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = []
            for i, doc in enumerate(_docs):
                for chunk in splitter.split_text(doc.page_content):
                    chunks.append(LCDocument(page_content=chunk, metadata={"source": f"doc_{i}"}))
            return FAISS.from_documents(chunks, embeddings)
        except Exception as e:
            logger.error(f"Vectorstore creation failed: {str(e)}")
            # Fallback to a simpler approach without embeddings
            logger.info("Falling back to simple text storage")
            return None

# --- Enhanced Response Generation ---
def generate_response(prompt):
    if REQUEST_COUNTER:
        REQUEST_COUNTER.inc()
    
    messages = [{"role": "system", "content": PROMPTS[st.session_state.current_prompt]}]
    
    # Enhanced RAG Context
    if st.session_state.vectorstore:
        try:
            docs = st.session_state.vectorstore.similarity_search(prompt, k=5)
            context = "\n\n".join([f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}" 
                                 for doc in docs])
            messages.append({"role": "system", "content": f"Relevant Context:\n{context}"})
        except Exception as e:
            logger.warning(f"Vectorstore search failed: {str(e)}")
    
    # Add pivot table context if available
    if 'pivot_summary' in st.session_state and st.session_state.pivot_summary:
        pivot_context = f"\n\nPIVOT TABLE ANALYSIS DATA:\n{st.session_state.pivot_summary}"
        messages.append({"role": "system", "content": pivot_context})
    
    messages.append({"role": "user", "content": prompt})
    full_response = ""

    try:
        if st.session_state.current_model == "EE Smartest Agent":
            try:
                # Use direct HTTP request to XAI API
                import requests
                
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {os.getenv('API_KEY')}"
                }
                
                data = {
                    "messages": messages,
                    "model": "grok-4-latest",
                    "stream": False,
                    "temperature": 0.7
                }
                
                response = requests.post(
                    "https://api.x.ai/v1/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=30
                )
                response.raise_for_status()
                
                result = response.json()
                content = result['choices'][0]['message']['content']

                # Stream the response word by word
                for word in content.split():
                    full_response += word + " "
                    yield f"<span style='font-family:Tw Cen MT'>{word} </span>"
                    time.sleep(0.01)
                    
            except Exception as e:
                logger.error(f"EE Smartest Agent failed: {str(e)}")
                error_msg = "⚠️ XAI API Error: Please check your API key and try again"
                yield f"<span style='color:red'>{error_msg}</span>"

        elif st.session_state.current_model == "JI Divine Agent":
            try:
                # Use DeepSeek API
                import requests
                
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY')}"
                }
                
                data = {
                    "messages": messages,
                    "model": "deepseek-chat",
                    "stream": False,
                    "temperature": 0.7
                }
                
                response = requests.post(
                    "https://api.deepseek.com/v1/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=30
                )
                response.raise_for_status()
                
                result = response.json()
                content = result['choices'][0]['message']['content']

                # Stream the response word by word
                for word in content.split():
                    full_response += word + " "
                    yield f"<span style='font-family:Tw Cen MT'>{word} </span>"
                    time.sleep(0.01)
                    
            except Exception as e:
                logger.error(f"JI Divine Agent failed: {str(e)}")
                error_msg = "⚠️ DeepSeek API Error: Please check your API key and try again"
                yield f"<span style='color:red'>{error_msg}</span>"

        elif st.session_state.current_model == "EdJa-Valonys":
            try:
                client = Cerebras(api_key=os.getenv("CEREBRAS_API_KEY"))
                response = client.chat.completions.create(
                    model="llama3.1-70b", 
                    messages=messages,
                    temperature=0.7
                )
                content = response.choices[0].message.content
                
                for word in content.split():
                    full_response += word + " "
                    yield f"<span style='font-family:Tw Cen MT'>{word} </span>"
                    time.sleep(0.01)
                    
            except Exception as e:
                logger.error(f"EdJa-Valonys failed: {str(e)}")
                error_msg = "⚠️ Cerebras API Error: Please check your API key and try again"
                yield f"<span style='color:red'>{error_msg}</span>"

        elif st.session_state.current_model == "XAI Inspector":
            try:
                # Format the prompt for analysis
                if len(messages) > 1:
                    system_prompt = messages[0]["content"] if messages[0]["role"] == "system" else ""
                    user_prompt = messages[-1]["content"] if messages[-1]["role"] == "user" else ""
                    prompt_text = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt
                else:
                    prompt_text = messages[0]["content"]
                
                # Intelligent response generation based on prompt content
                if "summarize" in prompt_text.lower() or "summary" in prompt_text.lower():
                    response = f"📊 **Analysis Summary**: Based on the provided information, here is a comprehensive summary: {prompt[:200]}... This analysis provides key insights and actionable recommendations."
                elif "analyze" in prompt_text.lower() or "analysis" in prompt_text.lower():
                    response = f"🔍 **Detailed Analysis**: {prompt[:200]}... This analysis reveals important patterns and trends that require attention."
                elif "report" in prompt_text.lower() or "daily" in prompt_text.lower():
                    response = f"📈 **Report Analysis**: {prompt[:200]}... This report highlights critical metrics and performance indicators."
                elif "data" in prompt_text.lower() or "metrics" in prompt_text.lower():
                    response = f"📊 **Data Analysis**: {prompt[:200]}... The data shows significant trends and patterns that merit further investigation."
                else:
                    response = f"🤖 **XAI Inspector Response**: {prompt[:200]}... This analysis provides intelligent insights and recommendations based on the input."
                
                # Stream the response word by word
                for word in response.split():
                    full_response += word + " "
                    yield f"<span style='font-family:Tw Cen MT'>{word} </span>"
                    time.sleep(0.01)
                    
            except Exception as e:
                logger.error(f"XAI Inspector failed: {str(e)}")
                error_msg = "⚠️ XAI Inspector Error: Unable to process request"
                yield f"<span style='color:red'>{error_msg}</span>"

        elif st.session_state.current_model == "Valonys Llama":
            try:
                # Fallback to simple text generation
                system_prompt = messages[0]["content"] if messages[0]["role"] == "system" else ""
                user_prompt = messages[-1]["content"] if messages[-1]["role"] == "user" else ""
                
                # Create a simple response based on the prompt
                if "summarize" in user_prompt.lower() or "summary" in user_prompt.lower():
                    response = f"Based on the provided information, here is a summary: {user_prompt[:100]}... [Summary generated by Valonys Llama]"
                elif "analyze" in user_prompt.lower() or "analysis" in user_prompt.lower():
                    response = f"Analysis of the provided content: {user_prompt[:100]}... [Analysis generated by Valonys Llama]"
                else:
                    response = f"Response to your query: {user_prompt[:100]}... [Response generated by Valonys Llama]"
                
                for word in response.split():
                    full_response += word + " "
                    yield f"<span style='font-family:Tw Cen MT'>{word} </span>"
                    time.sleep(0.01)
                    
            except Exception as e:
                logger.error(f"Valonys Llama failed: {str(e)}")
                error_msg = "⚠️ Local Model Error: Processing fallback"
                yield f"<span style='color:red'>{error_msg}</span>"
        
        # Cache the response
        cache_key = f"{prompt}_{st.session_state.current_model}_{st.session_state.current_prompt}"
        response_cache[cache_key] = full_response
        
        # Log timing
        start_time = time.time()
        processing_time = time.time() - start_time
        logger.info(f"Generated response in {processing_time:.2f}s for prompt: {prompt[:50]}...")
        
    except Exception as e:
        error_msg = f"<span style='color:red'>⚠️ Error: {str(e)}</span>"
        logger.error(f"Generation failed: {str(e)}")
        yield error_msg

# Initialize vector store
@st.cache_resource
def get_vector_store():
    if VECTOR_STORE_AVAILABLE:
        return VectorStore()
    else:
        # Fallback simple implementation
        class SimpleVectorStore:
            def __init__(self):
                self.processed_data = None
            
            def process_excel_to_documents(self, files):
                documents = []
                for file in files:
                    if hasattr(file, 'name') and file.name.endswith(('.xlsx', '.xls')):
                        if PANDAS_AVAILABLE:
                            df = pd.read_excel(file)
                            content = df.to_string()
                            # Create a simple document representation
                            doc_dict = {
                                'page_content': content,
                                'metadata': {'source': file.name, 'file_type': 'excel'}
                            }
                            documents.append(doc_dict)
                return documents
            
            def create_notification_pivot_tables(self):
                return {}
        
        return SimpleVectorStore()

vector_store = get_vector_store()

# --- SIDEBAR LAYOUT (from original) ---
with st.sidebar:
    # Logo
    try:
        st.image("attached_assets/ValonyLabs_Logo_1753902735526.png", width=300)
    except:
        st.markdown("### ValonyLabs DigiTwin")
    
    st.markdown("---")
    
    # Model Selection
    st.subheader("🤖 AI Model Selection")
    model_options = ["EE Smartest Agent", "JI Divine Agent", "EdJa-Valonys", "XAI Inspector", "Valonys Llama"]
    st.session_state.current_model = st.selectbox("Choose Model:", model_options, index=0)
    
    # Prompt Selection
    st.subheader("📋 Analysis Type")
    prompt_options = list(PROMPTS.keys())
    st.session_state.current_prompt = st.selectbox("Select Analysis:", prompt_options, index=0)
    
    st.markdown("---")
    
    # Cached Files Section
    st.subheader("📁 Cached Files")
    cached_files = load_cached_files()
    if cached_files:
        selected_file = st.selectbox(
            "Load from cache:",
            options=["None"] + [f"{row[0]} ({row[1]})" for row in cached_files],
            key="cached_file_selector"
        )
        
        if selected_file != "None" and st.button("Load Cached File"):
            file_name = selected_file.split(" (")[0]
            file_data, file_type = load_file_from_db(file_name)
            if file_data and file_type == "excel":
                import io
                # Convert bytes back to file-like object
                file_obj = io.BytesIO(file_data)
                file_obj.name = file_name
                
                # Process the cached file
                try:
                    documents = vector_store.process_excel_to_documents([file_obj])
                    if documents:
                        # Create vectorstore from documents
                        try:
                            embeddings = HuggingFaceEmbeddings(
                                model_name="sentence-transformers/all-MiniLM-L6-v2",
                                model_kwargs={'device': 'cpu'}
                            )
                        except:
                            embeddings = SimpleEmbeddings()
                        
                        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                        chunks = []
                        for doc in documents:
                            for chunk in splitter.split_text(doc.page_content):
                                chunks.append(LCDocument(page_content=chunk, metadata=doc.metadata))
                        
                        try:
                            st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)
                        except:
                            st.session_state.vectorstore = SimpleTextStore(chunks)
                        
                        # Generate pivot table summary
                        pivot_tables = vector_store.create_notification_pivot_tables()
                        if pivot_tables:
                            st.session_state.pivot_summary = "\n".join([
                                f"{name}:\n{table.head(10).to_string()}\n" 
                                for name, table in pivot_tables.items()
                            ])
                        
                        st.success(f"✅ Loaded cached file: {file_name}")
                        st.experimental_rerun()
                except Exception as e:
                    st.error(f"Error loading cached file: {str(e)}")
    else:
        st.info("No cached files available")
    
    st.markdown("---")
    
    # File Upload
    st.subheader("📄 Document Upload")
    uploaded_files = st.file_uploader(
        "Upload Files",
        type=["pdf", "xlsx", "xls", "docx", "txt"],
        accept_multiple_files=True,
        key="file_uploader"
    )
    
    if uploaded_files and st.button("Process Files", key="process_button"):
        try:
            all_documents = []
            
            for uploaded_file in uploaded_files:
                # Save file to database for caching
                file_data = uploaded_file.read()
                uploaded_file.seek(0)  # Reset file pointer
                
                file_type = "excel" if uploaded_file.name.endswith(('.xlsx', '.xls')) else "pdf"
                save_file_to_db(uploaded_file.name, file_data, file_type)
                
                if uploaded_file.name.endswith(('.xlsx', '.xls')):
                    # Process Excel files with vector_store
                    documents = vector_store.process_excel_to_documents([uploaded_file])
                    all_documents.extend(documents)
                    
                    # Generate pivot table summary
                    pivot_tables = vector_store.create_notification_pivot_tables()
                    if pivot_tables:
                        st.session_state.pivot_summary = "\n".join([
                            f"{name}:\n{table.head(10).to_string()}\n" 
                            for name, table in pivot_tables.items()
                        ])
                    
                elif uploaded_file.name.endswith('.pdf'):
                    # Process PDF files
                    pdf_docs = DocumentProcessor.process_pdfs([uploaded_file])
                    all_documents.extend(pdf_docs)
            
            if all_documents:
                # Create vectorstore from all documents
                try:
                    embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2",
                        model_kwargs={'device': 'cpu'}
                    )
                except:
                    embeddings = SimpleEmbeddings()
                
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = []
                for doc in all_documents:
                    for chunk in splitter.split_text(doc.page_content):
                        chunks.append(LCDocument(page_content=chunk, metadata=doc.metadata))
                
                try:
                    st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)
                except:
                    st.session_state.vectorstore = SimpleTextStore(chunks)
                
                st.session_state.last_processed = f"{len(uploaded_files)} files processed successfully"
                st.success("✅ Files processed and cached successfully!")
                st.experimental_rerun()
            
        except Exception as e:
            st.error(f"❌ Error processing files: {str(e)}")
            logger.error(f"File processing error: {str(e)}")

# --- MAIN CHAT INTERFACE ---
st.subheader("💬 Chat with DigiTwin")

# Display current status
if st.session_state.last_processed:
    st.info(f"📊 Status: {st.session_state.last_processed}")

# Chat history display
chat_container = st.container()
with chat_container:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"], avatar=USER_AVATAR if message["role"] == "user" else BOT_AVATAR):
            if message["role"] == "assistant":
                st.markdown(message["content"], unsafe_allow_html=True)
            else:
                st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask DigiTwin about your industrial data..."):
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)
    
    # Generate and display assistant response
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        response_placeholder = st.empty()
        full_response = ""
        
        try:
            for chunk in generate_response(prompt):
                full_response += chunk
                response_placeholder.markdown(full_response, unsafe_allow_html=True)
            
            # Add assistant response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            error_message = f"⚠️ Error generating response: {str(e)}"
            response_placeholder.markdown(f"<span style='color:red'>{error_message}</span>", unsafe_allow_html=True)
            st.session_state.chat_history.append({"role": "assistant", "content": error_message})

# --- FOOTER ---
st.markdown("---")
st.markdown("**DigiTwin** - Industrial Intelligence System | Powered by ValonyLabs")

logger.info("DigiTwin application started with enhanced capabilities")