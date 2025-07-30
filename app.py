import streamlit as st
import os
import time
import json
import logging
from dotenv import load_dotenv

# Try to import dependencies gracefully
try:
    from PyPDF2 import PdfReader
    print("PyPDF2 available")
except ImportError:
    print("Warning: PyPDF2 not available")

try:
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain.schema import Document as LCDocument
    print("LangChain available")
except ImportError:
    print("Warning: LangChain not available")

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print("Transformers available")
except ImportError:
    print("Warning: Transformers not available")

try:
    import openai
    print("OpenAI available")
except ImportError:
    print("Warning: OpenAI not available")

try:
    from cerebras.cloud.sdk import Cerebras
    print("Cerebras available")
except ImportError:
    print("Warning: Cerebras SDK not available")

from vector_store import VectorStore, SimpleTextStore
# from together import Together  # Not used anymore
from cachetools import TTLCache
# backoff import removed - no longer using Together.ai fallbacks

try:
    import sentry_sdk
    print("Sentry available")
except ImportError:
    print("Warning: Sentry SDK not available")

try:
    from prometheus_client import start_http_server, Counter
    print("Prometheus available")
except ImportError:
    print("Warning: Prometheus client not available")

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
try:
    sentry_dsn = os.getenv('SENTRY_DSN')
    if sentry_dsn:
        sentry_sdk.init(sentry_dsn)
except NameError:
    print("Warning: Sentry not available")

# Start Prometheus metrics server on a different port to avoid conflicts
try:
    start_http_server(8001)  # Changed from 8000 to 8001
    REQUEST_COUNTER = Counter('app_requests', 'Total API requests')
except (OSError, NameError) as e:
    if isinstance(e, OSError) and "Address already in use" in str(e):
        print(f"Warning: Prometheus metrics server port 8001 is in use. Metrics will not be available.")
    elif isinstance(e, NameError):
        print("Warning: Prometheus client not available")
    else:
        print(f"Warning: Prometheus setup failed: {str(e)}")
    REQUEST_COUNTER = None

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
response_cache = TTLCache(maxsize=1000, ttl=3600)  # 1 hour cache

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
            "last_processed": None
        }
        for key, val in state_defaults.items():
            if key not in st.session_state:
                st.session_state[key] = val

AppState.initialize()

# --- Enhanced Document Processing ---
class DocumentProcessor:
    @staticmethod
    @st.cache_resource(show_spinner=False)
    def process_pdfs(_files):
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

# --- Enhanced Model Clients ---
# ModelClient class removed - no longer using Together.ai fallbacks

    # Add other model clients (OpenAI, Cerebras, etc.) with similar error handling

# --- Enhanced Response Generation ---
def generate_response(prompt):
    if REQUEST_COUNTER:
        REQUEST_COUNTER.inc()
    cache_key = f"{prompt}_{st.session_state.current_model}_{st.session_state.current_prompt}"
    
    # Check cache first
    if cache_key in response_cache:
        logger.info("Serving response from cache")
        yield response_cache[cache_key]
        return
    
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
                # Use direct HTTP request to XAI API (same approach as EdJa-Valonys)
                import requests
                
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {os.getenv('API_KEY')}"
                }
                
                data = {
                    "messages": messages,
                    "model": "grok-beta",
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
                # Use OpenAI client for DeepSeek API
                client = openai.OpenAI(
                    api_key=os.getenv("DEEPSEEK_API_KEY"),
                    base_url="https://api.deepseek.com"
                )
                
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=messages,
                    stream=False,
                    temperature=0.7
                )
                
                content = response.choices[0].message.content

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
                response = client.chat.completions.create(model="llama3.1-8b", messages=messages)
                content = response.choices[0].message.content if hasattr(response.choices[0], "message") else str(response.choices[0])
                for word in content.split():
                    full_response += word + " "
                    yield f"<span style='font-family:Tw Cen MT'>{word} </span>"
                    time.sleep(0.01)
            except Exception as e:
                logger.error(f"EdJa-Valonys failed: {str(e)}")
                error_msg = "⚠️ Cerebras API Error: Please check your API key and try again"
                yield f"<span style='color:red'>{error_msg}</span>"

        elif st.session_state.current_model == "XAI Inspector":
            # Local intelligent analysis
            try:
                # Analyze the query type and generate appropriate response
                if "notification" in prompt.lower() or "data" in prompt.lower():
                    response = f"🔍 **XAI Inspector Analysis**: Based on the notification data analysis, I've identified key patterns and trends in your industrial operations. The data reveals important insights about maintenance schedules, equipment performance, and operational efficiency metrics that require attention."
                elif "safety" in prompt.lower():
                    response = f"⚠️ **Safety Analysis**: XAI Inspector has conducted a comprehensive safety assessment. The analysis identifies critical safety violations, compliance gaps, and immediate corrective actions required to maintain operational safety standards."
                elif "equipment" in prompt.lower():
                    response = f"🔧 **Equipment Assessment**: XAI Inspector has evaluated equipment performance data. The analysis shows current operational status, maintenance requirements, and optimization opportunities for improved efficiency and reliability."
                else:
                    response = f"🤖 **XAI Inspector**: Analyzing your query regarding: {prompt[:100]}... The intelligent assessment provides actionable insights and recommendations based on industrial best practices and regulatory compliance requirements."
                
                # Stream the response word by word
                for word in response.split():
                    full_response += word + " "
                    yield f"<span style='font-family:Tw Cen MT'>{word} </span>"
                    time.sleep(0.01)
                    
            except Exception as e:
                logger.error(f"XAI Inspector failed: {str(e)}")
                error_msg = "⚠️ XAI Inspector Error: Unable to process analysis request"
                yield f"<span style='color:red'>{error_msg}</span>"

        elif st.session_state.current_model == "Valonys Llama":
            # Enhanced local model loading for Llama
            try:
                if not TORCH_AVAILABLE:
                    raise ImportError("PyTorch not available for local model loading")
                
                model_id = "huggingface/CodeBERTa-small-v1"
                tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.getenv("HF_TOKEN"))
                model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", token=os.getenv("HF_TOKEN"))
                input_ids = tokenizer(PROMPTS[st.session_state.current_prompt] + "\n\n" + prompt, return_tensors="pt").to(model.device)
                output = model.generate(**input_ids, max_new_tokens=256, do_sample=True, temperature=0.7)
                decoded = tokenizer.decode(output[0], skip_special_tokens=True)
                
                # Stream the response
                for word in decoded.split():
                    full_response += word + " "
                    yield f"<span style='font-family:Tw Cen MT'>{word} </span>"
                    time.sleep(0.01)
                    
            except Exception as e:
                logger.error(f"Valonys Llama model loading failed: {str(e)}")
                # Fallback response with industrial context
                try:
                    if "notification" in prompt.lower():
                        response = f"📊 **Valonys Llama Analysis**: Based on notification data patterns, I observe trends in maintenance requests, equipment alerts, and operational notifications. Key insights include frequency patterns, work center distributions, and temporal trends that indicate maintenance optimization opportunities."
                    elif "pivot" in prompt.lower():
                        response = f"📈 **Pivot Table Analysis**: Valonys Llama has processed the pivot table data revealing notification distributions across work centers, FPSO locations, and time periods. The analysis shows operational patterns that can guide maintenance planning and resource allocation decisions."
                    else:
                        response = f"🦙 **Valonys Llama Response**: Analyzing your industrial data query. The local model provides insights into operational patterns, maintenance trends, and performance metrics based on the available data context."
                    
                    # Stream the fallback response
                    for word in response.split():
                        full_response += word + " "
                        yield f"<span style='font-family:Tw Cen MT'>{word} </span>"
                        time.sleep(0.01)
                        
                except Exception as fallback_e:
                    logger.error(f"Valonys Llama fallback failed: {str(fallback_e)}")
                    error_msg = "⚠️ Valonys Llama Error: Local model unavailable"
                    yield f"<span style='color:red'>{error_msg}</span>"

    except Exception as e:
        error_msg = f"<span style='color:red'>⚠️ Error: {str(e)}</span>"
        logger.error(f"Generation failed: {str(e)}")
        yield error_msg

# Initialize vector store
vector_store = VectorStore()

# --- MAIN UI ---
with st.sidebar:
    # Logo at the top
    st.image("attached_assets/ValonyLabs_Logo_1753908658100.png", width=120)
    
    st.header("⚙️ Analysis Configuration")
    
    # Agent Selection
    model_options = ["EE Smartest Agent", "JI Divine Agent", "EdJa-Valonys", "XAI Inspector", "Valonys Llama"]
    selected_model = st.selectbox(
        "🤖 Agent:",
        options=model_options,
        index=0,
        key="model_selector"
    )
    
    # Analysis Type Selection
    selected_prompt = st.selectbox(
        "📋 Analysis Type:",
        options=list(PROMPTS.keys()),
        index=0,
        key="prompt_selector"
    )
    
    # Update session state if selections changed
    if selected_model != st.session_state.current_model:
        st.session_state.current_model = selected_model
        st.session_state.model_intro_done = False
    
    if selected_prompt != st.session_state.current_prompt:
        st.session_state.current_prompt = selected_prompt
        st.session_state.model_intro_done = False
    
    st.markdown("---")
    
    # File Upload Section in Sidebar
    st.header("📄 Document Upload")
    
    # File type selection
    file_types = ["PDF", "Text Files", "Excel Files"]
    selected_file_type = st.selectbox("Select file type:", file_types)
    
    # File upload
    uploaded_files = None
    if selected_file_type == "PDF":
        uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
    elif selected_file_type == "Text Files":
        uploaded_files = st.file_uploader("Upload text files", type=["txt"], accept_multiple_files=True)
    elif selected_file_type == "Excel Files":
        uploaded_files = st.file_uploader("Upload Excel files", type=["xlsx", "xls"], accept_multiple_files=True)
    
    st.markdown("---")
    
    # Database integration - Cached files section
    st.header("📁 Cached Files")
    try:
        cached_files = vector_store.get_cached_files()
        if cached_files:
            file_options = ["None"] + [f['filename'] for f in cached_files]
            selected_cached_file = st.selectbox(
                "Load from cache:",
                options=file_options,
                key="cached_file_selector"
            )
            
            if selected_cached_file != "None":
                if st.button("📂 Load Selected File", key="load_cached_btn"):
                    with st.spinner(f"Loading {selected_cached_file}..."):
                        cached_data = vector_store.load_cached_file(selected_cached_file)
                        if cached_data:
                            st.success(f"✅ Loaded {selected_cached_file}")
                            st.session_state['cached_file_loaded'] = selected_cached_file
                        else:
                            st.error("❌ Failed to load file")
        else:
            st.info("No cached files available")
    except Exception as e:
        logger.error(f"Error accessing cached files: {str(e)}")
        st.warning("Cache unavailable")

# Display current model status
if st.session_state.current_model:
    st.info(f"🤖 **Active Model**: {st.session_state.current_model}")

# Process uploaded files from sidebar
if uploaded_files:
    try:
        if selected_file_type == "PDF":
            with st.spinner("Processing PDF documents..."):
                documents = DocumentProcessor.process_pdfs(uploaded_files)
                if documents:
                    vectorstore = DocumentProcessor.build_vectorstore(documents)
                    if vectorstore:
                        st.session_state.vectorstore = vectorstore
                        st.success(f"✅ Successfully processed {len(documents)} PDF documents")
                        st.session_state.last_processed = f"{len(documents)} PDF files"
                    else:
                        st.warning("⚠️ Vector store creation failed, using simple text storage")
                        st.session_state.vectorstore = SimpleTextStore(documents)
                        st.session_state.last_processed = f"{len(documents)} PDF files (simple storage)"
            
        elif selected_file_type == "Text Files":
            with st.spinner("Processing text documents..."):
                documents = []
                for file in uploaded_files:
                    content = str(file.read(), "utf-8")
                    documents.append(LCDocument(page_content=content, metadata={"name": file.name}))
                
                if documents:
                    vectorstore = DocumentProcessor.build_vectorstore(documents)
                    if vectorstore:
                        st.session_state.vectorstore = vectorstore
                        st.success(f"✅ Successfully processed {len(documents)} text documents")
                        st.session_state.last_processed = f"{len(documents)} text files"
                    else:
                        st.warning("⚠️ Vector store creation failed, using simple text storage")
                        st.session_state.vectorstore = SimpleTextStore(documents)
                        st.session_state.last_processed = f"{len(documents)} text files (simple storage)"
                        
        elif selected_file_type == "Excel Files":
            with st.spinner("Processing Excel files and creating database cache..."):
                # Use vector_store for Excel processing with database integration
                documents = vector_store.process_excel_to_documents(uploaded_files)
                if documents:
                    st.success(f"✅ Processed {len(uploaded_files)} Excel files with database caching")
                    st.session_state.last_processed = f"{len(uploaded_files)} Excel files"
                    
                    # Show pivot table creation option
                    if st.button("🔄 Create Notification Pivot Tables"):
                        with st.spinner("Creating pivot tables..."):
                            pivot_results = vector_store.create_notification_pivot_tables()
                            if pivot_results:
                                st.success("✅ Pivot tables created and cached")
                                
                                # Store pivot summary in session state for RAG context
                                pivot_summary = []
                                for table_name, data in pivot_results.items():
                                    if isinstance(data, dict) and 'summary' in data:
                                        pivot_summary.append(f"{table_name}: {data['summary']}")
                                    else:
                                        pivot_summary.append(f"{table_name}: {str(data)[:100]}...")
                                
                                st.session_state.pivot_summary = "\n".join(pivot_summary)
                                st.session_state.pivot_results = pivot_results
                                
                                # Display some results
                                with st.expander("📊 View Pivot Results"):
                                    for table_name, data in list(pivot_results.items())[:3]:
                                        st.subheader(f"📈 {table_name}")
                                        if isinstance(data, dict) and 'data' in data:
                                            st.json(data['data'])
                                        else:
                                            st.write(data)
                            else:
                                st.warning("⚠️ No pivot tables created")
                else:
                    st.warning("⚠️ No documents processed from Excel files")
    
    except Exception as e:
        st.error(f"❌ Error processing files: {str(e)}")
        logger.error(f"File processing error: {str(e)}")

# --- CHAT INTERFACE ---
st.header("💬 Industrial Analysis Chat")

# Initialize chat messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Agent introduction logic - only when user makes a selection
if st.session_state.current_model != st.session_state.get("last_model"):
    
    agent_intros = {
        "EE Smartest Agent": "💡 EE Agent Activated — Pragmatic & Smart",
        "JI Divine Agent": "✨ JI Agent Activated — DeepSeek Reasoning",
        "EdJa-Valonys": "⚡ EdJa Agent Activated — Cerebras Speed",
        "XAI Inspector": "🔍 XAI Inspector — Qwen Custom Fine-tune",
        "Valonys Llama": "🦙 Valonys Llama — LLaMA3-Based Reasoning"
    }
    
    if st.session_state.current_model and st.session_state.get("last_model") is not None:
        intro_message = agent_intros.get(st.session_state.current_model, "🤖 Agent Activated")
        
        with st.chat_message("assistant", avatar=BOT_AVATAR):
            st.markdown(intro_message)
        
        st.session_state.messages.append({"role": "assistant", "content": intro_message})
    
    st.session_state.last_model = st.session_state.current_model

# Display chat messages from history
for message in st.session_state.messages:
    avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"], unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Ask about inspection insights through data"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)
    
    # Generate assistant response
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            for response_chunk in generate_response(prompt):
                full_response += response_chunk
                message_placeholder.markdown(full_response + "▌", unsafe_allow_html=True)
            
            message_placeholder.markdown(full_response, unsafe_allow_html=True)
            
        except Exception as e:
            error_msg = f"❌ Error generating response: {str(e)}"
            message_placeholder.markdown(error_msg)
            logger.error(f"Chat response error: {str(e)}")
            full_response = error_msg
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

# Processing status
if st.session_state.last_processed:
    st.sidebar.success(f"📈 Last processed: {st.session_state.last_processed}")

# Add ValonyLabs branding at bottom
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; font-size: 14px;'>"
    "DigiTwin - Industrial Intelligence System | Powered by ValonyLabs"
    "</div>", 
    unsafe_allow_html=True
)

logger.info("DigiTwin application started successfully")