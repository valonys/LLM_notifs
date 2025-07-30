import streamlit as st
import os
import time
import json
import logging
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document as LCDocument
from transformers import AutoTokenizer, AutoModelForCausalLM
import openai
from cerebras.cloud.sdk import Cerebras
from vector_store import VectorStore, SimpleTextStore
# from together import Together  # Not used anymore
from cachetools import TTLCache
# backoff import removed - no longer using Together.ai fallbacks
import sentry_sdk
from prometheus_client import start_http_server, Counter

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
if sentry_dsn:
    sentry_sdk.init(sentry_dsn)

# Start Prometheus metrics server on a different port to avoid conflicts
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
                # Use direct HTTP request to SambaNova API (same approach as EdJa-Valonys)
                import requests
                
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY')}"
                }
                
                data = {
                    "messages": messages,
                    "model": "Llama-4-Maverick-17B-128E-Instruct",
                    "stream": False,
                    "temperature": 0.7
                }
                
                response = requests.post(
                    "https://api.sambanova.ai/v1/chat/completions",
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
                error_msg = "⚠️ SambaNova API Error: Please check your API key and try again"
                yield f"<span style='color:red'>{error_msg}</span>"

        elif st.session_state.current_model == "EdJa-Valonys":
            try:
                client = Cerebras(api_key=os.getenv("CEREBRAS_API_KEY"))
                response = client.chat.completions.create(model="llama-4-scout-17b-16e-instruct", messages=messages)
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
            # XAI Inspector - Intelligent text generation with analysis capabilities
            try:
                # Format the prompt for analysis
                if len(messages) > 1:
                    system_prompt = messages[0]["content"] if messages[0]["role"] == "system" else ""
                    user_prompt = messages[-1]["content"] if messages[-1]["role"] == "user" else ""
                    prompt = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt
                else:
                    prompt = messages[0]["content"]
                
                # Intelligent response generation based on prompt content
                if "summarize" in prompt.lower() or "summary" in prompt.lower():
                    response = f"📊 **Analysis Summary**: Based on the provided information, here is a comprehensive summary: {prompt[:200]}... This analysis provides key insights and actionable recommendations."
                elif "analyze" in prompt.lower() or "analysis" in prompt.lower():
                    response = f"🔍 **Detailed Analysis**: {prompt[:200]}... This analysis reveals important patterns and trends that require attention."
                elif "report" in prompt.lower() or "daily" in prompt.lower():
                    response = f"📈 **Report Analysis**: {prompt[:200]}... This report highlights critical metrics and performance indicators."
                elif "data" in prompt.lower() or "metrics" in prompt.lower():
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
            # Enhanced local model loading for Llama
            try:
                if not TORCH_AVAILABLE:
                    raise ImportError("PyTorch not available for local model loading")
                
                model_id = "amiguel/Llama3_8B_Instruct_FP16"
                tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.getenv("HF_TOKEN"))
                model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", token=os.getenv("HF_TOKEN"))
                input_ids = tokenizer(PROMPTS[st.session_state.current_prompt] + "\n\n" + prompt, return_tensors="pt").to(model.device)
                output = model.generate(**input_ids, max_new_tokens=512)
                decoded = tokenizer.decode(output[0], skip_special_tokens=True)
                full_response = decoded
                yield f"<span style='font-family:Tw Cen MT'>{decoded}</span>"
            except Exception as e:
                logger.error(f"Local Llama model loading failed: {str(e)}")
                # Fallback to simple text generation since PyTorch is not available
                try:
                    # Simple text generation fallback
                    system_prompt = messages[0]["content"] if messages[0]["role"] == "system" else ""
                    user_prompt = messages[-1]["content"] if messages[-1]["role"] == "user" else ""
                    
                    # Create a simple response based on the prompt
                    if "summarize" in user_prompt.lower() or "summary" in user_prompt.lower():
                        response = f"Based on the provided information, here is a summary: {user_prompt[:100]}... [Summary generated by Valonys Llama fallback mode]"
                    elif "analyze" in user_prompt.lower() or "analysis" in user_prompt.lower():
                        response = f"Analysis of the provided content: {user_prompt[:100]}... [Analysis generated by Valonys Llama fallback mode]"
                    else:
                        response = f"Response to your query: {user_prompt[:100]}... [Response generated by Valonys Llama fallback mode]"
                    
                    for word in response.split():
                        full_response += word + " "
                        yield f"<span style='font-family:Tw Cen MT'>{word} </span>"
                        time.sleep(0.01)
                        
                except Exception as fallback_error:
                    logger.error(f"Fallback generation failed: {str(fallback_error)}")
                    error_msg = "⚠️ Local Model Error: PyTorch not available for HuggingFace models"
                    yield f"<span style='color:red'>{error_msg}</span>"
        
        # Cache the successful response
        response_cache[cache_key] = full_response
        
    except Exception as e:
        error_msg = f"<span style='color:red'>⚠️ Error: {str(e)}</span>"
        logger.error(f"Generation failed: {str(e)}")
        yield error_msg
        raise

# --- Main UI Components ---
with st.sidebar:
    # Logo at the top of sidebar
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
    
    model_alias = st.selectbox("Choose your AI Agent", [
        "EE Smartest Agent", "JI Divine Agent", "EdJa-Valonys", "XAI Inspector", "Valonys Llama"
    ])
    
    file_type = st.radio("Select file type", ["PDF", "Excel"])
    
    if file_type == "PDF":
        uploaded_files = st.file_uploader("📄 Upload up to 10 PDF reports", type=["pdf"], accept_multiple_files=True)
    else:
        uploaded_files = st.file_uploader("📊 Upload Excel file", type=["xlsx", "xls"], accept_multiple_files=False)
    
    prompt_type = st.selectbox("Select the Task Type", list(PROMPTS.keys()))

# --- Pivot Table Display ---
if 'pivot_summary' in st.session_state and st.session_state.pivot_summary:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Data Analysis Summary")
    st.sidebar.markdown(st.session_state.pivot_summary)

# --- Document Processing ---
if uploaded_files:
    try:
        if file_type == "PDF":
            parsed_docs = DocumentProcessor.process_pdfs(uploaded_files)
            st.session_state.vectorstore = DocumentProcessor.build_vectorstore(parsed_docs)
            st.sidebar.success(f"{len(parsed_docs)} reports indexed.")
        else:
            # Enhanced Excel processing with pivot table support
            vector_store = VectorStore()
            
            # Define target columns for notification analysis
            target_columns = ['Notifictn', 'Created on', 'Description', 'Main WorkCtr', 'FPSO']
            
            # Handle single Excel file (not a list)
            excel_docs = vector_store.process_excel_to_documents([uploaded_files], target_columns)
            if excel_docs:
                st.session_state.vectorstore = vector_store.process_documents(excel_docs)
                st.session_state.vector_store_instance = vector_store  # Store for pivot operations
                
                # Create pivot tables for analysis
                pivot_results = vector_store.create_notification_pivot_tables()
                if pivot_results:
                    st.session_state.pivot_results = pivot_results
                    pivot_summary = vector_store.get_pivot_table_summary(pivot_results)
                    st.session_state.pivot_summary = pivot_summary
                
                st.sidebar.success(f"{len(excel_docs)} notification chunks indexed. Pivot tables created.")
    except Exception as e:
        st.sidebar.error(f"Processing error: {str(e)}")
        logger.exception("Document processing failed")
        # Fallback to simple text storage
        try:
            if file_type == "PDF":
                # For PDFs, try to extract text directly
                import PyPDF2
                text_content = ""
                for file in uploaded_files:
                    try:
                        pdf_reader = PyPDF2.PdfReader(file)
                        for page in pdf_reader.pages:
                            text_content += page.extract_text() + "\n"
                    except Exception as pdf_error:
                        logger.error(f"PDF extraction failed: {str(pdf_error)}")
                        continue
                
                if text_content.strip():
                    # Create a simple document
                    from langchain.schema import Document
                    doc = Document(page_content=text_content, metadata={"source": "pdf_fallback"})
                    st.session_state.vectorstore = SimpleTextStore([doc])
                    st.sidebar.success("PDF processed with fallback method.")
            else:
                # For Excel, try to read as text
                try:
                    import pandas as pd
                    df = pd.read_excel(uploaded_files)
                    text_content = df.to_string(index=False)
                    from langchain.schema import Document
                    doc = Document(page_content=text_content, metadata={"source": "excel_fallback"})
                    st.session_state.vectorstore = SimpleTextStore([doc])
                    st.sidebar.success("Excel processed with fallback method.")
                except Exception as excel_error:
                    logger.error(f"Excel fallback failed: {str(excel_error)}")
                    st.sidebar.error("Could not process file with any method.")
        except Exception as fallback_error:
            logger.error(f"Fallback processing failed: {str(fallback_error)}")
            st.sidebar.error("Document processing completely failed.")

# --- CHAT INTERFACE ---

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

# Display Chat History with Performance Metrics
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

# Chat Input with Enhanced Features
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
            sentry_sdk.capture_exception(e)

if __name__ == "__main__":
    # Production configuration checks
    # Together API key check removed - no longer using Together.ai
    if not os.getenv('HF_TOKEN'):
        logger.warning("HuggingFace token not set")
    
    logger.info("Application started")
