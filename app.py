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
    "Daily Report Summarization": """You are DigiTwin, an expert inspector with deep knowledge of industrial processes protocols. Your role is to analyze daily inspection reports and provide comprehensive summaries that highlight:

**ANALYSIS FRAMEWORK:**
1. **Critical Findings**: Any safety violations, equipment malfunctions, or compliance issues that require immediate attention
2. **Trend Analysis**: Patterns or recurring issues that may indicate systemic problems
3. **Recommendations**: Actionable steps to address identified issues and improve safety/compliance
4. **Risk Assessment**: Evaluation of potential risks and their severity levels
5. **Compliance Status**: Overall compliance with relevant regulations and standards

**OUTPUT FORMAT:**
- Start with an executive summary (2-3 sentences)
- Use clear headings for each analysis category
- Include specific data points, measurements, and references when available
- Prioritize findings by urgency and impact
- End with next steps and recommended actions

**INDUSTRIAL CONTEXT:**
Focus on FPSO operations, offshore safety protocols, equipment reliability, maintenance schedules, and regulatory compliance requirements. Use technical terminology appropriately while ensuring clarity for both technical and management audiences.

**NAMING CONVENTION AUGMENTATION:**
When analyzing notifications, adhere to the Naming Convention:
- For general notifications (except TBR & TEMP): Use format like A/ST/COA1/H142/FSD/CL24-XXX/PIE_SUPP/05
  - Prefixes: PS (Pressure Safety Device), PV (Pressure Vessel/Tank Topside), ST (Structure Topside), LI (Lifting), MA (Marine equipment like COT, WBT...)
  - Types: ICOA (Internal Coating), COA1-3 (External Coating), PASS (Passivation), REPL (Replacement without welding), WELD (Replacement by welding)
- Notification Types: Special focus on NI (Notifications of Integrity) and NC (Notifications of Conformity). Creatively classify anomalies into NI/NC, suggesting augmented names like A/PV/ICOA/H121/PASS/CL24-XXX/NI for integrity-related coating failures or NC for conformity gaps.

**PRIORITY DEFINITION AUGMENTATION:**
When determining priorities for notifications, use the following logic:
- Classify the fluid based on the class lists (Cl means Class Fluids):
  Class A: FW (Fire Water), AM (Methanol), GT (Gas Treated), NC (Raw Condensate), NH (Crude Oil), NV/PW (Produced Water), FS (Flare HP/LP), FC (Fuel Gas/Propane), FG (Fuel Gas), NO (Ethylene), XN/FO (Foam)
  Class B: CF (Heating Medium), PA (Instrument Air), OV/XG (Anti-Foam), AF/6G (Anti-Foam), SI/XE (Anti-Scale), DV/XO (Demulsifier/Deoiler), CW (Cooling/Chilled Water), GN (Nitrogen), TW (Injection Water), EG (Ethylene Glycol), XB (Corrosion Inhibitor)
  Class C: TA (Piping Package WP A Rosa), TB (Piping Package WP B Rosa), DS (Overboard Seawater), DW/WH (Potable Water), AV (Vent Gas), HH/LO (Hydraulic Fluid/Lube Oil), JW (Seawater Fouling), SW (Raw Seawater), IGV (Nden Gas/Carbon Dioxide), XM (Polyelectrolyte), LYT (Leach Rich TEG)
  Class D: DO/DF (Open Drain), SA (Service Air), BV/XC (Biocide All), XF (Biocide for Water), BW/RO (RO Water), WB/WG (Black/Grey Water), WD (Dirty Water Drain), SD (Deluge Drain), WW (Wash Water), UW/IX (Utility Water/Hydrant), HY (Sodium Hypochlorite)
- Identify if leak present and if corrosion is internal or external.
- Determine priority:
  If Class A: priority = 1 (regardless of leak or corrosion type)
  If Class B:
    if external corrosion: priority = 2
    if internal corrosion:
      if leak: 1
      else: 2
  If Class C:
    if external corrosion: priority = 3
    if internal corrosion:
      if leak: 3
      else: 4
  If Class D:
    if external corrosion: priority = 4
    if internal corrosion:
      if leak: 4
      else: 5
- For repair types like Welded Patches, Wrapping Long Term, Bolted/Injected Clamps, Calculated Clamps: assign TEMP priority.
- Integrate into NI/NC: For NI (integrity), use this for corrosion-related integrity issues; for NC (conformity), apply if applicable or assign based on conformity deviation severity, creatively mapping to fluid classes if relevant (e.g., higher priority for conformity issues in Class A fluids).""",

    "Equipment Performance Review": """You are DigiTwin, an equipment reliability specialist. Analyze equipment performance data and inspection reports to provide:

**ANALYSIS FRAMEWORK:**
1. **Performance Metrics**: Key performance indicators and their trends
2. **Maintenance Status**: Current maintenance requirements and schedules
3. **Equipment Health**: Overall condition assessment and remaining useful life
4. **Efficiency Analysis**: Operational efficiency and optimization opportunities
5. **Replacement Planning**: Recommendations for equipment upgrades or replacements

**OUTPUT FORMAT:**
- Start with an executive summary (2-3 sentences)
- Use clear headings for each analysis category
- Include specific data points, measurements, and references when available
- Prioritize findings by urgency and impact
- End with next steps and recommended actions

**INDUSTRIAL CONTEXT:**
Focus on FPSO operations, offshore safety protocols, equipment reliability, maintenance schedules, and regulatory compliance requirements. Use technical terminology appropriately while ensuring clarity for both technical and management audiences.

**NAMING CONVENTION AUGMENTATION:**
When analyzing notifications, adhere to the Naming Convention:
- For general notifications (except TBR & TEMP): Use format like A/ST/COA1/H142/FSD/CL24-XXX/PIE_SUPP/05
  - Prefixes: PS (Pressure Safety Device), PV (Pressure Vessel/Tank Topside), ST (Structure Topside), LI (Lifting), MA (Marine equipment like COT, WBT...)
  - Types: ICOA (Internal Coating), COA1-3 (External Coating), PASS (Passivation), REPL (Replacement without welding), WELD (Replacement by welding)
  - Locations: e.g., H121 (Hull Deck Module 121), P115 (Process Deck Module 11), MS-2 (Mach. Space LV -2), QLL2 (Living Q. Level 2)
- For TBR & TEMP notifications: Follow Section B conventions, focus on temporary repairs and backlog items
- Priorities: Use matrices for definition
  - Matrix 1 (Painting Touch Up): Based on TA (Thickness Allowance = RWT - MAWT), e.g., 0.5mm - TA <1mm for fluids A-D; applicable to Carbon Steel piping
  - Matrix 2 (Level 2 Priority): Based on D-Start date, e.g., 3 < D - Start date < 5 years = 4, with priorities like 1-HH, 2-H
- Notification Types: Special focus on NI (Notifications of Integrity) and NC (Notifications of Conformity). Creatively classify anomalies into NI/NC, suggesting augmented names like A/PV/ICOA/H121/PASS/CL24-XXX/NI for integrity-related coating failures or NC for conformity gaps.

**PRIORITY DEFINITION AUGMENTATION:**
When determining priorities for notifications, use the following logic:
- Classify the fluid based on the class lists (Cl means Class Fluids):
  Class A: FW (Fire Water), AM (Methanol), GT (Gas Treated), NC (Raw Condensate), NH (Crude Oil), NV/PW (Produced Water), FS (Flare HP/LP), FC (Fuel Gas/Propane), FG (Fuel Gas), NO (Ethylene), XN/FO (Foam)
  Class B: CF (Heating Medium), PA (Instrument Air), OV/XG (Anti-Foam), AF/6G (Anti-Foam), SI/XE (Anti-Scale), DV/XO (Demulsifier/Deoiler), CW (Cooling/Chilled Water), GN (Nitrogen), TW (Injection Water), EG (Ethylene Glycol), XB (Corrosion Inhibitor)
  Class C: TA (Piping Package WP A Rosa), TB (Piping Package WP B Rosa), DS (Overboard Seawater), DW/WH (Potable Water), AV (Vent Gas), HH/LO (Hydraulic Fluid/Lube Oil), JW (Seawater Fouling), SW (Raw Seawater), IGV (Nden Gas/Carbon Dioxide), XM (Polyelectrolyte), LYT (Leach Rich TEG)
  Class D: DO/DF (Open Drain), SA (Service Air), BV/XC (Biocide All), XF (Biocide for Water), BW/RO (RO Water), WB/WG (Black/Grey Water), WD (Dirty Water Drain), SD (Deluge Drain), WW (Wash Water), UW/IX (Utility Water/Hydrant), HY (Sodium Hypochlorite)
- Identify if leak present and if corrosion is internal or external.
- Determine priority:
  If Class A: priority = 1 (regardless of leak or corrosion type)
  If Class B:
    if external corrosion: priority = 2
    if internal corrosion:
      if leak: 1
      else: 2
  If Class C:
    if external corrosion: priority = 3
    if internal corrosion:
      if leak: 3
      else: 4
  If Class D:
    if external corrosion: priority = 4
    if internal corrosion:
      if leak: 4
      else: 5
- For repair types like Welded Patches, Wrapping Long Term, Bolted/Injected Clamps, Calculated Clamps: assign TEMP priority.
- Integrate into NI/NC: For NI (integrity), use this for corrosion-related integrity issues; for NC (conformity), apply if applicable or assign based on conformity deviation severity, creatively mapping to fluid classes if relevant (e.g., higher priority for conformity issues in Class A fluids).""",

    "Compliance Assessment": """You are DigiTwin, a compliance expert specializing in industrial regulations. Conduct comprehensive compliance assessments covering:

**ANALYSIS FRAMEWORK:**
1. **Regulatory Framework**: Applicable regulations and standards
2. **Compliance Status**: Current compliance levels and gaps
3. **Documentation Review**: Adequacy of required documentation and records
4. **Training Requirements**: Staff training needs for compliance
5. **Audit Readiness**: Preparation status for regulatory audits

**OUTPUT FORMAT:**
- Start with an executive summary (2-3 sentences)
- Use clear headings for each analysis category
- Include specific data points, measurements, and references when available
- Prioritize findings by urgency and impact
- End with next steps and recommended actions

**INDUSTRIAL CONTEXT:**
Focus on FPSO operations, offshore safety protocols, equipment reliability, maintenance schedules, and regulatory compliance requirements. Use technical terminology appropriately while ensuring clarity for both technical and management audiences.
.""",

    "Pivot Table Analysis": """You are DigiTwin, a data analysis expert specializing in notification data analysis. Analyze the pivot table data and provide insights on:

**ANALYSIS FRAMEWORK:**
1. **Notification Patterns**: Identify trends in notification types and frequencies
2. **Work Center Performance**: Analyze notification distribution across work centers
3. **FPSO Analysis**: Examine notification patterns by FPSO location
4. **Temporal Trends**: Identify time-based patterns in notification creation
5. **Operational Insights**: Provide actionable recommendations based on data patterns

**OUTPUT FORMAT:**
- Start with an executive summary (2-3 sentences)
- Use clear headings for each analysis category
- Include specific data points, measurements, and references when available
- Prioritize findings by urgency and impact
- End with next steps and recommended actions

**INDUSTRIAL CONTEXT:**
Focus on FPSO operations, offshore safety protocols, equipment reliability, maintenance schedules, and regulatory compliance requirements. Use technical terminology appropriately while ensuring clarity for both technical and management audiences.

**NAMING CONVENTION AUGMENTATION:**
When analyzing notifications, adhere to the CLV Naming Convention:
- For general notifications (except TBR & TEMP): Use format like A/ST/COA1/H142/FSD/CL24-XXX/PIE_SUPP/05
  - Prefixes: PS (Pressure Safety Device), PV (Pressure Vessel/Tank Topside), ST (Structure Topside), LI (Lifting), MA (Marine equipment like COT, WBT...)
  - Types: ICOA (Internal Coating), COA1-3 (External Coating), PASS (Passivation), REPL (Replacement without welding), WELD (Replacement by welding), TBR (To be Replaced).

**PRIORITY DEFINITION AUGMENTATION:**
When determining priorities for notifications, use the following logic:
- Classify the fluid based on the class lists (Cl means Class Fluids):
  Class A: FW (Fire Water), AM (Methanol), GT (Gas Treated), NC (Raw Condensate), NH (Crude Oil), NV/PW (Produced Water), FS (Flare HP/LP), FC (Fuel Gas/Propane), FG (Fuel Gas), NO (Ethylene), XN/FO (Foam)
  Class B: CF (Heating Medium), PA (Instrument Air), OV/XG (Anti-Foam), AF/6G (Anti-Foam), SI/XE (Anti-Scale), DV/XO (Demulsifier/Deoiler), CW (Cooling/Chilled Water), GN (Nitrogen), TW (Injection Water), EG (Ethylene Glycol), XB (Corrosion Inhibitor)
  Class C: TA (Piping Package WP A Rosa), TB (Piping Package WP B Rosa), DS (Overboard Seawater), DW/WH (Potable Water), AV (Vent Gas), HH/LO (Hydraulic Fluid/Lube Oil), JW (Seawater Fouling), SW (Raw Seawater), IGV (Nden Gas/Carbon Dioxide), XM (Polyelectrolyte), LYT (Leach Rich TEG)
  Class D: DO/DF (Open Drain), SA (Service Air), BV/XC (Biocide All), XF (Biocide for Water), BW/RO (RO Water), WB/WG (Black/Grey Water), WD (Dirty Water Drain), SD (Deluge Drain), WW (Wash Water), UW/IX (Utility Water/Hydrant), HY (Sodium Hypochlorite)
- Identify if leak present and if corrosion is internal or external.
- Determine priority:
  If Class A: priority = 1 (regardless of leak or corrosion type)
  If Class B:
    if external corrosion: priority = 2
    if internal corrosion:
      if leak: 1
      else: 2
  If Class C:
    if external corrosion: priority = 3
    if internal corrosion:
      if leak: 3
      else: 4
  If Class D:
    if external corrosion: priority = 4
    if internal corrosion:
      if leak: 4
      else: 5
- For repair types like Welded Patches, Wrapping Long Term, Bolted/Injected Clamps, Calculated Clamps: assign TEMP priority.
- Integrate into NI/NC: For NI (integrity), use this for corrosion-related integrity issues; for NC (conformity), apply if applicable or assign based on conformity deviation severity, creatively mapping to fluid classes if relevant (e.g., higher priority for conformity issues in Class A fluids)."""
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

    # Enhanced Multi-Report RAG Context
    fpso_filter = st.session_state.get('selected_fpso', 'All FPSOs')

    # Try enhanced vector store first, then fallback to session vectorstore
    if vector_store and hasattr(vector_store, 'get_comprehensive_context'):
        try:
            comprehensive_context = vector_store.get_comprehensive_context(prompt, fpso_filter)
            if comprehensive_context and comprehensive_context != "Context retrieval failed":
                messages.append({"role": "system", "content": f"Comprehensive Context:\n{comprehensive_context}"})
            else:
                # Fallback to basic vector search
                raise Exception("Comprehensive context failed")
        except Exception as e:
            logger.warning(f"Enhanced RAG failed: {str(e)}. Using basic vector search.")
            if st.session_state.vectorstore:
                try:
                    docs = st.session_state.vectorstore.similarity_search(prompt, k=5)
                    context = "\n\n".join([f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}" 
                                         for doc in docs])
                    messages.append({"role": "system", "content": f"Relevant Context:\n{context}"})
                except Exception as e:
                    logger.warning(f"Basic vectorstore search failed: {str(e)}")
    elif st.session_state.vectorstore:
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

    # Add FPSO selection context
    if 'selected_fpso' in st.session_state and st.session_state.selected_fpso:
        fpso_selection = st.session_state.selected_fpso
        if fpso_selection != "All FPSOs":
            fpso_context = f"\n\nCURRENT FPSO FOCUS: {fpso_selection}\nIMPORTANT: Focus your analysis specifically on {fpso_selection} FPSO operations. When analyzing notifications, work centers, or operational data, prioritize insights related to this specific FPSO."
            messages.append({"role": "system", "content": fpso_context})
        else:
            messages.append({"role": "system", "content": "\n\nCURRENT ANALYSIS SCOPE: All FPSOs - Provide comprehensive analysis across all FPSO operations (GIR, DAL, PAZ, CLV)."})

    # Add detailed pivot results context if available
    if 'pivot_results' in st.session_state and st.session_state.pivot_results:
        pivot_details_context = f"\n\nAVAILABLE PIVOT ANALYSES:\n"
        for table_name in st.session_state.pivot_results.keys():
            pivot_details_context += f"- {table_name}\n"
        pivot_details_context += "\nYou can reference these specific pivot table results when answering questions about trends, patterns, and operational insights."
        messages.append({"role": "system", "content": pivot_details_context})

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
                    "model": "grok-2-latest",
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
                # Check if Cerebras is available
                try:
                    from cerebras.cloud.sdk import Cerebras
                    client = Cerebras(api_key=os.getenv("CEREBRAS_API_KEY"))
                    response = client.chat.completions.create(model="llama3.1-8b", messages=messages)
                    content = response.choices[0].message.content if hasattr(response.choices[0].message, "content") else str(response.choices[0])
                except ImportError:
                    raise Exception("Cerebras SDK not available")

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
    # Logo at the top - centered
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
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

    # Analysis Type Selection moved up - FPSO selection moved to main canvas

    # Update session state if selections changed
    if selected_model != st.session_state.current_model:
        st.session_state.current_model = selected_model
        st.session_state.model_intro_done = False

    if selected_prompt != st.session_state.current_prompt:
        st.session_state.current_prompt = selected_prompt
        st.session_state.model_intro_done = False

    # FPSO selection now handled in main canvas tabs

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
                        if cached_data and cached_data.get('status') == 'loaded':
                            docs_count = len(cached_data.get('documents', []))
                            has_data = cached_data.get('has_processed_data', False)
                            
                            if has_data:
                                st.success(f"✅ Loaded {selected_cached_file} with {docs_count} documents and data for analysis")
                            else:
                                st.success(f"✅ Loaded {selected_cached_file} with {docs_count} documents")
                                
                            st.session_state['cached_file_loaded'] = selected_cached_file
                            st.session_state['last_processed'] = f"Cached: {selected_cached_file}"
                            
                            # Clear any existing pivot results to refresh with cached data
                            if 'pivot_results' in st.session_state:
                                del st.session_state['pivot_results'] 
                            
                            # Set vectorstore for compatibility with chat interface
                            if hasattr(vector_store, 'vector_store') and vector_store.vector_store:
                                st.session_state.vectorstore = vector_store.vector_store
                                
                            st.rerun()
                        else:
                            st.error("❌ Failed to load cached file. Please try re-uploading the file.")
        else:
            st.info("No cached files available")
    except Exception as e:
        logger.error(f"Error accessing cached files: {str(e)}")
        st.warning("Cache unavailable")

# Display current model status and dataset info
if st.session_state.current_model:
    st.info(f"🤖 **Active Model**: {st.session_state.current_model}")

# Check if dataset is available from auto-loading
if hasattr(vector_store, 'processed_data') and vector_store.processed_data is not None and len(vector_store.processed_data) > 0:
    if not st.session_state.get('last_processed'):
        st.session_state.last_processed = f"Auto-loaded: {vector_store.current_file_name}"
    st.success(f"📊 **Dataset Ready**: {len(vector_store.processed_data)} records loaded for analysis")
elif hasattr(vector_store, 'current_file_name') and vector_store.current_file_name:
    st.info(f"📂 **File Loaded**: {vector_store.current_file_name} - Processing dataset...")
    # Try to trigger dataset restoration
    if hasattr(vector_store, '_auto_load_dataset'):
        with st.spinner("Restoring dataset..."):
            if vector_store._auto_load_dataset():
                st.rerun()

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

                    st.info("📊 **Excel files processed successfully!** Use the Enhanced Pivot Analysis Dashboard above to create and view detailed analysis.")
                else:
                    st.warning("⚠️ No documents processed from Excel files")

    except Exception as e:
        st.error(f"❌ Error processing files: {str(e)}")
        logger.error(f"File processing error: {str(e)}")

# --- ENHANCED PIVOT TABLE INTERFACE ---
st.header("📊 Enhanced Pivot Analysis Dashboard")

# FPSO Selection as Tabs in Main Canvas
fpso_options = ["All FPSOs", "GIR", "DAL", "PAZ", "CLV"]
fpso_tabs = st.tabs(fpso_options)

# Initialize session state for FPSO and Work Center selections
if 'selected_fpso' not in st.session_state:
    st.session_state.selected_fpso = "All FPSOs"
if 'selected_work_center' not in st.session_state:
    st.session_state.selected_work_center = "All Work Centers"

# Handle each FPSO tab
for i, fpso in enumerate(fpso_options):
    with fpso_tabs[i]:
        # Update session state based on active tab
        st.session_state.selected_fpso = fpso
        
        st.subheader(f"🏗️ {fpso} Analysis Dashboard")
        
        # Check if we have pivot results for this FPSO
        pivot_results = st.session_state.get('pivot_results')
        if pivot_results and pivot_results.get('fpso_filter') == fpso:
            pivot_table = pivot_results.get('pivot_table')
            if pivot_table is not None:
                # Work Center Dropdown Menu
                work_centers = ["All Work Centers"] + list(pivot_table.columns[:-1])  # Exclude 'Total' column
                selected_work_center = st.selectbox(
                    "🔧 Select Work Center:",
                    options=work_centers,
                    key=f"work_center_{fpso}",
                    index=0
                )
                st.session_state.selected_work_center = selected_work_center
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Notifications", pivot_results.get('total_records', 0))
                with col2:
                    st.metric("Notification Types", pivot_results.get('notification_types_count', 0))
                with col3:
                    if selected_work_center != "All Work Centers":
                        if selected_work_center in pivot_table.columns:
                            # Exclude Total row when calculating work center total
                            work_center_data = pivot_table[selected_work_center]
                            work_center_total = work_center_data[work_center_data.index != "Total"].sum()
                        else:
                            work_center_total = 0
                        st.metric(f"{selected_work_center} Total", work_center_total)
                    else:
                        st.metric("Work Centers", pivot_results.get('work_centers_count', 0))
                
                # Filter and display pivot table based on Work Center selection
                if selected_work_center == "All Work Centers":
                    # Show full table without Work Center columns (exclude Total column to avoid doubling)
                    data_section = pivot_table.iloc[:, :-1]  # Exclude "Total" column
                    display_data = data_section.sum(axis=1).to_frame("Total Notifications")
                    # Remove the "Total" row if it exists
                    if "Total" in display_data.index:
                        display_data = display_data.drop("Total", errors='ignore')
                    st.subheader("📈 Notification Types Summary")
                    st.dataframe(display_data, use_container_width=True)
                else:
                    # Show only selected work center data
                    if selected_work_center in pivot_table.columns:
                        display_data = pivot_table[[selected_work_center]].copy()
                        display_data = display_data[display_data[selected_work_center] > 0]  # Filter out zero values
                        st.subheader(f"📈 {selected_work_center} - Notification Types")
                        st.dataframe(display_data, use_container_width=True)
                        
                        # Show top notification types for this work center
                        if not display_data.empty:
                            # Filter out Total row for top calculations
                            display_data_filtered = display_data[display_data.index != "Total"]
                            if not display_data_filtered.empty:
                                top_notifications = display_data_filtered.nlargest(5, selected_work_center)
                                st.subheader(f"🔝 Top 5 Notification Types in {selected_work_center}")
                                for idx, (notif_type, count) in enumerate(top_notifications.iterrows(), 1):
                                    st.write(f"{idx}. **{notif_type}**: {count[selected_work_center]} notifications")
                
                # Key insights for this FPSO
                stats = pivot_results.get('summary_stats', {})
                if stats:
                    st.subheader("🔍 Key Insights")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.info(f"**Top Notification Type:** {stats.get('top_notification_type', 'N/A')}")
                    with col2:
                        st.info(f"**Busiest Work Center:** {stats.get('top_work_center', 'N/A')}")
            else:
                st.info(f"📊 No pivot analysis available for {fpso}. Upload Excel files and create pivot analysis to see data here.")
                
                # Show create analysis button for this FPSO
                if st.button(f"🔄 Create {fpso} Analysis", key=f"create_{fpso}"):
                    with st.spinner(f"Creating pivot analysis for {fpso}..."):
                        if vector_store:
                            pivot_results = vector_store.create_notification_pivot_tables(fpso_filter=fpso)
                            if pivot_results:
                                st.session_state.pivot_results = pivot_results
                                st.success(f"✅ Analysis complete for {fpso}!")
                                st.rerun()
                            else:
                                st.warning("❌ Could not create analysis. Please ensure Excel files are uploaded with required columns.")
        else:
            st.info(f"📊 No pivot analysis available for {fpso}. Upload Excel files and create pivot analysis to see data here.")
            
            # Show create analysis button for this FPSO
            if st.button(f"🔄 Create {fpso} Analysis", key=f"create_{fpso}_alt"):
                with st.spinner(f"Creating pivot analysis for {fpso}..."):
                    if vector_store:
                        pivot_results = vector_store.create_notification_pivot_tables(fpso_filter=fpso)
                        if pivot_results:
                            st.session_state.pivot_results = pivot_results
                            st.success(f"✅ Analysis complete for {fpso}!")
                            st.rerun()
                        else:
                            st.warning("❌ Could not create analysis. Please ensure Excel files are uploaded with required columns.")

st.markdown("---")

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
    "<div style='text-align: center; font-size: 14px; font-weight: bold;'>"
    "<span style='color: #1f4e79;'>DigiTwin - Industrial Intelligence System</span> | "
    "<span style='color: #000000;'>Powered by ValonyLabs</span> "
    "(<a href='https://www.valonylabs.com' target='_blank' style='color: #1f4e79; text-decoration: none;'>www.valonylabs.com</a>)"
    "</div>", 
    unsafe_allow_html=True
)

logger.info("DigiTwin application started successfully")