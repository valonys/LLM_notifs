# DigiTwin RAG System

## Overview
DigiTwin is a production-ready Retrieval-Augmented Generation (RAG) system designed for industrial inspection and document analysis. Built using the user's proven app_rag.py architecture, the system provides AI-powered insights through the original Streamlit interface with preserved fonts, styling, and model configurations. Enhanced with PostgreSQL database persistence to eliminate redundant Excel file uploads and enable cached data retrieval. Specifically tailored for industrial environments like FPSO operations, offshore safety protocols, and regulatory compliance monitoring.

## Recent Changes (January 30, 2025)
- ✅ **Architecture Restored**: Reverted to user's original working app_rag.py design and structure
- ✅ **Database Integration**: Added PostgreSQL persistence for uploaded files and processed documents
- ✅ **File Caching System**: Implemented database-backed file caching to avoid re-uploading Excel files
- ✅ **Pivot Table Persistence**: Enhanced vector_store.py with database caching for pivot table operations
- ✅ **Model Connections**: Restored original API configurations (XAI, DeepSeek, Cerebras) with proper streaming
- ✅ **UI Preservation**: Maintained exact original styling, fonts ("Tw Cen MT"), and interface layout
- ✅ **FPSO-Focused Analysis**: Added FPSO radio buttons (GIR, DAL, PAZ, CLV) in sidebar for targeted analysis
- ✅ **Enhanced Pivot Analysis**: Focused on key columns (Main Work Ctr, Notifictn type, Description, Created on, FPSO, Completn date)
- ✅ **Chat Interface Enhancement**: Chat agents now receive FPSO context and pivot table analysis data
- ✅ **Database Fix**: Resolved foreign key constraint issues for document persistence
- ✅ **Professional Branding**: Added clickable ValonyLabs website link in footer

- ✅ **Pivot Table Vector Integration**: Enhanced chunking system converts pivot analysis into searchable documents with intelligent insights
- ✅ **Robust Multi-Report RAG**: Comprehensive multi-report RAG system with cross-referenced analysis and FPSO filtering
- ✅ **Enhanced Data Retrieval**: Advanced vector search combining structured pivot data with document analysis

## Latest Enhancements (January 30, 2025)
- ✅ **Enhanced Vector Store Architecture**: Integrated pivot table analysis directly into vector store for seamless RAG retrieval
- ✅ **Multi-Report RAG System**: Built comprehensive context system combining Excel data, pivot insights, and document analysis
- ✅ **Intelligent Pivot Chunking**: Advanced document generation from pivot tables with metadata and contextual insights
- ✅ **FPSO-Aware RAG**: Vector search automatically filters and enhances results based on selected FPSO (GIR, DAL, PAZ, CLV)
- ✅ **Comprehensive Context Engine**: Chat agents receive enriched context from multiple data sources and analytical insights

## Future Development Priorities
- 🔄 **Performance Optimization**: Implement advanced caching strategies for large-scale pivot analysis
- 🔄 **Multi-Document Intelligence**: Enhance cross-document pattern recognition and insight generation
- 🔄 **Advanced Analytics Dashboard**: Create interactive visualization for multi-FPSO trend analysis

## User Preferences
```
Preferred communication style: Simple, everyday language.
```

## System Architecture
The application follows a modular, component-based architecture with clear separation of concerns:

### Frontend Architecture
- **Framework**: Streamlit for web interface
- **Caching**: Streamlit's `@st.cache_resource` for component initialization
- **Streaming**: Real-time response streaming for better user experience
- **Layout**: Multi-tab interface for different analysis modes

### Backend Architecture
- **Core Components**: Modular design with dedicated managers for different functionalities
- **Document Processing**: Multi-format support (PDF, DOCX, Excel, CSV, TXT)
- **Vector Storage**: Enhanced vector store with multiple embedding strategies
- **Model Management**: Multi-provider AI model support with intelligent routing
- **Query Processing**: Smart routing between document analysis and pivot table generation
- **Database Integration**: PostgreSQL database for persistent data storage and analytics

## Key Components

### Core Modules
1. **Vector Store (`core/vector_store.py`)**: Enhanced vector storage with FAISS, ChromaDB support and fallback mechanisms
2. **Document Processor (`core/document_processor.py`)**: Multi-format document processing with metadata extraction
3. **Model Manager (`core/model_manager.py`)**: Multi-provider AI model management (OpenAI, Cerebras, xAI, local models)
4. **Query Router (`core/query_router.py`)**: Intelligent query classification and routing
5. **Pivot Analyzer (`core/pivot_analyzer.py`)**: Specialized data analysis for industrial metrics with JSON serialization fix
6. **Embedding Manager (`core/embedding_manager.py`)**: Multiple embedding model support with caching
7. **Database Manager (`core/database_manager.py`)**: PostgreSQL database integration with SQLAlchemy ORM for persistent storage

### Utility Components
1. **Cache Manager (`utils/cache_manager.py`)**: Multi-tier caching (memory + disk + database)
2. **Error Handler (`utils/error_handler.py`)**: Comprehensive error handling with Sentry integration
3. **Config Manager (`utils/config.py`)**: Configuration management with environment variable support

### Domain-Specific Features
1. **Industrial Prompts (`prompts/industrial_prompts.py`)**: Specialized prompts for safety analysis, compliance checking
2. **Domain Models (`models/domain_models.py`)**: Industrial-specific data structures and classifications

## Data Flow

### Document Processing Flow
1. **Upload**: Multi-format file upload through Streamlit interface
2. **Processing**: Format-specific processors extract text and metadata
3. **Chunking**: Advanced chunking strategies with semantic awareness
4. **Embedding**: Multiple embedding models with fallback mechanisms
5. **Storage**: Enhanced vector store with multiple backend options

### Query Processing Flow
1. **Input**: Natural language query from user
2. **Classification**: Query router determines processing strategy (document vs. pivot analysis)
3. **Retrieval**: Vector similarity search or data pivot operations
4. **Generation**: AI model generates contextual response
5. **Streaming**: Real-time response delivery to frontend

### Caching Strategy
- **L1 Cache**: In-memory cache for frequently accessed items
- **L2 Cache**: Disk-based cache for embeddings and processed documents
- **L3 Cache**: SQLite database for persistent cache with TTL support

## External Dependencies

### AI Model Providers
- **OpenAI**: GPT models for advanced reasoning
- **Cerebras**: High-performance inference
- **xAI (Grok)**: Latest AI capabilities
- **Hugging Face**: Local model support and embeddings

### Document Processing
- **PyPDF2/pypdf**: PDF processing
- **python-docx**: Word document handling
- **openpyxl**: Excel file processing
- **pandas**: Data manipulation and analysis

### Vector Storage
- **FAISS**: High-performance vector similarity search
- **ChromaDB**: Alternative vector database
- **Sentence Transformers**: Embedding generation

### Monitoring & Observability
- **Sentry**: Error tracking and monitoring
- **Prometheus**: Metrics collection
- **Custom logging**: Comprehensive application logging

## Deployment Strategy

### Environment Configuration
- **Environment Variables**: API keys and configuration through `.env` files
- **Graceful Degradation**: Fallback mechanisms when dependencies are unavailable
- **Error Resilience**: Comprehensive error handling with multiple recovery strategies

### Scalability Considerations
- **Caching**: Multi-tier caching reduces API calls and improves performance
- **Model Routing**: Intelligent model selection based on query complexity
- **Async Support**: Prepared for asynchronous operations

### Production Readiness
- **Logging**: Structured logging with multiple handlers
- **Error Tracking**: Sentry integration for production monitoring
- **Metrics**: Prometheus metrics for performance monitoring
- **Configuration**: Environment-based configuration management

### Resource Management
- **Memory Optimization**: Efficient caching with LRU eviction
- **GPU Support**: Optional GPU acceleration for local models
- **Fallback Systems**: CPU-based alternatives when GPU unavailable

The system is designed to handle industrial document analysis workloads with high reliability, featuring multiple fallback mechanisms and graceful degradation when external services are unavailable.