# DigiTwin RAG System v1.0

## Overview
DigiTwin is a production-ready Retrieval-Augmented Generation (RAG) system designed for industrial inspection and document analysis. Built specifically for FPSO (Floating Production Storage and Offloading) operations, offshore safety protocols, and regulatory compliance monitoring.

## Key Features

### 🔧 **Industrial-Grade RAG System**
- Advanced document processing for PDFs, Excel files, and text documents
- FAISS vector store integration for high-performance semantic search
- Multi-provider AI model support (OpenAI, Cerebras, xAI, local models)
- Specialized prompts for industrial safety and compliance analysis

### 📊 **FPSO Operations Analytics**
- Automated analysis of notification data across FPSOs (GIR, DAL, PAZ, CLV)
- Dynamic pivot table generation for work center and notification type analysis
- Individual FPSO filtering and comparative analytics
- Top performer identification and trend analysis

### 🚀 **Performance Optimized**
- Multi-tier caching system (LRU, TTL) for optimal response times
- PostgreSQL database integration with persistent data storage
- Auto-loading of cached datasets (19,778+ notification records)
- Smart query optimization and connection pooling

### 💾 **Persistent Data Management**
- Comprehensive database schema with 13 tables for data persistence
- Automatic file caching to eliminate redundant uploads
- Complete DataFrame restoration from cached data
- Cross-session data continuity

## Technical Architecture

### Frontend
- **Streamlit** web interface with tabbed navigation
- Real-time streaming responses
- Interactive pivot table dashboards
- Enhanced UI with work center filtering

### Backend
- **Python-based** RAG architecture
- **PostgreSQL** database for persistent storage
- **FAISS** vector similarity search
- **LangChain** document processing pipeline
- **SQLAlchemy** ORM for database operations

### AI Integration
- **OpenAI GPT** models for advanced reasoning
- **Cerebras** high-performance inference
- **xAI Grok** latest AI capabilities
- **HuggingFace** embeddings and local model support

## Data Processing

### Supported Formats
- **PDF**: Technical documents, reports, procedures
- **Excel**: Notification data, inspection records, operational metrics
- **Text**: Protocols, guidelines, specifications
- **DOCX**: Documentation and compliance materials

### FPSO-Specific Features
- Notification type classification (NI, NC, TBR, TEMP)
- Work center analysis and priority matrices
- Fluid class categorization (A, B, C, D)
- Corrosion type identification (internal/external)
- Repair type recommendations

## Installation & Setup

### Prerequisites
```bash
Python 3.11+
PostgreSQL database
Required packages in pyproject.toml
```

### Environment Variables
```bash
DATABASE_URL=postgresql://...
OPENAI_API_KEY=your_openai_key
CEREBRAS_API_KEY=your_cerebras_key
XAI_API_KEY=your_xai_key
```

### Quick Start
1. Clone repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up database and environment variables
4. Run: `streamlit run app.py --server.port 5000`

## Usage

### Document Upload
- Upload PDFs, Excel files, or text documents through the sidebar
- System automatically processes and caches files in database
- Vector embeddings generated for semantic search

### FPSO Analysis
- Select FPSO tab (All, GIR, DAL, PAZ, CLV) for targeted analysis
- Choose work center from dropdown for focused insights
- Generate pivot tables showing notification patterns
- View top performers and trend analysis

### AI Chat Interface
- Natural language queries about uploaded documents
- FPSO-aware responses with contextual data
- Streaming responses with industrial expertise
- Integration with pivot analysis data

## Key Components

### Core Modules
- `vector_store.py`: Enhanced vector storage with FAISS and database integration
- `app.py`: Main Streamlit application with UI and workflow management
- `core/`: Modular components for document processing, model management, query routing
- `utils/`: Performance optimization, caching, configuration management

### Database Schema
- **uploaded_files**: File metadata and upload tracking
- **cached_dataframes**: Persistent DataFrame storage
- **processed_documents**: Document chunks for vector search
- **pivot_cache**: Cached analysis results
- **query_history**: Chat and analysis history

## Performance Metrics

### System Capabilities
- **Dataset Size**: 19,778+ notification records automatically loaded
- **Response Time**: <2 seconds for cached queries
- **Vector Search**: Sub-second similarity search across documents
- **Concurrent Users**: Optimized for multi-user access
- **Memory Usage**: Efficient caching with automatic cleanup

### Caching Strategy
- **L1 Cache**: In-memory for frequently accessed data
- **L2 Cache**: Database persistence for DataFrames and analyses
- **L3 Cache**: Vector embeddings and processed documents

## Production Deployment

### Replit Deployment
- Configured for Replit hosting with proper port binding (5000)
- Environment variables managed through Replit secrets
- Automatic workflow configuration for continuous operation

### Scalability Features
- Connection pooling for database operations
- Efficient memory management with LRU eviction
- Graceful degradation when services unavailable
- Comprehensive error handling and recovery

## Version History

### v1.0 (August 2025)
- Complete RAG system with FPSO-specific functionality
- PostgreSQL integration with persistent caching
- Performance optimizations with multi-tier caching
- Enhanced UI with tabbed interface and filtering
- Production-ready deployment configuration

## Contributing

### Development Setup
1. Fork repository
2. Create feature branch
3. Implement changes with tests
4. Update documentation
5. Submit pull request

### Code Standards
- Python PEP 8 compliance
- Comprehensive error handling
- Detailed logging throughout
- Type hints for all functions
- Documentation for public APIs

## License

MIT License - see LICENSE file for details

## Support

For technical support or feature requests:
- GitHub Issues: https://github.com/valonys/LLM_notifs/issues
- Documentation: See `replit.md` for detailed technical architecture
- Contact: Valonys Engineering Team

---

**Built with ❤️ for Industrial Operations Excellence**