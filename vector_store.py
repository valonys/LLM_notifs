import os
import pandas as pd
import numpy as np
from langchain.schema import Document as LCDocument
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging
from datetime import datetime
import json
import sqlalchemy
from sqlalchemy import create_engine, text, MetaData, Table, Column, Integer, String, Text, DateTime, LargeBinary
from sqlalchemy.dialects.postgresql import BYTEA
import pickle

# Simple text storage fallback when FAISS is not available
class SimpleTextStore:
    def __init__(self, documents):
        self.documents = documents
        self.texts = [doc.page_content for doc in documents]
        self.metadata = [doc.metadata for doc in documents]
    
    def similarity_search(self, query, k=5):
        """Simple keyword-based search as fallback"""
        try:
            query_lower = query.lower()
            results = []
            
            for i, text in enumerate(self.texts):
                text_lower = text.lower()
                # Simple keyword matching
                if query_lower in text_lower:
                    results.append(LCDocument(
                        page_content=text,
                        metadata=self.metadata[i]
                    ))
            
            # If no exact matches, return first k documents
            if not results:
                results = [LCDocument(
                    page_content=text,
                    metadata=self.metadata[i]
                ) for i, text in enumerate(self.texts[:k])]
            
            return results[:k]
        except Exception as e:
            logger.error(f"Error in simple text search: {str(e)}")
            return []

logger = logging.getLogger(__name__)

# Simple fallback embedding class
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

class DatabaseManager:
    """Enhanced database manager for persistent data storage"""
    
    def __init__(self):
        self.database_url = os.getenv('DATABASE_URL')
        self.engine = None
        self.metadata = MetaData()
        
        if self.database_url:
            try:
                self.engine = create_engine(self.database_url)
                self.setup_tables()
                logger.info("Database connection established")
            except Exception as e:
                logger.error(f"Database connection failed: {str(e)}")
                self.engine = None
        else:
            logger.warning("No database URL provided")
    
    def setup_tables(self):
        """Create necessary tables for data persistence"""
        if not self.engine:
            return
        
        try:
            with self.engine.connect() as conn:
                # Table for uploaded files
                conn.execute(text("""
                CREATE TABLE IF NOT EXISTS uploaded_files (
                    id SERIAL PRIMARY KEY,
                    file_name VARCHAR(255) NOT NULL UNIQUE,
                    file_type VARCHAR(50) NOT NULL,
                    file_data BYTEA NOT NULL,
                    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    file_size INTEGER,
                    checksum VARCHAR(64)
                )
                """))
                
                # Table for processed documents
                conn.execute(text("""
                CREATE TABLE IF NOT EXISTS processed_documents (
                    id SERIAL PRIMARY KEY,
                    file_name VARCHAR(255) NOT NULL,
                    chunk_id INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    metadata JSONB,
                    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (file_name) REFERENCES uploaded_files(file_name)
                )
                """))
                
                # Table for pivot table cache
                conn.execute(text("""
                CREATE TABLE IF NOT EXISTS pivot_cache (
                    id SERIAL PRIMARY KEY,
                    file_name VARCHAR(255) NOT NULL,
                    pivot_type VARCHAR(100) NOT NULL,
                    pivot_data JSONB NOT NULL,
                    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(file_name, pivot_type)
                )
                """))
                
                # Table for embeddings cache
                conn.execute(text("""
                CREATE TABLE IF NOT EXISTS embeddings_cache (
                    id SERIAL PRIMARY KEY,
                    content_hash VARCHAR(64) NOT NULL UNIQUE,
                    embedding_vector BYTEA NOT NULL,
                    model_name VARCHAR(100) NOT NULL,
                    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """))
                
                conn.commit()
                logger.info("Database tables created successfully")
                
        except Exception as e:
            logger.error(f"Error setting up database tables: {str(e)}")
    
    def save_pivot_data(self, file_name, pivot_type, pivot_data):
        """Save pivot table data to database"""
        if not self.engine:
            return False
        
        try:
            # Convert DataFrame to JSON
            if isinstance(pivot_data, pd.DataFrame):
                pivot_json = pivot_data.to_json(orient='records')
            else:
                pivot_json = json.dumps(pivot_data)
            
            with self.engine.connect() as conn:
                conn.execute(text("""
                INSERT INTO pivot_cache (file_name, pivot_type, pivot_data)
                VALUES (:file_name, :pivot_type, :pivot_data)
                ON CONFLICT (file_name, pivot_type)
                DO UPDATE SET pivot_data = :pivot_data, created_date = CURRENT_TIMESTAMP
                """), {
                    'file_name': file_name,
                    'pivot_type': pivot_type,
                    'pivot_data': pivot_json
                })
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error saving pivot data: {str(e)}")
            return False
    
    def load_pivot_data(self, file_name, pivot_type):
        """Load pivot table data from database"""
        if not self.engine:
            return None
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                SELECT pivot_data FROM pivot_cache 
                WHERE file_name = :file_name AND pivot_type = :pivot_type
                """), {
                    'file_name': file_name,
                    'pivot_type': pivot_type
                })
                row = result.fetchone()
                if row:
                    return json.loads(row[0])
            return None
        except Exception as e:
            logger.error(f"Error loading pivot data: {str(e)}")
            return None

class PivotTableManager:
    """Handles pivot table operations for large datasets with database persistence"""
    
    def __init__(self, db_manager=None):
        self.pivot_cache = {}
        self.data_cache = {}
        self.db_manager = db_manager or DatabaseManager()
    
    def create_pivot_table(self, df, index_cols, values_cols=None, aggfunc='count', file_name=None):
        """
        Create pivot table with specified columns and cache to database
        
        Args:
            df: DataFrame
            index_cols: List of columns to use as index
            values_cols: List of columns to aggregate (optional)
            aggfunc: Aggregation function ('count', 'sum', 'mean', etc.)
            file_name: Name of the source file for caching
        """
        try:
            # Clean column names
            df_clean = df.copy()
            df_clean.columns = df_clean.columns.str.strip()
            
            # Ensure all index columns exist
            available_cols = [col for col in index_cols if col in df_clean.columns]
            if not available_cols:
                raise ValueError(f"None of the specified index columns {index_cols} found in DataFrame")
            
            # Create cache key
            cache_key = f"{'-'.join(available_cols)}_{aggfunc}"
            
            # Check database cache first
            if file_name and self.db_manager:
                cached_data = self.db_manager.load_pivot_data(file_name, cache_key)
                if cached_data:
                    logger.info(f"Loading pivot table from database cache: {cache_key}")
                    return pd.DataFrame(cached_data)
            
            # Create pivot table
            if values_cols:
                available_values = [col for col in values_cols if col in df_clean.columns]
                if available_values:
                    pivot_table = pd.pivot_table(
                        df_clean, 
                        index=available_cols,
                        values=available_values,
                        aggfunc=aggfunc,
                        fill_value=0
                    )
                else:
                    # Fallback to count if no valid value columns
                    pivot_table = pd.pivot_table(
                        df_clean,
                        index=available_cols,
                        aggfunc='count',
                        fill_value=0
                    )
            else:
                # Count occurrences if no values specified
                pivot_table = pd.pivot_table(
                    df_clean,
                    index=available_cols,
                    aggfunc='count',
                    fill_value=0
                )
            
            # Cache to database
            if file_name and self.db_manager:
                self.db_manager.save_pivot_data(file_name, cache_key, pivot_table)
            
            return pivot_table
            
        except Exception as e:
            logger.error(f"Error creating pivot table: {str(e)}")
            return None
    
    def get_summary_stats(self, df, target_columns):
        """Get summary statistics for specified columns"""
        try:
            stats = {}
            for col in target_columns:
                if col in df.columns:
                    if df[col].dtype in ['object', 'string']:
                        stats[col] = {
                            'unique_count': df[col].nunique(),
                            'top_values': df[col].value_counts().head(5).to_dict(),
                            'null_count': df[col].isnull().sum()
                        }
                    else:
                        stats[col] = {
                            'mean': df[col].mean() if df[col].dtype in ['int64', 'float64'] else None,
                            'min': df[col].min() if df[col].dtype in ['int64', 'float64'] else None,
                            'max': df[col].max() if df[col].dtype in ['int64', 'float64'] else None,
                            'unique_count': df[col].nunique(),
                            'null_count': df[col].isnull().sum()
                        }
            return stats
        except Exception as e:
            logger.error(f"Error getting summary stats: {str(e)}")
            return {}
    
    def filter_data_by_date_range(self, df, date_column, start_date=None, end_date=None):
        """Filter data by date range"""
        try:
            if date_column not in df.columns:
                return df
            
            # Convert date column to datetime
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
            
            if start_date:
                df = df[df[date_column] >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df[date_column] <= pd.to_datetime(end_date)]
            
            return df
        except Exception as e:
            logger.error(f"Error filtering by date range: {str(e)}")
            return df

class VectorStore:
    def __init__(self):
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
        except Exception as e:
            logger.warning(f"HuggingFace embeddings failed: {str(e)}. Using simple fallback.")
            self.embeddings = SimpleEmbeddings()
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Initialize managers
        self.db_manager = DatabaseManager()
        self.pivot_manager = PivotTableManager(self.db_manager)
        
        # Store processed data
        self.processed_data = None
        self.original_df = None
        self.current_file_name = None
        self.vector_store = None  # FAISS or other vector store
        self.all_documents = []  # Keep track of all documents for multi-report RAG
    
    def process_excel_to_documents(self, uploaded_files, target_columns=None):
        """
        Process Excel files and convert them to LangChain documents
        Enhanced for large datasets with specific column handling and database persistence
        """
        documents = []
        
        # Handle single file or list of files
        if not isinstance(uploaded_files, list):
            uploaded_files = [uploaded_files]
        
        for uploaded_file in uploaded_files:
            try:
                # Check if it's a file-like object with name attribute
                if hasattr(uploaded_file, 'name'):
                    file_name = uploaded_file.name
                else:
                    file_name = "excel_file.xlsx"
                
                self.current_file_name = file_name
                
                if file_name.endswith('.xlsx') or file_name.endswith('.xls'):
                    # Read Excel file with specific sheet if available
                    try:
                        # Try to read 'Global Notifications' sheet first
                        df = pd.read_excel(uploaded_file, sheet_name='Global Notifications')
                        sheet_name = 'Global Notifications'
                    except:
                        # Fallback to first sheet
                        df = pd.read_excel(uploaded_file)
                        sheet_name = 'Sheet1'
                    
                    # Store original DataFrame for pivot operations
                    self.original_df = df.copy()
                    
                    # Clean column names
                    df.columns = df.columns.str.strip()
                    
                    # Handle target columns if specified
                    if target_columns:
                        available_cols = [col for col in target_columns if col in df.columns]
                        if available_cols:
                            df = df[available_cols]
                    
                    # Create multiple document chunks for large datasets
                    chunk_size = 5000  # Process 5000 rows per chunk
                    total_rows = len(df)
                    
                    for i in range(0, total_rows, chunk_size):
                        chunk_df = df.iloc[i:i+chunk_size]
                        
                        # Convert chunk to text with better formatting
                        text_content = self._format_dataframe_to_text(chunk_df, i, total_rows)
                        
                        # Create document with enhanced metadata
                        doc = LCDocument(
                            page_content=text_content,
                            metadata={
                                "source": file_name,
                                "file_type": "excel",
                                "sheet_name": sheet_name,
                                "chunk_start": i,
                                "chunk_end": min(i + chunk_size, total_rows),
                                "total_rows": total_rows,
                                "columns": list(df.columns),
                                "chunk_size": len(chunk_df)
                            }
                        )
                        documents.append(doc)
                    
                    # Store processed data for pivot operations
                    self.processed_data = df
                    
                    # Add documents to all_documents for vector store
                    self.all_documents.extend(documents)
                    
                    # Save documents to database for persistence
                    self._save_documents_to_db(documents, file_name)
                    
                    # Create/update enhanced vector store
                    self.create_enhanced_vector_store()
                    
            except Exception as e:
                try:
                    file_name = getattr(uploaded_file, 'name', 'unknown_file')
                except:
                    file_name = 'unknown_file'
                logger.error(f"Error processing file {file_name}: {str(e)}")
                continue
        
        return documents
    
    def _save_documents_to_db(self, documents, file_name):
        """Save processed documents to database"""
        if not self.db_manager or not self.db_manager.engine:
            return
        
        try:
            with self.db_manager.engine.connect() as conn:
                # Ensure file exists in uploaded_files table first
                conn.execute(text("""
                INSERT INTO uploaded_files (file_name, file_type, file_data, upload_date, file_size, checksum)
                VALUES (:file_name, :file_type, :file_data, :upload_date, :file_size, :checksum)
                ON CONFLICT (file_name) DO NOTHING
                """), {
                    'file_name': file_name,
                    'file_type': 'excel',
                    'file_data': b'',  # Empty for now
                    'upload_date': datetime.now(),
                    'file_size': 0,
                    'checksum': 'placeholder'
                })
                
                # Clear existing documents for this file
                conn.execute(text("""
                DELETE FROM processed_documents WHERE file_name = :file_name
                """), {'file_name': file_name})
                
                # Insert new documents
                for i, doc in enumerate(documents):
                    conn.execute(text("""
                    INSERT INTO processed_documents (file_name, chunk_id, content, metadata)
                    VALUES (:file_name, :chunk_id, :content, :metadata)
                    """), {
                        'file_name': file_name,
                        'chunk_id': i,
                        'content': doc.page_content,
                        'metadata': json.dumps(doc.metadata)
                    })
                
                conn.commit()
                logger.info(f"Saved {len(documents)} documents to database for file: {file_name}")
                
        except Exception as e:
            logger.error(f"Error saving documents to database: {str(e)}")
    
    def _format_dataframe_to_text(self, df, chunk_start, total_rows):
        """Format DataFrame chunk to readable text"""
        try:
            # Create a more readable text format
            lines = []
            lines.append(f"Data chunk {chunk_start//5000 + 1} of {(total_rows-1)//5000 + 1}")
            lines.append(f"Rows {chunk_start+1} to {min(chunk_start + len(df), total_rows)} of {total_rows}")
            lines.append("=" * 50)
            
            # Add column headers
            headers = " | ".join(df.columns)
            lines.append(headers)
            lines.append("-" * len(headers))
            
            # Add sample rows (first 10 rows for readability)
            sample_size = min(10, len(df))
            for idx, row in df.head(sample_size).iterrows():
                row_text = " | ".join([str(val) if pd.notna(val) else "N/A" for val in row])
                lines.append(row_text)
            
            if len(df) > sample_size:
                lines.append(f"... and {len(df) - sample_size} more rows")
            
            return "\n".join(lines)
            
        except Exception as e:
            logger.error(f"Error formatting DataFrame to text: {str(e)}")
            return df.to_string(index=False)
    
    def create_notification_pivot_tables(self, date_range=None, fpso_filter=None):
        """
        Create simplified pivot table: Notification Type vs Main Work Center, filtered by FPSO
        Drop empty rows and focus on core operational data
        """
        if self.processed_data is None:
            logger.warning("No processed data available for pivot table creation")
            return None
        
        try:
            df = self.processed_data.copy()
            
            # Check for required columns (using exact column names from data)
            required_columns = ['Notifictn type', 'Main WorkCtr']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.warning(f"Missing required columns: {missing_columns}. Available: {list(df.columns)}")
                return None
            
            # Remove rows where key columns are empty
            df_clean = df.dropna(subset=required_columns, how='any')  # Drop if either is empty
            df_clean = df_clean[
                (df_clean['Notifictn type'].astype(str).str.strip() != '') & 
                (df_clean['Main WorkCtr'].astype(str).str.strip() != '') &
                (df_clean['Notifictn type'].astype(str).str.strip() != 'nan') &
                (df_clean['Main WorkCtr'].astype(str).str.strip() != 'nan')
            ]
            
            logger.info(f"Cleaned data: {len(df)} -> {len(df_clean)} rows (removed empty entries)")
            
            # Apply FPSO filter if specified
            if fpso_filter and fpso_filter != "All FPSOs":
                if 'FPSO' in df_clean.columns:
                    original_count = len(df_clean)
                    df_clean = df_clean[df_clean['FPSO'].str.contains(fpso_filter, case=False, na=False)]
                    logger.info(f"FPSO filter applied: {original_count} -> {len(df_clean)} rows for {fpso_filter}")
                else:
                    logger.warning("FPSO column not found - cannot apply FPSO filter")
            
            if len(df_clean) == 0:
                logger.warning("No data remaining after cleaning and filtering")
                return None
            
            # Create single pivot table: Notification Type vs Work Center
            try:
                pivot_table = pd.crosstab(
                    df_clean['Notifictn type'], 
                    df_clean['Main WorkCtr'], 
                    margins=True, 
                    margins_name="Total"
                )
                
                logger.info(f"Created pivot table: {len(pivot_table)-1} notification types × {len(pivot_table.columns)-1} work centers")
                
                # Package results
                result = {
                    'pivot_table': pivot_table,
                    'fpso_filter': fpso_filter or "All FPSOs",
                    'total_records': len(df_clean),
                    'notification_types_count': len(pivot_table.index) - 1,  # Exclude Total row
                    'work_centers_count': len(pivot_table.columns) - 1,  # Exclude Total column
                    'summary_stats': {
                        'top_notification_type': pivot_table.iloc[:-1, -1].idxmax(),  # Exclude Total row/col
                        'top_work_center': pivot_table.iloc[-1, :-1].idxmax(),
                        'total_notifications': pivot_table.iloc[-1, -1]  # Grand total
                    }
                }
                
                # Convert to documents for RAG integration
                self._convert_simplified_pivot_to_documents(result, fpso_filter)
                
                return result
                
            except Exception as e:
                logger.error(f"Error creating pivot table: {str(e)}")
                return None
            
        except Exception as e:
            logger.error(f"Error in create_notification_pivot_tables: {str(e)}")
            return None
    
    def _convert_pivot_tables_to_documents(self, pivot_tables, fpso_filter=None):
        """Convert pivot table results into searchable documents for vector store"""
        try:
            pivot_documents = []
            
            for table_name, pivot_data in pivot_tables.items():
                if pivot_data is None or len(pivot_data) == 0:
                    continue
                
                # Create a comprehensive text summary of the pivot table
                summary_text = self._create_pivot_summary_text(table_name, pivot_data, fpso_filter)
                
                # Create metadata for the document
                metadata = {
                    "source": f"pivot_analysis_{self.current_file_name}",
                    "type": "pivot_table",
                    "analysis_name": table_name,
                    "fpso_focus": fpso_filter or "All FPSOs",
                    "data_points": len(pivot_data),
                    "generated_on": datetime.now().isoformat()
                }
                
                # Create LangChain document
                doc = LCDocument(
                    page_content=summary_text,
                    metadata=metadata
                )
                pivot_documents.append(doc)
                
                # Create detailed breakdowns for larger pivot tables
                if len(pivot_data) > 10:
                    detail_docs = self._create_detailed_pivot_documents(table_name, pivot_data, fpso_filter)
                    pivot_documents.extend(detail_docs)
            
            # Store pivot documents in vector store if available
            if pivot_documents and hasattr(self, 'vector_store') and self.vector_store:
                try:
                    # Add to existing vector store
                    self.vector_store.add_documents(pivot_documents)
                    logger.info(f"Added {len(pivot_documents)} pivot analysis documents to vector store")
                except Exception as e:
                    logger.warning(f"Could not add pivot documents to vector store: {str(e)}")
            
            # Add pivot documents to all_documents for integrated RAG
            if pivot_documents:
                self.all_documents.extend(pivot_documents)
                
                # Update/create vector store with new pivot documents
                self.create_enhanced_vector_store()
                
                # Save pivot documents to database for persistence
                self._save_documents_to_db(pivot_documents, f"pivot_analysis_{self.current_file_name}")
                
                logger.info(f"Successfully integrated {len(pivot_documents)} pivot analysis documents into RAG system")
            
        except Exception as e:
            logger.error(f"Error converting pivot tables to documents: {str(e)}")
    
    def _create_pivot_summary_text(self, table_name, pivot_data, fpso_filter):
        """Create a comprehensive text summary of pivot table data"""
        try:
            text_parts = []
            
            # Header
            focus_text = f" for {fpso_filter}" if fpso_filter and fpso_filter != "All FPSOs" else ""
            text_parts.append(f"PIVOT ANALYSIS: {table_name}{focus_text}")
            text_parts.append("=" * 50)
            
            # Convert pivot data to DataFrame if it's not already
            if isinstance(pivot_data, dict):
                df = pd.DataFrame(pivot_data)
            else:
                df = pivot_data
            
            # Summary statistics
            text_parts.append(f"Total entries: {len(df)}")
            
            # Top entries analysis
            if not df.empty:
                # For series data (single column)
                if isinstance(df, pd.Series) or len(df.columns) == 1:
                    series_data = df if isinstance(df, pd.Series) else df.iloc[:, 0]
                    total = series_data.sum() if series_data.dtype in ['int64', 'float64'] else len(series_data)
                    text_parts.append(f"Total count: {total}")
                    
                    # Top 5 entries
                    top_entries = series_data.nlargest(5) if series_data.dtype in ['int64', 'float64'] else series_data.head(5)
                    text_parts.append("\nTop entries:")
                    for idx, value in top_entries.items():
                        percentage = (value / total * 100) if total > 0 else 0
                        text_parts.append(f"- {idx}: {value} ({percentage:.1f}%)")
                
                # For multi-column data
                else:
                    text_parts.append(f"Columns: {', '.join(df.columns)}")
                    # Sample of the data
                    text_parts.append("\nSample data:")
                    for i, (idx, row) in enumerate(df.head(5).iterrows()):
                        row_text = f"- {idx}: " + ", ".join([f"{col}={val}" for col, val in row.items()])
                        text_parts.append(row_text)
            
            # Insights and patterns
            insights = self._generate_pivot_insights(table_name, df, fpso_filter)
            if insights:
                text_parts.append("\nKey Insights:")
                text_parts.extend([f"- {insight}" for insight in insights])
            
            return "\n".join(text_parts)
            
        except Exception as e:
            logger.error(f"Error creating pivot summary text: {str(e)}")
            return f"Pivot Analysis: {table_name} - Error in summary generation"
    
    def _create_detailed_pivot_documents(self, table_name, pivot_data, fpso_filter):
        """Create detailed documents for large pivot tables"""
        detail_docs = []
        
        try:
            if isinstance(pivot_data, dict):
                df = pd.DataFrame(pivot_data)
            else:
                df = pivot_data
            
            # Split large tables into chunks
            chunk_size = 20
            for i in range(0, len(df), chunk_size):
                chunk_df = df.iloc[i:i+chunk_size]
                
                # Create detailed text for this chunk
                chunk_text = f"DETAILED DATA - {table_name} (Part {i//chunk_size + 1})\n"
                chunk_text += f"FPSO Focus: {fpso_filter or 'All FPSOs'}\n\n"
                
                # Add detailed entries
                for idx, row in chunk_df.iterrows():
                    if isinstance(row, pd.Series):
                        chunk_text += f"{idx}: {row.iloc[0] if len(row) > 0 else 'N/A'}\n"
                    else:
                        row_details = ", ".join([f"{col}={val}" for col, val in row.items()])
                        chunk_text += f"{idx}: {row_details}\n"
                
                # Create document
                metadata = {
                    "source": f"pivot_detail_{self.current_file_name}",
                    "type": "pivot_detail",
                    "analysis_name": table_name,
                    "fpso_focus": fpso_filter or "All FPSOs",
                    "chunk_number": i//chunk_size + 1,
                    "data_points": len(chunk_df)
                }
                
                doc = LCDocument(page_content=chunk_text, metadata=metadata)
                detail_docs.append(doc)
        
        except Exception as e:
            logger.error(f"Error creating detailed pivot documents: {str(e)}")
        
        return detail_docs
    
    def _generate_pivot_insights(self, table_name, df, fpso_filter):
        """Generate intelligent insights from pivot data"""
        insights = []
        
        try:
            if df.empty:
                return insights
            
            # Insights based on table type and data patterns
            if "FPSO" in table_name:
                if isinstance(df, pd.Series):
                    top_fpso = df.idxmax()
                    insights.append(f"Highest activity in {top_fpso} with {df.max()} notifications")
                    if len(df) > 1:
                        total = df.sum()
                        insights.append(f"Distribution across {len(df)} FPSOs, total: {total}")
            
            elif "Work Center" in table_name:
                if isinstance(df, pd.Series):
                    top_workctr = df.idxmax()
                    insights.append(f"Most active work center: {top_workctr} ({df.max()} notifications)")
                    if fpso_filter and fpso_filter != "All FPSOs":
                        insights.append(f"Work center analysis specific to {fpso_filter} operations")
            
            elif "Type" in table_name:
                if isinstance(df, pd.Series):
                    top_type = df.idxmax()
                    insights.append(f"Most common notification type: {top_type} ({df.max()} occurrences)")
                    
            elif "Monthly" in table_name:
                if isinstance(df, pd.Series) and len(df) > 1:
                    recent_months = df.tail(3)
                    avg_recent = recent_months.mean()
                    insights.append(f"Recent 3-month average: {avg_recent:.1f} notifications per month")
            
            elif "Completion" in table_name:
                if isinstance(df, pd.Series):
                    if True in df.index:
                        completed = df.get(True, 0)
                        total = df.sum()
                        completion_rate = (completed / total * 100) if total > 0 else 0
                        insights.append(f"Completion rate: {completion_rate:.1f}% ({completed} of {total})")
            
        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}")
        
        return insights
    
    def _convert_simplified_pivot_to_documents(self, result, fpso_filter=None):
        """Convert simplified pivot table result to searchable documents for RAG"""
        try:
            pivot_documents = []
            pivot_table = result['pivot_table']
            
            if pivot_table is None or len(pivot_table) == 0:
                return
            
            # Create main summary document
            summary_text = self._create_simplified_pivot_summary(result, fpso_filter)
            
            metadata = {
                "source": f"pivot_analysis_{self.current_file_name}",
                "type": "simplified_pivot",
                "fpso_focus": fpso_filter or "All FPSOs",
                "notification_types": result.get('notification_types_count', 0),
                "work_centers": result.get('work_centers_count', 0),
                "total_records": result.get('total_records', 0),
                "generated_on": datetime.now().isoformat()
            }
            
            doc = LCDocument(page_content=summary_text, metadata=metadata)
            pivot_documents.append(doc)
            
            # Create detailed breakdown documents for top performers
            top_combinations = self._get_top_combinations(pivot_table)
            if top_combinations:
                detail_text = f"TOP PERFORMING COMBINATIONS - {fpso_filter or 'All FPSOs'}\n"
                detail_text += "=" * 50 + "\n\n"
                
                for i, (notif_type, work_center, count) in enumerate(top_combinations[:10], 1):
                    detail_text += f"{i}. {notif_type} at {work_center}: {count} notifications\n"
                
                detail_doc = LCDocument(
                    page_content=detail_text,
                    metadata={**metadata, "type": "pivot_details"}
                )
                pivot_documents.append(detail_doc)
            
            # Add to vector store system
            if pivot_documents:
                self.all_documents.extend(pivot_documents)
                self.create_enhanced_vector_store()
                self._save_documents_to_db(pivot_documents, f"pivot_simplified_{self.current_file_name}")
                logger.info(f"Integrated {len(pivot_documents)} simplified pivot documents into RAG system")
            
        except Exception as e:
            logger.error(f"Error converting simplified pivot to documents: {str(e)}")
    
    def _create_simplified_pivot_summary(self, result, fpso_filter):
        """Create comprehensive text summary of simplified pivot analysis"""
        try:
            pivot_table = result['pivot_table']
            stats = result['summary_stats']
            
            text_parts = []
            focus_text = f" for {fpso_filter}" if fpso_filter and fpso_filter != "All FPSOs" else ""
            text_parts.append(f"SIMPLIFIED PIVOT ANALYSIS: Notification Types vs Work Centers{focus_text}")
            text_parts.append("=" * 60)
            
            # Key statistics
            text_parts.append(f"Total notifications analyzed: {stats['total_notifications']}")
            text_parts.append(f"Notification types covered: {result['notification_types_count']}")
            text_parts.append(f"Work centers involved: {result['work_centers_count']}")
            text_parts.append(f"Data records processed: {result['total_records']}")
            
            # Top performers
            text_parts.append(f"\nTop notification type: {stats['top_notification_type']}")
            text_parts.append(f"Busiest work center: {stats['top_work_center']}")
            
            # Pivot table sample (top 5x5)
            text_parts.append("\nPIVOT TABLE SAMPLE (Top 5x5):")
            text_parts.append("-" * 40)
            
            # Get top 5 notification types and work centers by total
            top_notif_types = pivot_table.iloc[:-1, -1].nlargest(5).index  # Exclude Total row
            top_work_centers = pivot_table.iloc[-1, :-1].nlargest(5).index  # Exclude Total col
            
            # Create sample table
            sample_table = pivot_table.loc[top_notif_types, top_work_centers]
            text_parts.append(sample_table.to_string())
            
            # Key insights
            text_parts.append("\nKEY INSIGHTS:")
            if fpso_filter and fpso_filter != "All FPSOs":
                text_parts.append(f"- Analysis focused specifically on {fpso_filter} operations")
            text_parts.append(f"- Highest concentration: {stats['top_notification_type']} notifications")
            text_parts.append(f"- Most active area: {stats['top_work_center']} work center")
            
            return "\n".join(text_parts)
            
        except Exception as e:
            logger.error(f"Error creating simplified pivot summary: {str(e)}")
            return f"Simplified Pivot Analysis - Error in summary generation: {str(e)}"
    
    def _get_top_combinations(self, pivot_table, top_n=10):
        """Get top notification type + work center combinations"""
        try:
            combinations = []
            
            # Exclude Total row and column
            data_section = pivot_table.iloc[:-1, :-1]
            
            for notif_type in data_section.index:
                for work_center in data_section.columns:
                    count = data_section.loc[notif_type, work_center]
                    if count > 0:  # Only include non-zero combinations
                        combinations.append((notif_type, work_center, count))
            
            # Sort by count descending
            combinations.sort(key=lambda x: x[2], reverse=True)
            
            return combinations[:top_n]
            
        except Exception as e:
            logger.error(f"Error getting top combinations: {str(e)}")
            return []
    
    def create_enhanced_vector_store(self):
        """Create or update vector store with all documents (data + pivot analysis)"""
        try:
            if not self.all_documents:
                logger.warning("No documents available for vector store creation")
                return None
            
            # Create or update FAISS vector store
            if self.vector_store is None:
                try:
                    self.vector_store = FAISS.from_documents(self.all_documents, self.embeddings)
                    logger.info(f"Created new vector store with {len(self.all_documents)} documents")
                except Exception as e:
                    logger.warning(f"FAISS creation failed: {str(e)}. Using simple text store.")
                    self.vector_store = SimpleTextStore(self.all_documents)
            else:
                # Add new documents to existing vector store
                try:
                    # Get only new documents that aren't already in vector store
                    new_docs = [doc for doc in self.all_documents if not hasattr(doc, '_in_vector_store')]
                    if new_docs:
                        self.vector_store.add_documents(new_docs)
                        # Mark documents as added
                        for doc in new_docs:
                            doc._in_vector_store = True
                        logger.info(f"Added {len(new_docs)} new documents to vector store")
                except Exception as e:
                    logger.warning(f"Could not add new documents to vector store: {str(e)}")
            
            return self.vector_store
            
        except Exception as e:
            logger.error(f"Error creating enhanced vector store: {str(e)}")
            return None
    
    def query_multi_report_rag(self, query, fpso_filter=None, k=5):
        """Enhanced RAG query that searches across all data sources and pivot analysis"""
        try:
            if not self.vector_store:
                logger.warning("No vector store available for querying")
                return []
            
            # Perform similarity search
            try:
                if hasattr(self.vector_store, 'similarity_search'):
                    docs = self.vector_store.similarity_search(query, k=k*2)  # Get more docs for filtering
                else:
                    docs = self.vector_store.similarity_search(query, k)
            except Exception as e:
                logger.warning(f"Vector search failed: {str(e)}")
                return []
            
            # Filter results based on FPSO if specified
            if fpso_filter and fpso_filter != "All FPSOs":
                filtered_docs = []
                for doc in docs:
                    # Check if document is relevant to the selected FPSO
                    if (fpso_filter.lower() in doc.page_content.lower() or 
                        doc.metadata.get('fpso_focus') == fpso_filter or
                        doc.metadata.get('fpso_focus') == "All FPSOs"):
                        filtered_docs.append(doc)
                docs = filtered_docs[:k]
            else:
                docs = docs[:k]
            
            # Enhance results with metadata context
            enhanced_results = []
            for doc in docs:
                # Add context about document type and source
                content = doc.page_content
                metadata = doc.metadata
                
                context_prefix = ""
                if metadata.get('type') == 'pivot_table':
                    context_prefix = f"[PIVOT ANALYSIS - {metadata.get('analysis_name', 'Unknown')}] "
                elif metadata.get('type') == 'pivot_detail':
                    context_prefix = f"[DETAILED DATA - {metadata.get('analysis_name', 'Unknown')}] "
                elif metadata.get('source', '').endswith('.xlsx'):
                    context_prefix = "[EXCEL DATA] "
                elif metadata.get('source', '').endswith('.pdf'):
                    context_prefix = "[PDF DOCUMENT] "
                
                enhanced_content = context_prefix + content
                enhanced_doc = LCDocument(page_content=enhanced_content, metadata=metadata)
                enhanced_results.append(enhanced_doc)
            
            logger.info(f"Multi-report RAG query returned {len(enhanced_results)} relevant documents")
            return enhanced_results
            
        except Exception as e:
            logger.error(f"Error in multi-report RAG query: {str(e)}")
            return []
    
    def get_comprehensive_context(self, query, fpso_filter=None):
        """Get comprehensive context combining RAG results with current analysis state"""
        try:
            context_parts = []
            
            # Get RAG results
            rag_docs = self.query_multi_report_rag(query, fpso_filter, k=5)
            if rag_docs:
                context_parts.append("=== RELEVANT DATA AND ANALYSIS ===")
                for i, doc in enumerate(rag_docs[:3], 1):
                    source_info = doc.metadata.get('source', 'Unknown')
                    context_parts.append(f"\nSource {i} ({source_info}):")
                    context_parts.append(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
            
            # Add current FPSO context
            if fpso_filter and fpso_filter != "All FPSOs":
                context_parts.append(f"\n=== CURRENT ANALYSIS FOCUS ===")
                context_parts.append(f"FPSO: {fpso_filter}")
                context_parts.append("Analysis is specifically focused on this FPSO's operations.")
            
            # Add data summary if available
            if self.processed_data is not None:
                summary = self.get_notification_summary()
                if summary:
                    context_parts.append("\n=== DATA OVERVIEW ===")
                    context_parts.append(f"Total notifications: {summary.get('total_notifications', 'Unknown')}")
                    if summary.get('fpso_distribution'):
                        context_parts.append(f"FPSO distribution: {summary['fpso_distribution']}")
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error getting comprehensive context: {str(e)}")
            return "Context retrieval failed"
    
    def get_notification_summary(self):
        """Get comprehensive summary of notification data"""
        if self.processed_data is None:
            return None
        
        try:
            df = self.processed_data
            summary = {
                'total_notifications': len(df),
                'date_range': None,
                'top_notification_types': None,
                'top_work_centers': None,
                'fpso_distribution': None
            }
            
            # Date range analysis
            if 'Created on' in df.columns:
                try:
                    df['Created on'] = pd.to_datetime(df['Created on'], errors='coerce')
                    valid_dates = df['Created on'].dropna()
                    if len(valid_dates) > 0:
                        summary['date_range'] = {
                            'start': str(valid_dates.min().date()),
                            'end': str(valid_dates.max().date())
                        }
                except:
                    pass
            
            # Top notification types
            if 'Notifictn' in df.columns:
                summary['top_notification_types'] = df['Notifictn'].value_counts().head(5).to_dict()
            
            # Top work centers
            if 'Main WorkCtr' in df.columns:
                summary['top_work_centers'] = df['Main WorkCtr'].value_counts().head(5).to_dict()
            
            # FPSO distribution
            if 'FPSO' in df.columns:
                summary['fpso_distribution'] = df['FPSO'].value_counts().to_dict()
            
            return summary
            
        except Exception as e:
            logger.error(f"Error creating notification summary: {str(e)}")
            return None
    
    def get_cached_files(self):
        """Get list of cached files from database"""
        if not self.db_manager or not self.db_manager.engine:
            return []
        
        try:
            with self.db_manager.engine.connect() as conn:
                result = conn.execute(text("""
                SELECT file_name, file_type, upload_date, file_size 
                FROM uploaded_files 
                ORDER BY upload_date DESC
                """))
                
                files = []
                for row in result:
                    files.append({
                        'filename': row[0],
                        'file_type': row[1],
                        'upload_date': row[2],
                        'file_size': row[3]
                    })
                
                return files
                
        except Exception as e:
            logger.error(f"Error getting cached files: {str(e)}")
            return []
    
    def load_cached_file(self, filename):
        """Load a cached file from database"""
        if not self.db_manager or not self.db_manager.engine:
            return None
        
        try:
            with self.db_manager.engine.connect() as conn:
                # Get file data
                result = conn.execute(text("""
                SELECT file_data, file_type FROM uploaded_files 
                WHERE file_name = :filename
                """), {'filename': filename})
                
                row = result.fetchone()
                if not row:
                    return None
                
                file_data = row[0]
                file_type = row[1]
                
                # Get processed documents
                doc_result = conn.execute(text("""
                SELECT content, metadata FROM processed_documents 
                WHERE file_name = :filename
                ORDER BY chunk_id
                """), {'filename': filename})
                
                documents = []
                for doc_row in doc_result:
                    content = doc_row[0]
                    metadata = json.loads(doc_row[1]) if doc_row[1] else {}
                    documents.append({
                        'content': content,
                        'metadata': metadata
                    })
                
                return {
                    'file_data': file_data,
                    'file_type': file_type,
                    'documents': documents
                }
                
        except Exception as e:
            logger.error(f"Error loading cached file {filename}: {str(e)}")
            return None