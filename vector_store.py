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
                    
                    # Save documents to database for persistence
                    self._save_documents_to_db(documents, file_name)
                    
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
        Create pivot tables for notification data with database persistence
        Target columns: Main Work Ctr, Notifictn type, Description, Created on, FPSO, Completn date
        """
        if self.processed_data is None:
            return None
        
        try:
            df = self.processed_data.copy()
            
            # Filter by FPSO if specified
            if fpso_filter and fpso_filter != "All FPSOs" and 'FPSO' in df.columns:
                df = df[df['FPSO'] == fpso_filter]
                logger.info(f"Filtered data for FPSO: {fpso_filter}, Rows: {len(df)}")
            
            # Filter by date range if specified
            if date_range and 'Created on' in df.columns:
                df = self.pivot_manager.filter_data_by_date_range(
                    df, 'Created on', date_range.get('start'), date_range.get('end')
                )
            
            # Define target columns for notifications (key columns as requested)
            target_columns = ['Main Work Ctr', 'Notifictn type', 'Description', 'Created on', 'FPSO', 'Completn date']
            available_columns = [col for col in target_columns if col in df.columns]
            
            if not available_columns:
                logger.warning("No target columns found in DataFrame")
                return None
            
            pivot_tables = {}
            
            # 1. Notifications by Type
            if 'Notifictn type' in available_columns:
                notif_pivot = self.pivot_manager.create_pivot_table(
                    df, ['Notifictn type'], aggfunc='count', file_name=self.current_file_name
                )
                if notif_pivot is not None:
                    pivot_tables['Notifications by Type'] = notif_pivot
            
            # 2. Notifications by Work Center (Main Work Ctr)
            if 'Main Work Ctr' in available_columns:
                workctr_pivot = self.pivot_manager.create_pivot_table(
                    df, ['Main Work Ctr'], aggfunc='count', file_name=self.current_file_name
                )
                if workctr_pivot is not None:
                    pivot_tables['Notifications by Work Center'] = workctr_pivot
            
            # 3. Notifications by FPSO
            if 'FPSO' in available_columns:
                fpso_pivot = self.pivot_manager.create_pivot_table(
                    df, ['FPSO'], aggfunc='count', file_name=self.current_file_name
                )
                if fpso_pivot is not None:
                    pivot_tables['Notifications by FPSO'] = fpso_pivot
            
            # 4. Cross-analysis: Type vs Work Center
            if 'Notifictn type' in available_columns and 'Main Work Ctr' in available_columns:
                cross_pivot = self.pivot_manager.create_pivot_table(
                    df, ['Notifictn type', 'Main Work Ctr'], aggfunc='count', file_name=self.current_file_name
                )
                if cross_pivot is not None:
                    pivot_tables['Type vs Work Center'] = cross_pivot
            
            # 5. FPSO vs Work Center Analysis
            if 'FPSO' in available_columns and 'Main Work Ctr' in available_columns:
                fpso_workctr_pivot = self.pivot_manager.create_pivot_table(
                    df, ['FPSO', 'Main Work Ctr'], aggfunc='count', file_name=self.current_file_name
                )
                if fpso_workctr_pivot is not None:
                    pivot_tables['FPSO vs Work Center'] = fpso_workctr_pivot
            
            # 6. Monthly trends if date column exists
            if 'Created on' in available_columns:
                try:
                    df_copy = df.copy()
                    df_copy['Created on'] = pd.to_datetime(df_copy['Created on'], errors='coerce')
                    df_copy['Month'] = df_copy['Created on'].dt.to_period('M').astype(str)
                    
                    monthly_pivot = self.pivot_manager.create_pivot_table(
                        df_copy, ['Month'], aggfunc='count', file_name=self.current_file_name
                    )
                    if monthly_pivot is not None:
                        pivot_tables['Monthly Trends'] = monthly_pivot
                except Exception as e:
                    logger.warning(f"Could not create monthly trends: {str(e)}")
            
            # 7. Completion Analysis (if completion date exists)
            if 'Completn date' in available_columns:
                try:
                    df_copy = df.copy()
                    df_copy['Is_Completed'] = df_copy['Completn date'].notna()
                    
                    completion_pivot = self.pivot_manager.create_pivot_table(
                        df_copy, ['Is_Completed'], aggfunc='count', file_name=self.current_file_name
                    )
                    if completion_pivot is not None:
                        pivot_tables['Completion Status'] = completion_pivot
                except Exception as e:
                    logger.warning(f"Could not create completion analysis: {str(e)}")
            
            return pivot_tables
            
        except Exception as e:
            logger.error(f"Error creating notification pivot tables: {str(e)}")
            return None
    
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