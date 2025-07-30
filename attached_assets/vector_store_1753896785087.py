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

class PivotTableManager:
    """Handles pivot table operations for large datasets"""
    
    def __init__(self):
        self.pivot_cache = {}
        self.data_cache = {}
    
    def create_pivot_table(self, df, index_cols, values_cols=None, aggfunc='count'):
        """
        Create pivot table with specified columns
        
        Args:
            df: DataFrame
            index_cols: List of columns to use as index
            values_cols: List of columns to aggregate (optional)
            aggfunc: Aggregation function ('count', 'sum', 'mean', etc.)
        """
        try:
            # Clean column names
            df_clean = df.copy()
            df_clean.columns = df_clean.columns.str.strip()
            
            # Ensure all index columns exist
            available_cols = [col for col in index_cols if col in df_clean.columns]
            if not available_cols:
                raise ValueError(f"None of the specified index columns {index_cols} found in DataFrame")
            
            # Create pivot table
            if values_cols:
                pivot_table = pd.pivot_table(
                    df_clean, 
                    index=available_cols,
                    values=values_cols,
                    aggfunc=aggfunc,
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
        
        # Initialize pivot table manager
        self.pivot_manager = PivotTableManager()
        
        # Store processed data
        self.processed_data = None
        self.original_df = None
    
    def process_excel_to_documents(self, uploaded_files, target_columns=None):
        """
        Process Excel files and convert them to LangChain documents
        Enhanced for large datasets with specific column handling
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
                
                if file_name.endswith('.xlsx') or file_name.endswith('.xls'):
                    # Read Excel file with specific sheet if available
                    try:
                        # Try to read 'Global Notifications' sheet first
                        df = pd.read_excel(uploaded_file, sheet_name='Global Notifications')
                    except:
                        # Fallback to first sheet
                        df = pd.read_excel(uploaded_file)
                    
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
                                "sheet_name": "Global Notifications",
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
                    
            except Exception as e:
                try:
                    file_name = getattr(uploaded_file, 'name', 'unknown_file')
                except:
                    file_name = 'unknown_file'
                logger.error(f"Error processing file {file_name}: {str(e)}")
                continue
        
        return documents
    
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
    
    def create_notification_pivot_tables(self, date_range=None):
        """
        Create pivot tables for notification data with specific columns:
        - Notifictn type
        - Created on
        - Description
        - Main WorkCtr
        - FPSO
        """
        if self.processed_data is None:
            return None
        
        try:
            df = self.processed_data.copy()
            
            # Filter by date range if specified
            if date_range and 'Created on' in df.columns:
                df = self.pivot_manager.filter_data_by_date_range(
                    df, 'Created on', date_range.get('start'), date_range.get('end')
                )
            
            # Define target columns for notifications
            target_columns = ['Notifictn', 'Created on', 'Description', 'Main WorkCtr', 'FPSO']
            available_columns = [col for col in target_columns if col in df.columns]
            
            pivot_results = {}
            
            # 1. Notifications by type
            if 'Notifictn' in df.columns:
                pivot_results['notifications_by_type'] = self.pivot_manager.create_pivot_table(
                    df, ['Notifictn'], aggfunc='count'
                )
            
            # 2. Notifications by Work Center
            if 'Main WorkCtr' in df.columns:
                pivot_results['notifications_by_workcenter'] = self.pivot_manager.create_pivot_table(
                    df, ['Main WorkCtr'], aggfunc='count'
                )
            
            # 3. Notifications by FPSO
            if 'FPSO' in df.columns:
                pivot_results['notifications_by_fpso'] = self.pivot_manager.create_pivot_table(
                    df, ['FPSO'], aggfunc='count'
                )
            
            # 4. Notifications by type and work center
            if 'Notifictn' in df.columns and 'Main WorkCtr' in df.columns:
                pivot_results['notifications_by_type_workcenter'] = self.pivot_manager.create_pivot_table(
                    df, ['Notifictn', 'Main WorkCtr'], aggfunc='count'
                )
            
            # 5. Notifications by type and FPSO
            if 'Notifictn' in df.columns and 'FPSO' in df.columns:
                pivot_results['notifications_by_type_fpso'] = self.pivot_manager.create_pivot_table(
                    df, ['Notifictn', 'FPSO'], aggfunc='count'
                )
            
            # 6. Summary statistics
            pivot_results['summary_stats'] = self.pivot_manager.get_summary_stats(df, available_columns)
            
            return pivot_results
            
        except Exception as e:
            logger.error(f"Error creating notification pivot tables: {str(e)}")
            return None
    
    def get_pivot_table_summary(self, pivot_results):
        """Generate a text summary of pivot table results"""
        if not pivot_results:
            return "No pivot table data available."
        
        summary_lines = []
        summary_lines.append("📊 **NOTIFICATION DATA ANALYSIS SUMMARY**")
        summary_lines.append("=" * 50)
        
        # Summary statistics
        if 'summary_stats' in pivot_results:
            stats = pivot_results['summary_stats']
            summary_lines.append("\n📈 **DATA OVERVIEW:**")
            for col, stat in stats.items():
                if 'unique_count' in stat:
                    summary_lines.append(f"• {col}: {stat['unique_count']} unique values")
                if 'null_count' in stat and stat['null_count'] > 0:
                    summary_lines.append(f"  - {stat['null_count']} null values")
        
        # Top notifications by type
        if 'notifications_by_type' in pivot_results:
            pivot = pivot_results['notifications_by_type']
            if not pivot.empty:
                summary_lines.append("\n🔔 **TOP NOTIFICATION TYPES:**")
                top_types = pivot.head(5)
                for idx, count in top_types.items():
                    summary_lines.append(f"• {idx}: {count} notifications")
        
        # Top work centers
        if 'notifications_by_workcenter' in pivot_results:
            pivot = pivot_results['notifications_by_workcenter']
            if not pivot.empty:
                summary_lines.append("\n🏭 **TOP WORK CENTERS:**")
                top_workcenters = pivot.head(5)
                for idx, count in top_workcenters.items():
                    summary_lines.append(f"• {idx}: {count} notifications")
        
        # Top FPSOs
        if 'notifications_by_fpso' in pivot_results:
            pivot = pivot_results['notifications_by_fpso']
            if not pivot.empty:
                summary_lines.append("\n⛽ **TOP FPSOs:**")
                top_fpsos = pivot.head(5)
                for idx, count in top_fpsos.items():
                    summary_lines.append(f"• {idx}: {count} notifications")
        
        return "\n".join(summary_lines)
    
    def process_documents(self, documents):
        """Process documents and create vector store"""
        try:
            if not documents:
                return None
            
            # Split documents into chunks
            texts = self.text_splitter.split_documents(documents)
            
            # Try to create FAISS vector store
            try:
                vectorstore = FAISS.from_documents(texts, self.embeddings)
                return vectorstore
            except Exception as faiss_error:
                logger.warning(f"FAISS vector store failed: {str(faiss_error)}. Using simple text storage.")
                # Create a simple text-based storage as fallback
                return SimpleTextStore(texts)
            
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            return None
    
    def similarity_search(self, vectorstore, query, k=5):
        """Perform similarity search on the vector store"""
        try:
            if vectorstore is None:
                return []
            
            results = vectorstore.similarity_search(query, k=k)
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            return [] 