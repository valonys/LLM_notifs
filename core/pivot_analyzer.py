import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import json
import re
from dataclasses import dataclass

class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder for handling datetime objects"""
    def default(self, obj):
        if isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        elif isinstance(obj, pd.Period):
            return str(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        return super().default(obj)

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import sqlparse
    SQL_PARSE_AVAILABLE = True
except ImportError:
    SQL_PARSE_AVAILABLE = False

from utils.config import Config

logger = logging.getLogger(__name__)

@dataclass
class PivotResult:
    """Enhanced pivot analysis result"""
    data: Union[pd.DataFrame, Dict[str, Any]]
    summary: str
    insights: List[str]
    visualizations: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

class PivotAnalyzer:
    """Advanced pivot table analyzer with natural language processing"""
    
    def __init__(self, config: Config):
        self.config = config
        self.data_cache = {}
        self.pivot_cache = {}
        
        # Column mapping for common industrial terms
        self.column_mappings = {
            'notification': ['Notifictn', 'Notification', 'Notif', 'Notif Type'],
            'created_on': ['Created on', 'Creation Date', 'Date Created', 'Created'],
            'work_center': ['Main WorkCtr', 'Work Center', 'WorkCenter', 'WC'],
            'fpso': ['FPSO', 'Platform', 'Location'],
            'description': ['Description', 'Desc', 'Details'],
            'status': ['Status', 'State', 'Condition'],
            'priority': ['Priority', 'Urgency', 'Severity'],
            'equipment': ['Equipment', 'Asset', 'Component'],
            'maintenance_type': ['Maint Type', 'Maintenance Type', 'Work Type']
        }
        
        # SQL-like query patterns
        self.query_patterns = {
            'count': r'(?:count|number)\s+of\s+(\w+)',
            'sum': r'(?:sum|total)\s+(?:of\s+)?(\w+)',
            'average': r'(?:average|avg|mean)\s+(?:of\s+)?(\w+)',
            'group_by': r'(?:by|group\s+by)\s+(\w+)',
            'where': r'(?:where|for)\s+(\w+)\s*(?:=|is|equals?)\s*["\']?([^"\']+)["\']?',
            'time_range': r'(?:from|since|after)\s+([^\s]+)\s+(?:to|until|before)\s+([^\s]+)',
            'top': r'(?:top|first|highest)\s+(\d+)',
            'bottom': r'(?:bottom|last|lowest)\s+(\d+)'
        }
        
        # Analysis statistics
        self.analysis_stats = {
            'total_queries': 0,
            'successful_analyses': 0,
            'cache_hits': 0,
            'avg_processing_time': 0
        }
    
    def process_excel_files(self, excel_files: List[Any]) -> Dict[str, pd.DataFrame]:
        """Process Excel files and extract data for pivot analysis"""
        processed_data = {}
        
        for file in excel_files:
            try:
                file_name = getattr(file, 'name', 'unknown_file')
                logger.info(f"Processing Excel file: {file_name}")
                
                # Read all sheets
                excel_data = pd.read_excel(file, sheet_name=None, engine='openpyxl')
                
                for sheet_name, df in excel_data.items():
                    # Clean and prepare data
                    df_clean = self._clean_dataframe(df)
                    
                    if not df_clean.empty:
                        key = f"{file_name}_{sheet_name}"
                        processed_data[key] = df_clean
                        self.data_cache[key] = df_clean
                        
                        logger.info(f"Processed sheet '{sheet_name}' with {len(df_clean)} rows")
            
            except Exception as e:
                logger.error(f"Error processing Excel file: {str(e)}")
                continue
        
        return processed_data
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize dataframe"""
        try:
            # Remove completely empty rows and columns
            df = df.dropna(how='all').dropna(axis=1, how='all')
            
            # Clean column names
            df.columns = df.columns.astype(str).str.strip()
            
            # Standardize column names using mappings
            df = self._standardize_column_names(df)
            
            # Handle date columns
            df = self._process_date_columns(df)
            
            # Handle numeric columns
            df = self._process_numeric_columns(df)
            
            return df
            
        except Exception as e:
            logger.error(f"DataFrame cleaning failed: {str(e)}")
            return df
    
    def _standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names using predefined mappings"""
        new_columns = {}
        
        for standard_name, variations in self.column_mappings.items():
            for col in df.columns:
                if any(var.lower() in col.lower() for var in variations):
                    new_columns[col] = standard_name
                    break
        
        return df.rename(columns=new_columns)
    
    def _process_date_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and convert date columns"""
        date_columns = ['created_on', 'Creation Date', 'Date Created']
        
        for col in df.columns:
            if any(date_col.lower() in col.lower() for date_col in date_columns):
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                except Exception as e:
                    logger.warning(f"Could not convert {col} to datetime: {str(e)}")
        
        return df
    
    def _process_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process numeric columns"""
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try to convert to numeric if it looks like a number
                numeric_series = pd.to_numeric(df[col], errors='coerce')
                if not numeric_series.isna().all():
                    df[col] = numeric_series
        
        return df
    
    def natural_language_to_sql(self, query: str) -> Dict[str, Any]:
        """Convert natural language query to SQL-like operations"""
        try:
            query_lower = query.lower().strip()
            
            sql_ops = {
                'select': [],
                'from': None,
                'where': [],
                'group_by': [],
                'having': [],
                'order_by': [],
                'limit': None,
                'aggregations': []
            }
            
            # Extract aggregation functions
            for agg_type, pattern in self.query_patterns.items():
                if agg_type in ['count', 'sum', 'average']:
                    matches = re.findall(pattern, query_lower)
                    for match in matches:
                        sql_ops['aggregations'].append({
                            'function': agg_type,
                            'column': self._resolve_column_name(match)
                        })
            
            # Extract GROUP BY
            group_by_match = re.search(self.query_patterns['group_by'], query_lower)
            if group_by_match:
                column = self._resolve_column_name(group_by_match.group(1))
                sql_ops['group_by'].append(column)
            
            # Extract WHERE conditions
            where_matches = re.findall(self.query_patterns['where'], query_lower)
            for column, value in where_matches:
                sql_ops['where'].append({
                    'column': self._resolve_column_name(column),
                    'operator': '=',
                    'value': value.strip()
                })
            
            # Extract LIMIT
            top_match = re.search(self.query_patterns['top'], query_lower)
            if top_match:
                sql_ops['limit'] = int(top_match.group(1))
            
            return sql_ops
            
        except Exception as e:
            logger.error(f"Natural language to SQL conversion failed: {str(e)}")
            return {}
    
    def _resolve_column_name(self, column_hint: str) -> str:
        """Resolve column name from hint using mappings"""
        column_hint = column_hint.lower().strip()
        
        for standard_name, variations in self.column_mappings.items():
            if any(var.lower() in column_hint for var in variations):
                return standard_name
        
        return column_hint
    
    def execute_query(self, sql_ops: Dict[str, Any], data: Dict[str, pd.DataFrame]) -> PivotResult:
        """Execute SQL-like operations on data"""
        try:
            # For now, use the first available dataset
            if not data:
                return PivotResult(
                    data={},
                    summary="No data available for analysis",
                    insights=["Upload Excel files to perform pivot table analysis"]
                )
            
            df = list(data.values())[0]  # Use first dataset
            
            # Apply WHERE conditions
            for condition in sql_ops.get('where', []):
                column = condition['column']
                value = condition['value']
                
                # Find matching column in dataframe
                matching_cols = [col for col in df.columns if column.lower() in col.lower()]
                if matching_cols:
                    df = df[df[matching_cols[0]].astype(str).str.contains(value, case=False, na=False)]
            
            # Apply GROUP BY and aggregations
            if sql_ops.get('group_by') and sql_ops.get('aggregations'):
                group_col = sql_ops['group_by'][0]
                matching_group_cols = [col for col in df.columns if group_col.lower() in col.lower()]
                
                if matching_group_cols:
                    grouped = df.groupby(matching_group_cols[0])
                    
                    result_data = {}
                    for agg in sql_ops['aggregations']:
                        if agg['function'] == 'count':
                            result_data[f"count"] = grouped.size()
                        elif agg['function'] == 'sum':
                            numeric_cols = df.select_dtypes(include=[np.number]).columns
                            if len(numeric_cols) > 0:
                                result_data[f"sum"] = grouped[numeric_cols[0]].sum()
                        elif agg['function'] == 'average':
                            numeric_cols = df.select_dtypes(include=[np.number]).columns
                            if len(numeric_cols) > 0:
                                result_data[f"average"] = grouped[numeric_cols[0]].mean()
                    
                    result_df = pd.DataFrame(result_data)
                else:
                    result_df = df.head(10)  # Fallback
            else:
                # Simple aggregation without grouping
                if sql_ops.get('aggregations'):
                    agg = sql_ops['aggregations'][0]
                    result_data = {}
                    if agg['function'] == 'count':
                        result_data = {'total_count': len(df)}
                    elif agg['function'] == 'sum':
                        numeric_cols = df.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) > 0:
                            result_data = {'total_sum': df[numeric_cols[0]].sum()}
                    elif agg['function'] == 'average':
                        numeric_cols = df.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) > 0:
                            result_data = {'average': df[numeric_cols[0]].mean()}
                    result_df = pd.DataFrame([result_data])
                else:
                    result_df = df.head(10)
            
            # Apply LIMIT
            if sql_ops.get('limit'):
                result_df = result_df.head(sql_ops['limit'])
            
            # Generate summary and insights
            summary = self._generate_summary(result_df, sql_ops)
            insights = self._generate_insights(result_df, df)
            
            return PivotResult(
                data=result_df,
                summary=summary,
                insights=insights,
                metadata={'original_rows': len(df), 'result_rows': len(result_df)}
            )
            
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            return PivotResult(
                data={},
                summary=f"Query execution failed: {str(e)}",
                insights=["Please check your query syntax and try again"]
            )
    
    def _generate_summary(self, result_df: pd.DataFrame, sql_ops: Dict[str, Any]) -> str:
        """Generate summary of the pivot analysis results"""
        try:
            if result_df.empty:
                return "No data matches your query criteria."
            
            summary_parts = []
            
            # Basic statistics
            summary_parts.append(f"Analysis returned {len(result_df)} results.")
            
            # Describe aggregations
            if sql_ops.get('aggregations'):
                for agg in sql_ops['aggregations']:
                    if agg['function'] == 'count':
                        if 'count' in result_df.columns:
                            total = result_df['count'].sum()
                            summary_parts.append(f"Total count: {total}")
                    elif agg['function'] == 'sum':
                        if 'sum' in result_df.columns:
                            total = result_df['sum'].sum()
                            summary_parts.append(f"Total sum: {total:.2f}")
                    elif agg['function'] == 'average':
                        if 'average' in result_df.columns:
                            avg = result_df['average'].mean()
                            summary_parts.append(f"Overall average: {avg:.2f}")
            
            # Describe grouping
            if sql_ops.get('group_by'):
                group_col = sql_ops['group_by'][0]
                summary_parts.append(f"Data grouped by {group_col}")
            
            return " ".join(summary_parts)
            
        except Exception as e:
            logger.error(f"Summary generation failed: {str(e)}")
            return "Analysis completed successfully."
    
    def _generate_insights(self, result_df: pd.DataFrame, original_df: pd.DataFrame) -> List[str]:
        """Generate insights from the analysis results"""
        insights = []
        
        try:
            if result_df.empty:
                insights.append("No data found matching the specified criteria")
                return insights
            
            # Statistical insights
            if 'count' in result_df.columns:
                max_count = result_df['count'].max()
                min_count = result_df['count'].min()
                
                if max_count > min_count:
                    max_idx = result_df['count'].idxmax()
                    min_idx = result_df['count'].idxmin()
                    insights.append(f"Highest count: {max_count} (for {max_idx})")
                    insights.append(f"Lowest count: {min_count} (for {min_idx})")
            
            # Trend insights
            if len(result_df) > 1:
                numeric_cols = result_df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if result_df[col].std() > 0:
                        cv = result_df[col].std() / result_df[col].mean()
                        if cv > 0.5:
                            insights.append(f"High variability detected in {col} (CV: {cv:.2f})")
            
            # Data quality insights
            data_quality = self._assess_data_quality(original_df)
            insights.extend(data_quality)
            
            return insights[:5]  # Limit to top 5 insights
            
        except Exception as e:
            logger.error(f"Insight generation failed: {str(e)}")
            return ["Analysis completed with limited insights due to processing error"]
    
    def _assess_data_quality(self, df: pd.DataFrame) -> List[str]:
        """Assess data quality and provide insights"""
        quality_insights = []
        
        try:
            # Missing data assessment
            missing_pct = (df.isnull().sum() / len(df)) * 100
            high_missing = missing_pct[missing_pct > 20]
            
            if len(high_missing) > 0:
                quality_insights.append(f"Columns with high missing data: {list(high_missing.index)}")
            
            # Duplicate assessment
            duplicate_count = df.duplicated().sum()
            if duplicate_count > 0:
                quality_insights.append(f"Found {duplicate_count} duplicate records")
            
            # Data freshness (if date columns exist)
            date_cols = df.select_dtypes(include=['datetime64']).columns
            if len(date_cols) > 0:
                latest_date = df[date_cols[0]].max()
                if pd.notna(latest_date):
                    days_old = (datetime.now() - latest_date).days
                    if days_old > 30:
                        quality_insights.append(f"Data may be outdated (latest: {latest_date.date()})")
            
            return quality_insights
            
        except Exception as e:
            logger.error(f"Data quality assessment failed: {str(e)}")
            return []
    
    def analyze_for_query(self, query: str) -> Dict[str, Any]:
        """Analyze available data based on natural language query"""
        try:
            if not self.data_cache:
                return {
                    'error': 'No data available for analysis',
                    'suggestion': 'Please upload Excel files first'
                }
            
            # Convert query to SQL operations
            sql_ops = self.natural_language_to_sql(query)
            
            # Execute query
            result = self.execute_query(sql_ops, self.data_cache)
            
            # Convert DataFrame to JSON-serializable format
            if hasattr(result.data, 'to_dict'):
                # Convert DataFrame to dict with proper datetime handling
                data_dict = result.data.to_dict('records')
                # Clean the data to ensure JSON serializability
                cleaned_data = json.loads(json.dumps(data_dict, cls=DateTimeEncoder))
            else:
                cleaned_data = result.data
            
            return {
                'data': cleaned_data,
                'summary': result.summary,
                'insights': result.insights,
                'metadata': result.metadata
            }
            
        except Exception as e:
            logger.error(f"Query analysis failed: {str(e)}")
            return {
                'error': f'Analysis failed: {str(e)}',
                'suggestion': 'Please try rephrasing your query'
            }
    
    def analyze_trends(self) -> List[Dict[str, Any]]:
        """Analyze trends in the available data"""
        trends = []
        
        try:
            if not self.data_cache:
                return trends
            
            for data_key, df in self.data_cache.items():
                # Look for date columns
                date_cols = df.select_dtypes(include=['datetime64']).columns
                
                if len(date_cols) > 0:
                    date_col = date_cols[0]
                    
                    # Monthly trend analysis
                    df['month'] = df[date_col].dt.to_period('M')
                    monthly_counts = df.groupby('month').size()
                    
                    if len(monthly_counts) > 1:
                        # Calculate trend direction
                        recent_months = monthly_counts.tail(3)
                        if len(recent_months) >= 2:
                            trend_direction = "increasing" if recent_months.iloc[-1] > recent_months.iloc[0] else "decreasing"
                            trend_magnitude = abs(recent_months.iloc[-1] - recent_months.iloc[0]) / recent_months.iloc[0] * 100
                            
                            trends.append({
                                'metric': f'Monthly notifications ({data_key})',
                                'direction': trend_direction,
                                'magnitude': f"{trend_magnitude:.1f}%",
                                'insight': f"Notifications have been {trend_direction} over the last 3 months"
                            })
            
            return trends
            
        except Exception as e:
            logger.error(f"Trend analysis failed: {str(e)}")
            return []
    
    def generate_automated_insights(self) -> List[Dict[str, Any]]:
        """Generate automated insights from all available data"""
        insights = []
        
        try:
            if not self.data_cache:
                return insights
            
            for data_key, df in self.data_cache.items():
                # Data overview insight
                insights.append({
                    'category': 'Data Overview',
                    'description': f'{data_key}: {len(df)} records with {len(df.columns)} columns',
                    'recommendation': 'Review data quality and completeness'
                })
                
                # Top categories insight
                categorical_cols = df.select_dtypes(include=['object']).columns
                for col in categorical_cols[:3]:  # Top 3 categorical columns
                    top_values = df[col].value_counts().head(3)
                    insights.append({
                        'category': f'Top {col} Categories',
                        'description': f'Most common: {", ".join([f"{k} ({v})" for k, v in top_values.items()])}',
                        'recommendation': f'Focus attention on high-frequency {col} categories'
                    })
                
                # Time-based insights
                date_cols = df.select_dtypes(include=['datetime64']).columns
                if len(date_cols) > 0:
                    date_col = date_cols[0]
                    max_date = df[date_col].max()
                    min_date = df[date_col].min()
                    if pd.notna(max_date) and pd.notna(min_date):
                        date_range = max_date - min_date
                        insights.append({
                            'category': 'Temporal Coverage',
                            'description': f'Data spans {date_range.days} days',
                            'recommendation': 'Consider seasonal patterns in your analysis'
                        })
            
            return insights[:10]  # Limit to top 10 insights
            
        except Exception as e:
            logger.error(f"Automated insight generation failed: {str(e)}")
            return []
    
    def get_analysis_stats(self) -> Dict[str, Any]:
        """Get pivot analysis statistics"""
        return self.analysis_stats.copy()
