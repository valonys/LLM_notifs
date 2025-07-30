import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import numpy as np

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from utils.config import Config

logger = logging.getLogger(__name__)

class QueryRouter:
    """Intelligent query router for determining optimal processing strategy"""
    
    def __init__(self, config: Config):
        self.config = config
        
        # Query classification patterns
        self.patterns = {
            'pivot_table': [
                r'pivot\s+table',
                r'show\s+me\s+(?:data|statistics|numbers)',
                r'(?:count|sum|average|total)\s+of',
                r'(?:by|group\s+by)\s+\w+',
                r'notifications?\s+(?:by|per|from)',
                r'(?:trend|pattern)s?\s+in',
                r'FPSO\s+\w+',
                r'work\s*center',
                r'created\s+on',
                r'notification\s+type',
                r'breakdown\s+(?:by|of)',
                r'distribution\s+of',
                r'percentage\s+of'
            ],
            'document': [
                r'safety\s+(?:violation|incident|issue)',
                r'compliance\s+(?:check|review|status)',
                r'equipment\s+(?:performance|status|condition)',
                r'inspection\s+report',
                r'maintenance\s+(?:log|record)',
                r'risk\s+(?:assessment|analysis)',
                r'procedure\s+(?:review|analysis)',
                r'(?:what|how|why)\s+(?:does|is|are)',
                r'explain\s+(?:the|this|how)',
                r'describe\s+(?:the|this)',
                r'analyze\s+(?:the|this)\s+(?:document|report)',
                r'summarize\s+(?:the|this)'
            ],
            'hybrid': [
                r'compare\s+(?:data|documents)',
                r'correlation\s+between',
                r'(?:trend|pattern)\s+(?:in|from)\s+(?:both|documents?\s+and\s+data)',
                r'comprehensive\s+(?:analysis|review)',
                r'overall\s+(?:status|summary)',
                r'(?:combine|merge)\s+(?:data|information)',
                r'cross-reference',
                r'validate\s+(?:against|with)'
            ]
        }
        
        # Initialize TF-IDF vectorizer if available
        self.vectorizer = None
        self.query_embeddings = {}
        if SKLEARN_AVAILABLE:
            self._initialize_vectorizer()
        
        # Query intent keywords
        self.intent_keywords = {
            'safety': ['safety', 'violation', 'incident', 'hazard', 'risk', 'accident', 'injury'],
            'compliance': ['compliance', 'regulation', 'standard', 'audit', 'requirement', 'procedure'],
            'maintenance': ['maintenance', 'equipment', 'repair', 'service', 'inspection', 'condition'],
            'analytics': ['trend', 'pattern', 'statistics', 'analysis', 'insight', 'correlation'],
            'reporting': ['report', 'summary', 'status', 'overview', 'dashboard', 'metrics']
        }
        
        # Processing statistics
        self.routing_stats = {
            'total_queries': 0,
            'pivot_queries': 0,
            'document_queries': 0,
            'hybrid_queries': 0,
            'classification_accuracy': 0
        }
    
    def _initialize_vectorizer(self):
        """Initialize TF-IDF vectorizer for semantic similarity"""
        try:
            # Sample queries for each category to train the vectorizer
            sample_queries = {
                'pivot_table': [
                    "show me notifications by FPSO",
                    "count of safety violations by work center",
                    "trend in equipment failures",
                    "breakdown of notifications by type"
                ],
                'document': [
                    "analyze safety violations in inspection reports",
                    "review compliance status from documents",
                    "summarize equipment maintenance procedures",
                    "explain safety protocols"
                ],
                'hybrid': [
                    "compare document findings with notification data",
                    "comprehensive analysis of safety performance",
                    "validate maintenance reports against data trends",
                    "overall operational status review"
                ]
            }
            
            all_samples = []
            self.sample_labels = []
            
            for category, queries in sample_queries.items():
                all_samples.extend(queries)
                self.sample_labels.extend([category] * len(queries))
            
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            self.sample_vectors = self.vectorizer.fit_transform(all_samples)
            logger.info("Query vectorizer initialized successfully")
            
        except Exception as e:
            logger.error(f"Vectorizer initialization failed: {str(e)}")
    
    def classify_query(self, query: str) -> str:
        """Classify query into appropriate processing category"""
        try:
            self.routing_stats['total_queries'] += 1
            
            # Normalize query
            query_lower = query.lower().strip()
            
            # Rule-based classification first
            rule_based_result = self._rule_based_classification(query_lower)
            
            # If TF-IDF is available, use semantic similarity
            if SKLEARN_AVAILABLE and self.vectorizer:
                semantic_result = self._semantic_classification(query_lower)
                
                # Combine results (give more weight to rule-based)
                if rule_based_result == semantic_result:
                    result = rule_based_result
                elif rule_based_result != 'unknown':
                    result = rule_based_result
                else:
                    result = semantic_result
            else:
                result = rule_based_result
            
            # Update statistics
            if result == 'pivot_table':
                self.routing_stats['pivot_queries'] += 1
            elif result == 'document':
                self.routing_stats['document_queries'] += 1
            elif result == 'hybrid':
                self.routing_stats['hybrid_queries'] += 1
            
            logger.info(f"Query classified as: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Query classification failed: {str(e)}")
            return 'hybrid'  # Default to hybrid on error
    
    def _rule_based_classification(self, query: str) -> str:
        """Rule-based query classification using regex patterns"""
        scores = {}
        
        for category, patterns in self.patterns.items():
            score = 0
            for pattern in patterns:
                matches = re.findall(pattern, query, re.IGNORECASE)
                score += len(matches)
            scores[category] = score
        
        # Check for explicit keywords that strongly indicate category
        if any(keyword in query for keyword in ['pivot', 'count', 'sum', 'total', 'breakdown', 'distribution']):
            scores['pivot_table'] += 5
        
        if any(keyword in query for keyword in ['document', 'report', 'procedure', 'explain', 'describe']):
            scores['document'] += 3
        
        if any(keyword in query for keyword in ['compare', 'comprehensive', 'overall', 'validate']):
            scores['hybrid'] += 4
        
        # Return category with highest score
        if max(scores.values()) > 0:
            return max(scores, key=scores.get)
        else:
            return 'unknown'
    
    def _semantic_classification(self, query: str) -> str:
        """Semantic classification using TF-IDF similarity"""
        try:
            # Transform query to vector
            query_vector = self.vectorizer.transform([query])
            
            # Calculate similarities with sample queries
            similarities = cosine_similarity(query_vector, self.sample_vectors)[0]
            
            # Find best match
            best_match_idx = np.argmax(similarities)
            best_similarity = similarities[best_match_idx]
            
            # Only use if similarity is above threshold
            if best_similarity > 0.1:
                return self.sample_labels[best_match_idx]
            else:
                return 'unknown'
                
        except Exception as e:
            logger.error(f"Semantic classification failed: {str(e)}")
            return 'unknown'
    
    def extract_query_intent(self, query: str) -> Dict[str, Any]:
        """Extract detailed intent information from query"""
        try:
            intent_info = {
                'primary_intent': None,
                'secondary_intents': [],
                'keywords': [],
                'entities': {},
                'time_references': [],
                'confidence': 0.0
            }
            
            query_lower = query.lower()
            
            # Identify primary intent
            intent_scores = {}
            for intent, keywords in self.intent_keywords.items():
                score = sum(1 for keyword in keywords if keyword in query_lower)
                if score > 0:
                    intent_scores[intent] = score
            
            if intent_scores:
                intent_info['primary_intent'] = max(intent_scores, key=intent_scores.get)
                intent_info['confidence'] = intent_scores[intent_info['primary_intent']] / len(query_lower.split())
                
                # Secondary intents
                sorted_intents = sorted(intent_scores.items(), key=lambda x: x[1], reverse=True)
                intent_info['secondary_intents'] = [intent for intent, score in sorted_intents[1:] if score > 0]
            
            # Extract entities
            intent_info['entities'] = self._extract_entities(query)
            
            # Extract time references
            intent_info['time_references'] = self._extract_time_references(query)
            
            # Extract keywords
            intent_info['keywords'] = self._extract_keywords(query_lower)
            
            return intent_info
            
        except Exception as e:
            logger.error(f"Intent extraction failed: {str(e)}")
            return {'primary_intent': None, 'confidence': 0.0}
    
    def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract entities like FPSO names, work centers, etc."""
        entities = {
            'fpso': [],
            'work_center': [],
            'notification_type': [],
            'equipment': [],
            'locations': []
        }
        
        try:
            # FPSO patterns
            fpso_matches = re.findall(r'FPSO\s+(\w+)', query, re.IGNORECASE)
            entities['fpso'].extend(fpso_matches)
            
            # Work center patterns
            wc_matches = re.findall(r'work\s*center\s+(\w+)', query, re.IGNORECASE)
            entities['work_center'].extend(wc_matches)
            
            # Equipment patterns
            equipment_keywords = ['pump', 'valve', 'compressor', 'generator', 'turbine', 'vessel', 'tank']
            for keyword in equipment_keywords:
                if keyword in query.lower():
                    entities['equipment'].append(keyword)
            
            return entities
            
        except Exception as e:
            logger.error(f"Entity extraction failed: {str(e)}")
            return entities
    
    def _extract_time_references(self, query: str) -> List[str]:
        """Extract time-related references from query"""
        time_patterns = [
            r'last\s+(?:week|month|year|quarter)',
            r'(?:this|current)\s+(?:week|month|year|quarter)',
            r'(?:in|during)\s+\d{4}',
            r'(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}',
            r'Q[1-4]\s+\d{4}',
            r'yesterday|today|tomorrow',
            r'\d{1,2}\/\d{1,2}\/\d{4}',
            r'\d{4}-\d{2}-\d{2}'
        ]
        
        time_refs = []
        for pattern in time_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            time_refs.extend(matches)
        
        return time_refs
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from query"""
        # Common stop words to exclude
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can'}
        
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords
    
    def suggest_query_improvements(self, query: str, query_type: str) -> List[str]:
        """Suggest improvements to make queries more effective"""
        suggestions = []
        
        try:
            query_lower = query.lower()
            
            if query_type == 'pivot_table':
                if 'by' not in query_lower and 'group' not in query_lower:
                    suggestions.append("Consider adding 'by FPSO' or 'by work center' to group your data")
                
                if not any(word in query_lower for word in ['count', 'sum', 'average', 'total']):
                    suggestions.append("Specify the aggregation you want (count, sum, average, etc.)")
            
            elif query_type == 'document':
                if len(query.split()) < 3:
                    suggestions.append("Try to be more specific about what you're looking for in the documents")
                
                if not any(word in query_lower for word in ['analyze', 'explain', 'describe', 'summarize']):
                    suggestions.append("Consider using action words like 'analyze', 'explain', or 'summarize'")
            
            elif query_type == 'hybrid':
                if 'compare' not in query_lower and 'correlation' not in query_lower:
                    suggestions.append("Use words like 'compare' or 'correlation' to better utilize both data sources")
            
            # General suggestions
            if not any(char in query for char in '?!.'):
                suggestions.append("End your query with appropriate punctuation for clarity")
            
            if len(query.split()) > 20:
                suggestions.append("Consider breaking complex queries into smaller, more focused questions")
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Suggestion generation failed: {str(e)}")
            return []
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get query routing statistics"""
        stats = self.routing_stats.copy()
        
        # Calculate distribution percentages
        total = stats['total_queries']
        if total > 0:
            stats['pivot_percentage'] = (stats['pivot_queries'] / total) * 100
            stats['document_percentage'] = (stats['document_queries'] / total) * 100
            stats['hybrid_percentage'] = (stats['hybrid_queries'] / total) * 100
        
        return stats
    
    def reset_stats(self):
        """Reset routing statistics"""
        self.routing_stats = {
            'total_queries': 0,
            'pivot_queries': 0,
            'document_queries': 0,
            'hybrid_queries': 0,
            'classification_accuracy': 0
        }
