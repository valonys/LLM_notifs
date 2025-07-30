import logging
import traceback
import sys
from datetime import datetime
from typing import Any, Optional, Dict, List, Callable
from functools import wraps
from enum import Enum
import json

try:
    import sentry_sdk
    SENTRY_AVAILABLE = True
except ImportError:
    SENTRY_AVAILABLE = False

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """Error categories for better classification"""
    SYSTEM = "system"
    USER_INPUT = "user_input"
    API = "api"
    DATA_PROCESSING = "data_processing"
    MODEL = "model"
    CACHE = "cache"
    NETWORK = "network"
    AUTHENTICATION = "authentication"

class ErrorHandler:
    """Comprehensive error handling and monitoring system"""
    
    def __init__(self, enable_sentry: bool = True):
        self.enable_sentry = enable_sentry and SENTRY_AVAILABLE
        
        # Error tracking
        self.error_history = []
        self.error_counts = {}
        self.max_history_size = 1000
        
        # Error patterns for classification
        self.error_patterns = {
            ErrorCategory.API: [
                'api', 'http', 'request', 'response', 'timeout', 'connection',
                'unauthorized', 'forbidden', 'not found', 'rate limit'
            ],
            ErrorCategory.DATA_PROCESSING: [
                'pandas', 'dataframe', 'csv', 'excel', 'json', 'parsing',
                'column', 'row', 'index', 'missing', 'null'
            ],
            ErrorCategory.MODEL: [
                'model', 'embedding', 'transformer', 'tokenizer', 'inference',
                'prediction', 'gpu', 'cuda', 'memory'
            ],
            ErrorCategory.CACHE: [
                'cache', 'redis', 'memcached', 'storage', 'disk', 'serialize'
            ],
            ErrorCategory.NETWORK: [
                'network', 'socket', 'dns', 'ssl', 'certificate', 'proxy'
            ],
            ErrorCategory.USER_INPUT: [
                'invalid', 'missing', 'format', 'validation', 'input', 'parameter'
            ]
        }
        
        # Initialize Sentry if available
        if self.enable_sentry:
            self._init_sentry()
        
        logger.info("Error handler initialized")
    
    def _init_sentry(self):
        """Initialize Sentry error monitoring"""
        try:
            import os
            sentry_dsn = os.getenv('SENTRY_DSN')
            
            if sentry_dsn:
                sentry_sdk.init(
                    dsn=sentry_dsn,
                    traces_sample_rate=0.1,
                    profiles_sample_rate=0.1,
                )
                logger.info("Sentry error monitoring initialized")
            else:
                logger.warning("Sentry DSN not found, monitoring disabled")
                self.enable_sentry = False
                
        except Exception as e:
            logger.error(f"Sentry initialization failed: {str(e)}")
            self.enable_sentry = False
    
    def handle_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: Optional[ErrorCategory] = None,
        user_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """Handle and log an error with context"""
        
        error_info = {
            'timestamp': datetime.now().isoformat(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'severity': severity.value,
            'category': category.value if category else self._classify_error(error),
            'context': context or {},
            'traceback': traceback.format_exc(),
            'user_message': user_message or self._generate_user_message(error, severity)
        }
        
        # Log the error
        self._log_error(error_info)
        
        # Track error statistics
        self._track_error(error_info)
        
        # Send to Sentry if enabled
        if self.enable_sentry:
            self._send_to_sentry(error, error_info)
        
        # Store in history
        self._store_error_history(error_info)
        
        return error_info
    
    def _classify_error(self, error: Exception) -> str:
        """Automatically classify error based on error message and type"""
        try:
            error_text = f"{type(error).__name__} {str(error)}".lower()
            
            # Check each category for matching patterns
            for category, patterns in self.error_patterns.items():
                for pattern in patterns:
                    if pattern in error_text:
                        return category.value
            
            # Default classification based on exception type
            if isinstance(error, (ConnectionError, TimeoutError)):
                return ErrorCategory.NETWORK.value
            elif isinstance(error, (ValueError, TypeError)):
                return ErrorCategory.USER_INPUT.value
            elif isinstance(error, (ImportError, ModuleNotFoundError)):
                return ErrorCategory.SYSTEM.value
            else:
                return ErrorCategory.SYSTEM.value
                
        except Exception:
            return ErrorCategory.SYSTEM.value
    
    def _generate_user_message(self, error: Exception, severity: ErrorSeverity) -> str:
        """Generate user-friendly error message"""
        error_type = type(error).__name__
        
        # Common user-friendly messages
        user_messages = {
            'ConnectionError': 'Unable to connect to the service. Please check your internet connection and try again.',
            'TimeoutError': 'The operation timed out. Please try again in a few moments.',
            'FileNotFoundError': 'The requested file could not be found. Please check the file path and try again.',
            'PermissionError': 'Permission denied. Please check your access rights and try again.',
            'ValueError': 'Invalid input provided. Please check your data and try again.',
            'KeyError': 'Required information is missing. Please ensure all necessary fields are provided.',
            'ImportError': 'A required component is not available. Please contact support.',
            'ModuleNotFoundError': 'A required module is missing. Please contact support.',
        }
        
        base_message = user_messages.get(error_type, 'An unexpected error occurred.')
        
        # Add severity-specific advice
        if severity == ErrorSeverity.CRITICAL:
            base_message += ' This is a critical error that requires immediate attention.'
        elif severity == ErrorSeverity.HIGH:
            base_message += ' Please contact support if this problem persists.'
        elif severity == ErrorSeverity.LOW:
            base_message += ' This error should resolve automatically.'
        
        return base_message
    
    def _log_error(self, error_info: Dict[str, Any]):
        """Log error with appropriate level"""
        severity = error_info['severity']
        message = f"[{error_info['category'].upper()}] {error_info['error_type']}: {error_info['error_message']}"
        
        if severity == ErrorSeverity.CRITICAL.value:
            logger.critical(message, extra={'error_info': error_info})
        elif severity == ErrorSeverity.HIGH.value:
            logger.error(message, extra={'error_info': error_info})
        elif severity == ErrorSeverity.MEDIUM.value:
            logger.warning(message, extra={'error_info': error_info})
        else:
            logger.info(message, extra={'error_info': error_info})
    
    def _track_error(self, error_info: Dict[str, Any]):
        """Track error statistics"""
        error_key = f"{error_info['category']}:{error_info['error_type']}"
        
        if error_key not in self.error_counts:
            self.error_counts[error_key] = {
                'count': 0,
                'first_seen': error_info['timestamp'],
                'last_seen': error_info['timestamp'],
                'severity': error_info['severity']
            }
        
        self.error_counts[error_key]['count'] += 1
        self.error_counts[error_key]['last_seen'] = error_info['timestamp']
    
    def _send_to_sentry(self, error: Exception, error_info: Dict[str, Any]):
        """Send error to Sentry for monitoring"""
        try:
            with sentry_sdk.push_scope() as scope:
                # Add context
                scope.set_context("error_info", error_info)
                scope.set_tag("category", error_info['category'])
                scope.set_tag("severity", error_info['severity'])
                
                # Add context data
                for key, value in error_info.get('context', {}).items():
                    scope.set_extra(key, value)
                
                sentry_sdk.capture_exception(error)
                
        except Exception as e:
            logger.error(f"Failed to send error to Sentry: {str(e)}")
    
    def _store_error_history(self, error_info: Dict[str, Any]):
        """Store error in history with size limit"""
        self.error_history.append(error_info)
        
        # Maintain size limit
        if len(self.error_history) > self.max_history_size:
            self.error_history.pop(0)
    
    def with_error_handling(
        self,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: Optional[ErrorCategory] = None,
        user_message: Optional[str] = None,
        return_on_error: Any = None
    ):
        """Decorator for automatic error handling"""
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    context = {
                        'function': func.__name__,
                        'args': str(args)[:200],  # Limit size
                        'kwargs': str(kwargs)[:200]
                    }
                    
                    self.handle_error(
                        error=e,
                        context=context,
                        severity=severity,
                        category=category,
                        user_message=user_message
                    )
                    
                    return return_on_error
            return wrapper
        return decorator
    
    def safe_execute(
        self,
        func: Callable,
        *args,
        default_return: Any = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: Optional[ErrorCategory] = None,
        **kwargs
    ) -> Any:
        """Safely execute a function with error handling"""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            context = {
                'function': func.__name__ if hasattr(func, '__name__') else str(func),
                'args': str(args)[:200],
                'kwargs': str(kwargs)[:200]
            }
            
            self.handle_error(
                error=e,
                context=context,
                severity=severity,
                category=category
            )
            
            return default_return
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics"""
        total_errors = sum(info['count'] for info in self.error_counts.values())
        
        stats = {
            'total_errors': total_errors,
            'unique_error_types': len(self.error_counts),
            'error_history_size': len(self.error_history),
            'top_errors': [],
            'errors_by_category': {},
            'errors_by_severity': {}
        }
        
        # Top errors by count
        sorted_errors = sorted(
            self.error_counts.items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )
        stats['top_errors'] = [
            {
                'error': error_key,
                'count': info['count'],
                'first_seen': info['first_seen'],
                'last_seen': info['last_seen'],
                'severity': info['severity']
            }
            for error_key, info in sorted_errors[:10]
        ]
        
        # Errors by category
        for error_key, info in self.error_counts.items():
            category = error_key.split(':')[0]
            if category not in stats['errors_by_category']:
                stats['errors_by_category'][category] = 0
            stats['errors_by_category'][category] += info['count']
        
        # Errors by severity
        for info in self.error_counts.values():
            severity = info['severity']
            if severity not in stats['errors_by_severity']:
                stats['errors_by_severity'][severity] = 0
            stats['errors_by_severity'][severity] += info['count']
        
        return stats
    
    def get_recent_errors(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent errors from history"""
        return self.error_history[-limit:]
    
    def clear_error_history(self):
        """Clear error history and statistics"""
        self.error_history.clear()
        self.error_counts.clear()
        logger.info("Error history and statistics cleared")
    
    def export_error_report(self, filepath: str) -> bool:
        """Export comprehensive error report"""
        try:
            report = {
                'export_timestamp': datetime.now().isoformat(),
                'statistics': self.get_error_stats(),
                'recent_errors': self.get_recent_errors(100),
                'error_counts': self.error_counts,
                'system_info': {
                    'python_version': sys.version,
                    'platform': sys.platform
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Error report exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error report export failed: {str(e)}")
            return False
    
    def create_error_context(self, **kwargs) -> Dict[str, Any]:
        """Create error context with common information"""
        context = {
            'timestamp': datetime.now().isoformat(),
            'user_agent': kwargs.get('user_agent'),
            'session_id': kwargs.get('session_id'),
            'request_id': kwargs.get('request_id'),
            'user_id': kwargs.get('user_id'),
        }
        
        # Add custom context
        context.update(kwargs)
        
        # Remove None values
        return {k: v for k, v in context.items() if v is not None}
    
    def is_recoverable_error(self, error: Exception) -> bool:
        """Determine if an error is recoverable"""
        recoverable_errors = [
            ConnectionError,
            TimeoutError,
            PermissionError
        ]
        
        # Check if it's a known recoverable error type
        for error_type in recoverable_errors:
            if isinstance(error, error_type):
                return True
        
        # Check error message for recoverable patterns
        error_message = str(error).lower()
        recoverable_patterns = [
            'timeout',
            'connection',
            'temporary',
            'retry',
            'rate limit'
        ]
        
        return any(pattern in error_message for pattern in recoverable_patterns)
    
    def suggest_resolution(self, error: Exception) -> List[str]:
        """Suggest possible resolutions for an error"""
        suggestions = []
        error_type = type(error).__name__
        error_message = str(error).lower()
        
        # Type-specific suggestions
        if error_type == 'ConnectionError':
            suggestions.extend([
                'Check internet connection',
                'Verify API endpoint URL',
                'Check firewall settings',
                'Try again in a few moments'
            ])
        elif error_type == 'TimeoutError':
            suggestions.extend([
                'Increase timeout duration',
                'Check network stability',
                'Reduce request size',
                'Try again later'
            ])
        elif error_type == 'FileNotFoundError':
            suggestions.extend([
                'Verify file path is correct',
                'Check file permissions',
                'Ensure file exists',
                'Check file name spelling'
            ])
        elif error_type == 'ValueError':
            suggestions.extend([
                'Validate input data format',
                'Check data types',
                'Verify parameter values',
                'Review input requirements'
            ])
        
        # Pattern-specific suggestions
        if 'api key' in error_message:
            suggestions.extend([
                'Check API key is set correctly',
                'Verify API key is valid',
                'Ensure API key has proper permissions'
            ])
        elif 'memory' in error_message:
            suggestions.extend([
                'Reduce batch size',
                'Free up system memory',
                'Use smaller model',
                'Process data in chunks'
            ])
        elif 'model' in error_message:
            suggestions.extend([
                'Check model name is correct',
                'Verify model is available',
                'Try alternative model',
                'Check model requirements'
            ])
        
        return suggestions[:5]  # Limit to top 5 suggestions

