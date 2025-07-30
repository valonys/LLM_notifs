import os
import logging
from typing import Dict, List, Any, Optional, Generator
import json
import time
from datetime import datetime
import requests

# AI model imports
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from cerebras.cloud.sdk import Cerebras
    CEREBRAS_AVAILABLE = True
except ImportError:
    CEREBRAS_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from utils.config import Config
from utils.cache_manager import CacheManager
from prompts.industrial_prompts import INDUSTRIAL_PROMPTS

logger = logging.getLogger(__name__)

class ModelManager:
    """Enhanced model manager with multiple AI providers and intelligent routing"""
    
    def __init__(self, config: Config):
        self.config = config
        self.cache_manager = CacheManager()
        
        # Model configurations
        self.model_configs = {
            'EE Smartest Agent': {
                'provider': 'xai',
                'model_name': 'grok-4-latest',
                'endpoint': 'https://api.x.ai/v1/chat/completions',
                'max_tokens': 4000,
                'temperature': 0.7
            },
            'EdJa-Valonys': {
                'provider': 'cerebras',
                'model_name': 'llama3.1-70b',
                'max_tokens': 2000,
                'temperature': 0.6
            },
            'JI Divine Agent': {
                'provider': 'cerebras',
                'model_name': 'llama3.1-8b',
                'max_tokens': 2000,
                'temperature': 0.8
            },
            'OpenAI GPT-4': {
                'provider': 'openai',
                'model_name': 'gpt-4',
                'max_tokens': 3000,
                'temperature': 0.7
            },
            'Domain-Specific Industrial Model': {
                'provider': 'huggingface',
                'model_name': 'microsoft/DialoGPT-medium',
                'max_tokens': 1500,
                'temperature': 0.6
            }
        }
        
        # Initialize available models
        self.available_models = {}
        self._initialize_models()
        
        # Performance tracking
        self.model_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_response_time': 0,
            'model_usage': {}
        }
    
    def _initialize_models(self):
        """Initialize available AI models"""
        
        # Initialize OpenAI
        if OPENAI_AVAILABLE and os.getenv('OPENAI_API_KEY'):
            try:
                self.available_models['openai'] = openai.OpenAI(
                    api_key=os.getenv('OPENAI_API_KEY')
                )
                logger.info("OpenAI client initialized")
            except Exception as e:
                logger.warning(f"OpenAI initialization failed: {str(e)}")
        
        # Initialize Cerebras
        if CEREBRAS_AVAILABLE and os.getenv('CEREBRAS_API_KEY'):
            try:
                self.available_models['cerebras'] = Cerebras(
                    api_key=os.getenv('CEREBRAS_API_KEY')
                )
                logger.info("Cerebras client initialized")
            except Exception as e:
                logger.warning(f"Cerebras initialization failed: {str(e)}")
        
        # Initialize XAI (using requests for API calls)
        if os.getenv('API_KEY'):  # XAI API key
            self.available_models['xai'] = True
            logger.info("XAI client configured")
        
        # Initialize HuggingFace models
        if TRANSFORMERS_AVAILABLE:
            try:
                # Load a lightweight model for domain-specific tasks
                self._load_huggingface_model()
                logger.info("HuggingFace models initialized")
            except Exception as e:
                logger.warning(f"HuggingFace initialization failed: {str(e)}")
    
    def _load_huggingface_model(self):
        """Load HuggingFace model for industrial domain tasks"""
        try:
            # Use a smaller model that can run efficiently
            model_name = "microsoft/DialoGPT-small"  # Lightweight conversational model
            
            tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Use pipeline for easier inference
            generator = pipeline(
                'text-generation',
                model=model_name,
                tokenizer=tokenizer,
                device=-1,  # Use CPU
                do_sample=True,
                temperature=0.7,
                max_length=512
            )
            
            self.available_models['huggingface'] = {
                'generator': generator,
                'tokenizer': tokenizer
            }
            
        except Exception as e:
            logger.error(f"HuggingFace model loading failed: {str(e)}")
    
    def generate_response(
        self, 
        query: str, 
        context: str, 
        model_name: str, 
        prompt_type: str = "Daily Report Summarization"
    ) -> str:
        """Generate response using specified model"""
        start_time = time.time()
        
        try:
            self.model_stats['total_requests'] += 1
            
            # Get model configuration
            if model_name not in self.model_configs:
                raise ValueError(f"Unknown model: {model_name}")
            
            model_config = self.model_configs[model_name]
            provider = model_config['provider']
            
            # Check cache first
            cache_key = f"{model_name}_{prompt_type}_{hash(query + context)}"
            cached_response = self.cache_manager.get(cache_key)
            if cached_response:
                return cached_response
            
            # Build messages
            messages = self._build_messages(query, context, prompt_type)
            
            # Route to appropriate provider
            response = ""
            if provider == 'openai':
                response = self._generate_openai_response(messages, model_config)
            elif provider == 'cerebras':
                response = self._generate_cerebras_response(messages, model_config)
            elif provider == 'xai':
                response = self._generate_xai_response(messages, model_config)
            elif provider == 'huggingface':
                response = self._generate_huggingface_response(query, context, model_config)
            else:
                raise ValueError(f"Unsupported provider: {provider}")
            
            # Cache the response
            self.cache_manager.set(cache_key, response, ttl=3600)
            
            # Update statistics
            response_time = time.time() - start_time
            self._update_stats(model_name, response_time, success=True)
            
            return response
            
        except Exception as e:
            logger.error(f"Response generation failed for {model_name}: {str(e)}")
            self._update_stats(model_name, time.time() - start_time, success=False)
            return f"Error generating response: {str(e)}"
    
    def _build_messages(self, query: str, context: str, prompt_type: str) -> List[Dict[str, str]]:
        """Build message array for API calls"""
        system_prompt = INDUSTRIAL_PROMPTS.get(prompt_type, INDUSTRIAL_PROMPTS["Daily Report Summarization"])
        
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        if context:
            messages.append({
                "role": "system", 
                "content": f"Relevant Context:\n{context}"
            })
        
        messages.append({
            "role": "user", 
            "content": query
        })
        
        return messages
    
    def _generate_openai_response(self, messages: List[Dict[str, str]], config: Dict[str, Any]) -> str:
        """Generate response using OpenAI"""
        if 'openai' not in self.available_models:
            raise RuntimeError("OpenAI client not available")
        
        try:
            client = self.available_models['openai']
            
            response = client.chat.completions.create(
                model=config['model_name'],
                messages=messages,
                max_tokens=config['max_tokens'],
                temperature=config['temperature']
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise
    
    def _generate_cerebras_response(self, messages: List[Dict[str, str]], config: Dict[str, Any]) -> str:
        """Generate response using Cerebras"""
        if 'cerebras' not in self.available_models:
            raise RuntimeError("Cerebras client not available")
        
        try:
            client = self.available_models['cerebras']
            
            response = client.chat.completions.create(
                model=config['model_name'],
                messages=messages,
                max_tokens=config['max_tokens'],
                temperature=config['temperature']
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Cerebras API error: {str(e)}")
            raise
    
    def _generate_xai_response(self, messages: List[Dict[str, str]], config: Dict[str, Any]) -> str:
        """Generate response using XAI API"""
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.getenv('API_KEY')}"
            }
            
            data = {
                "messages": messages,
                "model": config['model_name'],
                "max_tokens": config['max_tokens'],
                "temperature": config['temperature'],
                "stream": False
            }
            
            response = requests.post(
                config['endpoint'],
                headers=headers,
                json=data,
                timeout=60
            )
            
            response.raise_for_status()
            result = response.json()
            
            return result['choices'][0]['message']['content']
            
        except Exception as e:
            logger.error(f"XAI API error: {str(e)}")
            raise
    
    def _generate_huggingface_response(self, query: str, context: str, config: Dict[str, Any]) -> str:
        """Generate response using HuggingFace model"""
        if 'huggingface' not in self.available_models:
            raise RuntimeError("HuggingFace model not available")
        
        try:
            generator = self.available_models['huggingface']['generator']
            
            # Combine context and query
            prompt = f"Context: {context}\n\nQuestion: {query}\n\nResponse:"
            
            # Generate response
            outputs = generator(
                prompt,
                max_length=len(prompt.split()) + config['max_tokens'],
                num_return_sequences=1,
                temperature=config['temperature'],
                do_sample=True,
                pad_token_id=generator.tokenizer.eos_token_id
            )
            
            # Extract generated text
            generated_text = outputs[0]['generated_text']
            response = generated_text[len(prompt):].strip()
            
            return response if response else "I need more specific information to provide a detailed response."
            
        except Exception as e:
            logger.error(f"HuggingFace generation error: {str(e)}")
            raise
    
    def generate_document_response(
        self, 
        query: str, 
        relevant_docs: List[Any], 
        model_name: str, 
        prompt_type: str
    ) -> str:
        """Generate response for document-based queries"""
        try:
            # Build context from relevant documents
            context_parts = []
            for i, doc in enumerate(relevant_docs):
                source = doc.metadata.get('source', f'Document {i+1}') if hasattr(doc, 'metadata') else f'Document {i+1}'
                content = doc.content if hasattr(doc, 'content') else str(doc)
                context_parts.append(f"Source: {source}\n{content}")
            
            context = "\n\n".join(context_parts)
            
            return self.generate_response(query, context, model_name, prompt_type)
            
        except Exception as e:
            logger.error(f"Document response generation failed: {str(e)}")
            return f"Error generating document response: {str(e)}"
    
    def generate_pivot_response(
        self, 
        query: str, 
        pivot_results: Dict[str, Any], 
        model_name: str
    ) -> str:
        """Generate response for pivot table queries"""
        try:
            # Format pivot results as context
            context = "Pivot Table Analysis Results:\n"
            
            for key, value in pivot_results.items():
                if isinstance(value, (dict, list)):
                    context += f"{key}: {json.dumps(value, indent=2)}\n"
                else:
                    context += f"{key}: {value}\n"
            
            return self.generate_response(query, context, model_name, "Pivot Table Analysis")
            
        except Exception as e:
            logger.error(f"Pivot response generation failed: {str(e)}")
            return f"Error generating pivot response: {str(e)}"
    
    def combine_responses(
        self, 
        query: str, 
        doc_response: str, 
        pivot_response: str
    ) -> str:
        """Combine document and pivot responses"""
        try:
            combined_context = f"Document Analysis:\n{doc_response}\n\nPivot Analysis:\n{pivot_response}"
            
            synthesis_prompt = f"""Based on both document analysis and data analysis results, provide a comprehensive response to: {query}

Document Analysis Results:
{doc_response}

Data Analysis Results:
{pivot_response}

Please synthesize these insights into a coherent, actionable response."""
            
            return self.generate_response(
                synthesis_prompt, 
                "", 
                "EE Smartest Agent",  # Use the most capable model for synthesis
                "Daily Report Summarization"
            )
            
        except Exception as e:
            logger.error(f"Response combination failed: {str(e)}")
            return f"Document Analysis:\n{doc_response}\n\nData Analysis:\n{pivot_response}"
    
    def _update_stats(self, model_name: str, response_time: float, success: bool):
        """Update model performance statistics"""
        try:
            if success:
                self.model_stats['successful_requests'] += 1
            else:
                self.model_stats['failed_requests'] += 1
            
            # Update model usage
            if model_name not in self.model_stats['model_usage']:
                self.model_stats['model_usage'][model_name] = {
                    'requests': 0,
                    'total_time': 0,
                    'avg_time': 0
                }
            
            stats = self.model_stats['model_usage'][model_name]
            stats['requests'] += 1
            stats['total_time'] += response_time
            stats['avg_time'] = stats['total_time'] / stats['requests']
            
            # Update overall average
            total_requests = self.model_stats['total_requests']
            current_avg = self.model_stats['average_response_time']
            new_avg = ((current_avg * (total_requests - 1)) + response_time) / total_requests
            self.model_stats['average_response_time'] = new_avg
            
        except Exception as e:
            logger.error(f"Stats update failed: {str(e)}")
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get model performance statistics"""
        return self.model_stats.copy()
    
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        return list(self.model_configs.keys())
