import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import hashlib
from dataclasses import dataclass
from enum import Enum

from langchain.schema import Document as LCDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

from utils.config import Config

logger = logging.getLogger(__name__)

class ChunkType(Enum):
    """Types of document chunks"""
    HEADER = "header"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    LIST = "list"
    CODE = "code"
    SEMANTIC = "semantic"
    HIERARCHICAL = "hierarchical"
    OVERLAP = "overlap"

@dataclass
class ChunkMetadata:
    """Enhanced metadata for document chunks"""
    chunk_id: str
    chunk_type: ChunkType
    parent_id: Optional[str]
    children_ids: List[str]
    level: int
    position: int
    size: int
    overlap_with: List[str]
    semantic_similarity: Optional[float]
    keywords: List[str]
    entities: List[str]

class ChunkingManager:
    """Advanced chunking manager with multiple strategies"""
    
    def __init__(self, config: Config):
        self.config = config
        
        # Chunking configurations
        self.chunking_configs = {
            'hierarchical': {
                'header_chunk_size': 2000,
                'paragraph_chunk_size': 1000,
                'overlap': 200,
                'min_chunk_size': 100
            },
            'semantic': {
                'similarity_threshold': 0.7,
                'max_chunk_size': 1500,
                'min_chunk_size': 200
            },
            'hybrid': {
                'base_chunk_size': 1000,
                'overlap': 150,
                'semantic_adjustment': True
            }
        }
        
        # Text splitters
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Initialize semantic model if available
        self.semantic_model = None
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Semantic chunking model loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load semantic model: {str(e)}")
        
        # Document structure patterns
        self.structure_patterns = {
            'header': re.compile(r'^#{1,6}\s+.+$', re.MULTILINE),
            'list_item': re.compile(r'^\s*[-*+]\s+.+$', re.MULTILINE),
            'numbered_list': re.compile(r'^\s*\d+\.\s+.+$', re.MULTILINE),
            'table_row': re.compile(r'\|.+\|', re.MULTILINE),
            'code_block': re.compile(r'```[\s\S]*?```', re.MULTILINE),
            'paragraph': re.compile(r'\n\s*\n'),
        }
        
        # Industrial document patterns
        self.industrial_patterns = {
            'safety_section': re.compile(r'(?i)safety|hazard|risk|warning|caution', re.MULTILINE),
            'equipment_section': re.compile(r'(?i)equipment|machinery|component|system', re.MULTILINE),
            'procedure_step': re.compile(r'(?i)step\s+\d+|procedure|process', re.MULTILINE),
            'notification_entry': re.compile(r'(?i)notification|alert|incident|event', re.MULTILINE)
        }
    
    def apply_hierarchical_chunking(self, documents: List[LCDocument]) -> List[LCDocument]:
        """Apply hierarchical chunking strategy to documents"""
        try:
            logger.info(f"Applying hierarchical chunking to {len(documents)} documents")
            
            chunked_documents = []
            
            for doc_idx, document in enumerate(documents):
                doc_chunks = self._hierarchical_chunk_document(document, doc_idx)
                chunked_documents.extend(doc_chunks)
            
            logger.info(f"Hierarchical chunking produced {len(chunked_documents)} chunks")
            return chunked_documents
            
        except Exception as e:
            logger.error(f"Hierarchical chunking failed: {str(e)}")
            # Fallback to simple chunking
            return self._apply_simple_chunking(documents)
    
    def _hierarchical_chunk_document(self, document: LCDocument, doc_idx: int) -> List[LCDocument]:
        """Apply hierarchical chunking to a single document"""
        chunks = []
        content = document.page_content
        
        try:
            # Detect document structure
            structure = self._analyze_document_structure(content)
            
            # Create hierarchical chunks based on structure
            if structure['has_headers']:
                chunks.extend(self._chunk_by_headers(document, doc_idx, structure))
            elif structure['has_lists']:
                chunks.extend(self._chunk_by_lists(document, doc_idx, structure))
            elif structure['has_tables']:
                chunks.extend(self._chunk_by_tables(document, doc_idx, structure))
            else:
                # Default paragraph-based chunking
                chunks.extend(self._chunk_by_paragraphs(document, doc_idx))
            
            # Add overlapping chunks for better context
            overlapping_chunks = self._create_overlapping_chunks(chunks)
            chunks.extend(overlapping_chunks)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Hierarchical chunking failed for document {doc_idx}: {str(e)}")
            return self._fallback_chunking(document, doc_idx)
    
    def _analyze_document_structure(self, content: str) -> Dict[str, Any]:
        """Analyze document structure to determine optimal chunking strategy"""
        structure = {
            'has_headers': False,
            'has_lists': False,
            'has_tables': False,
            'has_code': False,
            'paragraph_count': 0,
            'average_paragraph_length': 0,
            'industrial_content': False
        }
        
        try:
            # Check for headers
            header_matches = self.structure_patterns['header'].findall(content)
            structure['has_headers'] = len(header_matches) > 0
            
            # Check for lists
            list_matches = self.structure_patterns['list_item'].findall(content)
            numbered_matches = self.structure_patterns['numbered_list'].findall(content)
            structure['has_lists'] = len(list_matches) > 0 or len(numbered_matches) > 0
            
            # Check for tables
            table_matches = self.structure_patterns['table_row'].findall(content)
            structure['has_tables'] = len(table_matches) > 2  # Need multiple rows
            
            # Check for code blocks
            code_matches = self.structure_patterns['code_block'].findall(content)
            structure['has_code'] = len(code_matches) > 0
            
            # Analyze paragraphs
            paragraphs = self.structure_patterns['paragraph'].split(content)
            paragraphs = [p.strip() for p in paragraphs if p.strip()]
            structure['paragraph_count'] = len(paragraphs)
            if paragraphs:
                structure['average_paragraph_length'] = sum(len(p) for p in paragraphs) / len(paragraphs)
            
            # Check for industrial content
            for pattern_name, pattern in self.industrial_patterns.items():
                if pattern.search(content):
                    structure['industrial_content'] = True
                    break
            
            return structure
            
        except Exception as e:
            logger.error(f"Document structure analysis failed: {str(e)}")
            return structure
    
    def _chunk_by_headers(self, document: LCDocument, doc_idx: int, structure: Dict[str, Any]) -> List[LCDocument]:
        """Chunk document based on header structure"""
        chunks = []
        content = document.page_content
        
        try:
            # Split by headers
            header_pattern = self.structure_patterns['header']
            sections = header_pattern.split(content)
            headers = header_pattern.findall(content)
            
            for i, section in enumerate(sections):
                if not section.strip():
                    continue
                
                # Determine header level and create metadata
                header_text = headers[i-1] if i > 0 else "Introduction"
                level = self._get_header_level(header_text)
                
                # Create chunk with hierarchical metadata
                chunk_id = f"doc_{doc_idx}_header_{i}"
                
                metadata = document.metadata.copy()
                metadata.update({
                    'chunk_id': chunk_id,
                    'chunk_type': ChunkType.HEADER.value,
                    'level': level,
                    'position': i,
                    'header_text': header_text,
                    'section_size': len(section),
                    'parent_document': doc_idx
                })
                
                # Further split large sections
                if len(section) > self.chunking_configs['hierarchical']['header_chunk_size']:
                    sub_chunks = self._split_large_section(section, chunk_id, metadata)
                    chunks.extend(sub_chunks)
                else:
                    chunk = LCDocument(
                        page_content=section.strip(),
                        metadata=metadata
                    )
                    chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Header-based chunking failed: {str(e)}")
            return []
    
    def _chunk_by_lists(self, document: LCDocument, doc_idx: int, structure: Dict[str, Any]) -> List[LCDocument]:
        """Chunk document based on list structure"""
        chunks = []
        content = document.page_content
        
        try:
            # Identify list sections
            list_pattern = self.structure_patterns['list_item']
            numbered_pattern = self.structure_patterns['numbered_list']
            
            # Split content into sections containing lists
            sections = re.split(r'\n\s*\n', content)
            
            for i, section in enumerate(sections):
                if not section.strip():
                    continue
                
                chunk_id = f"doc_{doc_idx}_list_{i}"
                
                # Determine if section contains lists
                has_bullet_list = bool(list_pattern.search(section))
                has_numbered_list = bool(numbered_pattern.search(section))
                
                metadata = document.metadata.copy()
                metadata.update({
                    'chunk_id': chunk_id,
                    'chunk_type': ChunkType.LIST.value,
                    'position': i,
                    'has_bullet_list': has_bullet_list,
                    'has_numbered_list': has_numbered_list,
                    'section_size': len(section),
                    'parent_document': doc_idx
                })
                
                chunk = LCDocument(
                    page_content=section.strip(),
                    metadata=metadata
                )
                chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            logger.error(f"List-based chunking failed: {str(e)}")
            return []
    
    def _chunk_by_tables(self, document: LCDocument, doc_idx: int, structure: Dict[str, Any]) -> List[LCDocument]:
        """Chunk document based on table structure"""
        chunks = []
        content = document.page_content
        
        try:
            # Split content around tables
            table_pattern = self.structure_patterns['table_row']
            
            # Find table sections
            table_matches = list(table_pattern.finditer(content))
            
            if not table_matches:
                return []
            
            current_pos = 0
            
            for i, match in enumerate(table_matches):
                # Add pre-table content
                if match.start() > current_pos:
                    pre_content = content[current_pos:match.start()].strip()
                    if pre_content:
                        chunk_id = f"doc_{doc_idx}_pre_table_{i}"
                        
                        metadata = document.metadata.copy()
                        metadata.update({
                            'chunk_id': chunk_id,
                            'chunk_type': ChunkType.PARAGRAPH.value,
                            'position': i * 2,
                            'parent_document': doc_idx
                        })
                        
                        chunk = LCDocument(
                            page_content=pre_content,
                            metadata=metadata
                        )
                        chunks.append(chunk)
                
                # Find complete table
                table_start = match.start()
                table_end = self._find_table_end(content, table_start)
                table_content = content[table_start:table_end]
                
                chunk_id = f"doc_{doc_idx}_table_{i}"
                
                metadata = document.metadata.copy()
                metadata.update({
                    'chunk_id': chunk_id,
                    'chunk_type': ChunkType.TABLE.value,
                    'position': i * 2 + 1,
                    'table_rows': len(table_content.split('\n')),
                    'parent_document': doc_idx
                })
                
                chunk = LCDocument(
                    page_content=table_content.strip(),
                    metadata=metadata
                )
                chunks.append(chunk)
                
                current_pos = table_end
            
            # Add remaining content
            if current_pos < len(content):
                remaining_content = content[current_pos:].strip()
                if remaining_content:
                    chunk_id = f"doc_{doc_idx}_post_table"
                    
                    metadata = document.metadata.copy()
                    metadata.update({
                        'chunk_id': chunk_id,
                        'chunk_type': ChunkType.PARAGRAPH.value,
                        'position': len(table_matches) * 2,
                        'parent_document': doc_idx
                    })
                    
                    chunk = LCDocument(
                        page_content=remaining_content,
                        metadata=metadata
                    )
                    chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Table-based chunking failed: {str(e)}")
            return []
    
    def _chunk_by_paragraphs(self, document: LCDocument, doc_idx: int) -> List[LCDocument]:
        """Chunk document by paragraphs with intelligent sizing"""
        chunks = []
        content = document.page_content
        
        try:
            # Split into paragraphs
            paragraphs = self.structure_patterns['paragraph'].split(content)
            paragraphs = [p.strip() for p in paragraphs if p.strip()]
            
            # Group paragraphs into appropriately sized chunks
            current_chunk = []
            current_size = 0
            target_size = self.chunking_configs['hierarchical']['paragraph_chunk_size']
            
            for i, paragraph in enumerate(paragraphs):
                para_size = len(paragraph)
                
                # If adding this paragraph would exceed target size, create a chunk
                if current_size + para_size > target_size and current_chunk:
                    chunk_content = '\n\n'.join(current_chunk)
                    chunk_id = f"doc_{doc_idx}_para_{len(chunks)}"
                    
                    metadata = document.metadata.copy()
                    metadata.update({
                        'chunk_id': chunk_id,
                        'chunk_type': ChunkType.PARAGRAPH.value,
                        'position': len(chunks),
                        'paragraph_count': len(current_chunk),
                        'parent_document': doc_idx
                    })
                    
                    chunk = LCDocument(
                        page_content=chunk_content,
                        metadata=metadata
                    )
                    chunks.append(chunk)
                    
                    current_chunk = [paragraph]
                    current_size = para_size
                else:
                    current_chunk.append(paragraph)
                    current_size += para_size
            
            # Add remaining paragraphs as final chunk
            if current_chunk:
                chunk_content = '\n\n'.join(current_chunk)
                chunk_id = f"doc_{doc_idx}_para_{len(chunks)}"
                
                metadata = document.metadata.copy()
                metadata.update({
                    'chunk_id': chunk_id,
                    'chunk_type': ChunkType.PARAGRAPH.value,
                    'position': len(chunks),
                    'paragraph_count': len(current_chunk),
                    'parent_document': doc_idx
                })
                
                chunk = LCDocument(
                    page_content=chunk_content,
                    metadata=metadata
                )
                chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Paragraph-based chunking failed: {str(e)}")
            return []
    
    def _create_overlapping_chunks(self, chunks: List[LCDocument]) -> List[LCDocument]:
        """Create overlapping chunks for better context retrieval"""
        overlapping_chunks = []
        
        try:
            overlap_size = self.chunking_configs['hierarchical']['overlap']
            
            for i in range(len(chunks) - 1):
                current_chunk = chunks[i]
                next_chunk = chunks[i + 1]
                
                # Create overlap between consecutive chunks
                current_content = current_chunk.page_content
                next_content = next_chunk.page_content
                
                # Take last part of current and first part of next
                current_words = current_content.split()
                next_words = next_content.split()
                
                if len(current_words) + len(next_words) > overlap_size:
                    overlap_words = current_words[-overlap_size//2:] + next_words[:overlap_size//2]
                    overlap_content = ' '.join(overlap_words)
                    
                    chunk_id = f"{current_chunk.metadata.get('chunk_id', '')}_overlap_{i}"
                    
                    metadata = current_chunk.metadata.copy()
                    metadata.update({
                        'chunk_id': chunk_id,
                        'chunk_type': ChunkType.OVERLAP.value,
                        'overlap_with': [
                            current_chunk.metadata.get('chunk_id', ''),
                            next_chunk.metadata.get('chunk_id', '')
                        ],
                        'is_overlap': True
                    })
                    
                    overlap_chunk = LCDocument(
                        page_content=overlap_content,
                        metadata=metadata
                    )
                    overlapping_chunks.append(overlap_chunk)
            
            return overlapping_chunks
            
        except Exception as e:
            logger.error(f"Overlap chunk creation failed: {str(e)}")
            return []
    
    def apply_semantic_chunking(self, documents: List[LCDocument]) -> List[LCDocument]:
        """Apply semantic chunking using sentence similarity"""
        if not self.semantic_model:
            logger.warning("Semantic model not available, falling back to hierarchical chunking")
            return self.apply_hierarchical_chunking(documents)
        
        try:
            logger.info(f"Applying semantic chunking to {len(documents)} documents")
            
            chunked_documents = []
            
            for doc_idx, document in enumerate(documents):
                doc_chunks = self._semantic_chunk_document(document, doc_idx)
                chunked_documents.extend(doc_chunks)
            
            logger.info(f"Semantic chunking produced {len(chunked_documents)} chunks")
            return chunked_documents
            
        except Exception as e:
            logger.error(f"Semantic chunking failed: {str(e)}")
            return self.apply_hierarchical_chunking(documents)
    
    def _semantic_chunk_document(self, document: LCDocument, doc_idx: int) -> List[LCDocument]:
        """Apply semantic chunking to a single document"""
        chunks = []
        content = document.page_content
        
        try:
            # Split into sentences
            if NLTK_AVAILABLE:
                sentences = sent_tokenize(content)
            else:
                sentences = re.split(r'[.!?]+', content)
                sentences = [s.strip() for s in sentences if s.strip()]
            
            if len(sentences) < 2:
                # Too few sentences, return as single chunk
                return [document]
            
            # Get sentence embeddings
            embeddings = self.semantic_model.encode(sentences)
            
            # Group sentences based on semantic similarity
            chunks = self._group_sentences_by_similarity(
                sentences, embeddings, document, doc_idx
            )
            
            return chunks
            
        except Exception as e:
            logger.error(f"Semantic chunking failed for document {doc_idx}: {str(e)}")
            return [document]
    
    def _group_sentences_by_similarity(
        self, 
        sentences: List[str], 
        embeddings: List[List[float]], 
        document: LCDocument, 
        doc_idx: int
    ) -> List[LCDocument]:
        """Group sentences into chunks based on semantic similarity"""
        chunks = []
        
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np
            
            similarity_threshold = self.chunking_configs['semantic']['similarity_threshold']
            max_chunk_size = self.chunking_configs['semantic']['max_chunk_size']
            
            current_group = [0]  # Start with first sentence
            current_content = [sentences[0]]
            
            for i in range(1, len(sentences)):
                # Calculate similarity with current group
                current_embedding = embeddings[i]
                group_embeddings = [embeddings[j] for j in current_group]
                
                # Average similarity with current group
                similarities = cosine_similarity([current_embedding], group_embeddings)[0]
                avg_similarity = np.mean(similarities)
                
                # Current content size
                current_size = sum(len(s) for s in current_content)
                
                # Decide whether to add to current group or start new one
                if (avg_similarity >= similarity_threshold and 
                    current_size + len(sentences[i]) <= max_chunk_size):
                    current_group.append(i)
                    current_content.append(sentences[i])
                else:
                    # Create chunk from current group
                    if current_content:
                        chunk = self._create_semantic_chunk(
                            current_content, current_group, document, doc_idx, len(chunks)
                        )
                        chunks.append(chunk)
                    
                    # Start new group
                    current_group = [i]
                    current_content = [sentences[i]]
            
            # Add final group
            if current_content:
                chunk = self._create_semantic_chunk(
                    current_content, current_group, document, doc_idx, len(chunks)
                )
                chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Sentence grouping failed: {str(e)}")
            return [document]
    
    def _create_semantic_chunk(
        self, 
        sentences: List[str], 
        sentence_indices: List[int], 
        document: LCDocument, 
        doc_idx: int, 
        chunk_idx: int
    ) -> LCDocument:
        """Create a semantic chunk from grouped sentences"""
        content = ' '.join(sentences)
        chunk_id = f"doc_{doc_idx}_semantic_{chunk_idx}"
        
        metadata = document.metadata.copy()
        metadata.update({
            'chunk_id': chunk_id,
            'chunk_type': ChunkType.SEMANTIC.value,
            'position': chunk_idx,
            'sentence_count': len(sentences),
            'sentence_indices': sentence_indices,
            'parent_document': doc_idx,
            'semantic_grouping': True
        })
        
        return LCDocument(
            page_content=content,
            metadata=metadata
        )
    
    def _apply_simple_chunking(self, documents: List[LCDocument]) -> List[LCDocument]:
        """Simple fallback chunking using RecursiveCharacterTextSplitter"""
        try:
            logger.info("Applying simple fallback chunking")
            
            all_chunks = []
            
            for doc_idx, document in enumerate(documents):
                chunks = self.recursive_splitter.split_documents([document])
                
                # Add enhanced metadata
                for i, chunk in enumerate(chunks):
                    chunk.metadata.update({
                        'chunk_id': f"doc_{doc_idx}_simple_{i}",
                        'chunk_type': ChunkType.PARAGRAPH.value,
                        'position': i,
                        'parent_document': doc_idx,
                        'chunking_method': 'simple'
                    })
                
                all_chunks.extend(chunks)
            
            return all_chunks
            
        except Exception as e:
            logger.error(f"Simple chunking failed: {str(e)}")
            return documents
    
    def _fallback_chunking(self, document: LCDocument, doc_idx: int) -> List[LCDocument]:
        """Fallback chunking for individual document"""
        try:
            chunks = self.recursive_splitter.split_documents([document])
            
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    'chunk_id': f"doc_{doc_idx}_fallback_{i}",
                    'chunk_type': ChunkType.PARAGRAPH.value,
                    'position': i,
                    'parent_document': doc_idx,
                    'chunking_method': 'fallback'
                })
            
            return chunks
            
        except Exception as e:
            logger.error(f"Fallback chunking failed: {str(e)}")
            return [document]
    
    def _get_header_level(self, header_text: str) -> int:
        """Determine header level from header text"""
        if header_text.startswith('######'):
            return 6
        elif header_text.startswith('#####'):
            return 5
        elif header_text.startswith('####'):
            return 4
        elif header_text.startswith('###'):
            return 3
        elif header_text.startswith('##'):
            return 2
        elif header_text.startswith('#'):
            return 1
        else:
            return 0
    
    def _find_table_end(self, content: str, start_pos: int) -> int:
        """Find the end position of a table"""
        lines = content[start_pos:].split('\n')
        table_end_pos = start_pos
        
        for i, line in enumerate(lines):
            if '|' in line:
                table_end_pos += len(line) + 1  # +1 for newline
            else:
                break
        
        return table_end_pos
    
    def _split_large_section(self, section: str, parent_chunk_id: str, parent_metadata: Dict[str, Any]) -> List[LCDocument]:
        """Split large sections into smaller chunks"""
        sub_chunks = []
        
        try:
            # Use recursive splitter for large sections
            temp_doc = LCDocument(page_content=section, metadata=parent_metadata)
            splits = self.recursive_splitter.split_documents([temp_doc])
            
            for i, split in enumerate(splits):
                sub_chunk_id = f"{parent_chunk_id}_sub_{i}"
                
                split.metadata.update({
                    'chunk_id': sub_chunk_id,
                    'parent_chunk_id': parent_chunk_id,
                    'sub_chunk_index': i,
                    'is_sub_chunk': True
                })
                
                sub_chunks.append(split)
            
            return sub_chunks
            
        except Exception as e:
            logger.error(f"Large section splitting failed: {str(e)}")
            return []

