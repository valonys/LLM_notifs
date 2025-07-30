"""
Database Manager for DigiTwin RAG System
Comprehensive database management with SQLAlchemy ORM for industrial data persistence
"""

import os
import logging
import json
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from contextlib import contextmanager
from dataclasses import asdict
import pandas as pd

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Text, Boolean, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import text
import uuid

from utils.config import Config
from models.domain_models import SafetyIncident, Equipment, Notification, ComplianceItem

logger = logging.getLogger(__name__)

# Database Models
Base = declarative_base()

class Document(Base):
    """Document storage model"""
    __tablename__ = 'documents'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String(255), nullable=False)
    file_type = Column(String(50), nullable=False)
    content = Column(Text)
    file_metadata = Column(JSON)
    upload_date = Column(DateTime, default=datetime.utcnow)
    file_size = Column(Integer)
    processed = Column(Boolean, default=False)
    
    # Relationships
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")

class DocumentChunk(Base):
    """Document chunk storage for RAG"""
    __tablename__ = 'document_chunks'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey('documents.id'), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    chunk_metadata = Column(JSON)
    embedding_hash = Column(String(64))  # For embedding tracking
    
    # Relationships
    document = relationship("Document", back_populates="chunks")

class QueryHistory(Base):
    """Query history and analytics"""
    __tablename__ = 'query_history'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query_text = Column(Text, nullable=False)
    query_type = Column(String(50))  # 'document', 'pivot', 'safety', etc.
    model_used = Column(String(100))
    response_summary = Column(Text)
    processing_time = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
    user_feedback = Column(String(50))  # 'helpful', 'not_helpful', etc.
    session_id = Column(String(100))

class SafetyIncidentDB(Base):
    """Safety incidents database model"""
    __tablename__ = 'safety_incidents'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    incident_id = Column(String(100), unique=True, nullable=False)
    title = Column(String(255), nullable=False)
    description = Column(Text)
    severity = Column(String(50), nullable=False)
    date_occurred = Column(DateTime, nullable=False)
    location = Column(String(255))
    fpso = Column(String(100))
    work_center = Column(String(100))
    injured_persons = Column(Integer, default=0)
    environmental_impact = Column(Boolean, default=False)
    root_causes = Column(JSON)
    corrective_actions = Column(JSON)
    investigation_status = Column(String(100), default='pending')
    lessons_learned = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class EquipmentDB(Base):
    """Equipment database model"""
    __tablename__ = 'equipment'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    equipment_id = Column(String(100), unique=True, nullable=False)
    name = Column(String(255), nullable=False)
    type = Column(String(100))
    manufacturer = Column(String(255))
    model = Column(String(255))
    installation_date = Column(DateTime)
    location = Column(String(255))
    fpso = Column(String(100))
    work_center = Column(String(100))
    status = Column(String(50), default='operational')
    criticality = Column(String(50), default='medium')
    last_maintenance = Column(DateTime)
    next_maintenance = Column(DateTime)
    operating_hours = Column(Float, default=0.0)
    performance_indicators = Column(JSON)
    maintenance_history = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class NotificationDB(Base):
    """Notifications database model"""
    __tablename__ = 'notifications'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    notification_id = Column(String(100), unique=True, nullable=False)
    title = Column(String(255), nullable=False)
    description = Column(Text)
    type = Column(String(100), nullable=False)
    priority = Column(String(50), nullable=False)
    created_date = Column(DateTime, nullable=False)
    fpso = Column(String(100))
    work_center = Column(String(100))
    equipment_id = Column(String(100))
    creator = Column(String(255))
    assigned_to = Column(String(255))
    status = Column(String(100), default='open')
    estimated_hours = Column(Float)
    actual_hours = Column(Float)
    cost_estimate = Column(Float)
    completion_date = Column(DateTime)
    tags = Column(JSON)
    attachments = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class ComplianceItemDB(Base):
    """Compliance items database model"""
    __tablename__ = 'compliance_items'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    item_id = Column(String(100), unique=True, nullable=False)
    regulation = Column(String(255), nullable=False)
    requirement = Column(String(500), nullable=False)
    description = Column(Text)
    applicable_locations = Column(JSON)
    status = Column(String(50), nullable=False)
    last_assessment = Column(DateTime, nullable=False)
    next_assessment = Column(DateTime)
    responsible_party = Column(String(255))
    evidence_documents = Column(JSON)
    non_conformities = Column(JSON)
    corrective_actions = Column(JSON)
    risk_level = Column(String(50), default='medium')
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class CacheEntry(Base):
    """Cache entries for improved performance"""
    __tablename__ = 'cache_entries'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    cache_key = Column(String(255), unique=True, nullable=False)
    cache_value = Column(JSON)
    cache_type = Column(String(100))  # 'embedding', 'query_result', 'analysis', etc.
    ttl = Column(DateTime)  # Time to live
    created_at = Column(DateTime, default=datetime.utcnow)
    accessed_at = Column(DateTime, default=datetime.utcnow)
    access_count = Column(Integer, default=1)

class DatabaseManager:
    """Comprehensive database manager for DigiTwin system"""
    
    def __init__(self, config: Config):
        self.config = config
        self.database_url = os.getenv('DATABASE_URL')
        
        if not self.database_url:
            raise ValueError("DATABASE_URL environment variable not found")
        
        # Create engine with optimized settings
        self.engine = create_engine(
            self.database_url,
            pool_size=10,
            max_overflow=20,
            pool_timeout=30,
            pool_recycle=3600,
            echo=False  # Set to True for SQL debugging
        )
        
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Initialize database
        self._initialize_database()
        
        logger.info("Database manager initialized successfully")
    
    def _initialize_database(self):
        """Initialize database tables"""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database: {str(e)}")
            raise
    
    @contextmanager
    def get_session(self) -> Session:
        """Get database session with context manager"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {str(e)}")
            raise
        finally:
            session.close()
    
    # Document Management
    def store_document(self, filename: str, file_type: str, content: str, 
                      metadata: Dict[str, Any], file_size: int = 0) -> str:
        """Store document in database"""
        try:
            with self.get_session() as session:
                document = Document(
                    filename=filename,
                    file_type=file_type,
                    content=content,
                    file_metadata=metadata,
                    file_size=file_size
                )
                session.add(document)
                session.flush()
                
                logger.info(f"Document stored: {filename} ({document.id})")
                return str(document.id)
                
        except Exception as e:
            logger.error(f"Failed to store document: {str(e)}")
            raise
    
    def store_document_chunks(self, document_id: str, chunks: List[Dict[str, Any]]):
        """Store document chunks for RAG"""
        try:
            with self.get_session() as session:
                for i, chunk_data in enumerate(chunks):
                    chunk = DocumentChunk(
                        document_id=document_id,
                        chunk_index=i,
                        content=chunk_data.get('content', ''),
                        chunk_metadata=chunk_data.get('metadata', {}),
                        embedding_hash=chunk_data.get('embedding_hash')
                    )
                    session.add(chunk)
                
                logger.info(f"Stored {len(chunks)} chunks for document {document_id}")
                
        except Exception as e:
            logger.error(f"Failed to store document chunks: {str(e)}")
            raise
    
    def get_documents(self, limit: int = 100, processed_only: bool = False) -> List[Dict[str, Any]]:
        """Retrieve documents from database"""
        try:
            with self.get_session() as session:
                query = session.query(Document)
                
                if processed_only:
                    query = query.filter(Document.processed == True)
                
                documents = query.order_by(Document.upload_date.desc()).limit(limit).all()
                
                return [{
                    'id': str(doc.id),
                    'filename': doc.filename,
                    'file_type': doc.file_type,
                    'upload_date': doc.upload_date.isoformat(),
                    'file_size': doc.file_size,
                    'processed': doc.processed,
                    'metadata': doc.file_metadata
                } for doc in documents]
                
        except Exception as e:
            logger.error(f"Failed to retrieve documents: {str(e)}")
            return []
    
    def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """Retrieve document chunks"""
        try:
            with self.get_session() as session:
                chunks = session.query(DocumentChunk).filter(
                    DocumentChunk.document_id == document_id
                ).order_by(DocumentChunk.chunk_index).all()
                
                return [{
                    'id': str(chunk.id),
                    'chunk_index': chunk.chunk_index,
                    'content': chunk.content,
                    'metadata': chunk.chunk_metadata,
                    'embedding_hash': chunk.embedding_hash
                } for chunk in chunks]
                
        except Exception as e:
            logger.error(f"Failed to retrieve document chunks: {str(e)}")
            return []
    
    # Query History Management
    def log_query(self, query_text: str, query_type: str, model_used: str,
                 response_summary: str, processing_time: float, session_id: str = None) -> str:
        """Log query for analytics"""
        try:
            with self.get_session() as session:
                query_log = QueryHistory(
                    query_text=query_text,
                    query_type=query_type,
                    model_used=model_used,
                    response_summary=response_summary,
                    processing_time=processing_time,
                    session_id=session_id or str(uuid.uuid4())
                )
                session.add(query_log)
                session.flush()
                
                return str(query_log.id)
                
        except Exception as e:
            logger.error(f"Failed to log query: {str(e)}")
            return ""
    
    def update_query_feedback(self, query_id: str, feedback: str):
        """Update query feedback"""
        try:
            with self.get_session() as session:
                query_log = session.query(QueryHistory).filter(
                    QueryHistory.id == query_id
                ).first()
                
                if query_log:
                    query_log.user_feedback = feedback
                    logger.info(f"Updated feedback for query {query_id}")
                
        except Exception as e:
            logger.error(f"Failed to update query feedback: {str(e)}")
    
    def get_query_analytics(self, days: int = 30) -> Dict[str, Any]:
        """Get query analytics"""
        try:
            with self.get_session() as session:
                cutoff_date = datetime.utcnow() - timedelta(days=days)
                
                # Total queries
                total_queries = session.query(QueryHistory).filter(
                    QueryHistory.timestamp >= cutoff_date
                ).count()
                
                # Average processing time
                avg_time_result = session.query(
                    text("AVG(processing_time)")
                ).filter(QueryHistory.timestamp >= cutoff_date).scalar()
                
                avg_processing_time = float(avg_time_result) if avg_time_result else 0.0
                
                # Query types distribution
                query_types = session.execute(
                    text("""
                    SELECT query_type, COUNT(*) as count 
                    FROM query_history 
                    WHERE timestamp >= :cutoff_date 
                    GROUP BY query_type
                    """),
                    {'cutoff_date': cutoff_date}
                ).fetchall()
                
                return {
                    'total_queries': total_queries,
                    'avg_processing_time': avg_processing_time,
                    'query_types': dict(query_types),
                    'period_days': days
                }
                
        except Exception as e:
            logger.error(f"Failed to get query analytics: {str(e)}")
            return {}
    
    # Industrial Data Management
    def store_safety_incident(self, incident: SafetyIncident) -> str:
        """Store safety incident"""
        try:
            with self.get_session() as session:
                incident_db = SafetyIncidentDB(
                    incident_id=incident.incident_id,
                    title=incident.title,
                    description=incident.description,
                    severity=incident.severity.value,
                    date_occurred=incident.date_occurred,
                    location=incident.location,
                    fpso=incident.fpso,
                    work_center=incident.work_center,
                    injured_persons=incident.injured_persons,
                    environmental_impact=incident.environmental_impact,
                    root_causes=incident.root_causes,
                    corrective_actions=incident.corrective_actions,
                    investigation_status=incident.investigation_status,
                    lessons_learned=incident.lessons_learned
                )
                session.add(incident_db)
                session.flush()
                
                logger.info(f"Safety incident stored: {incident.incident_id}")
                return str(incident_db.id)
                
        except Exception as e:
            logger.error(f"Failed to store safety incident: {str(e)}")
            raise
    
    def get_safety_incidents(self, limit: int = 100, severity: str = None) -> List[Dict[str, Any]]:
        """Retrieve safety incidents"""
        try:
            with self.get_session() as session:
                query = session.query(SafetyIncidentDB)
                
                if severity:
                    query = query.filter(SafetyIncidentDB.severity == severity)
                
                incidents = query.order_by(SafetyIncidentDB.date_occurred.desc()).limit(limit).all()
                
                return [{
                    'id': str(incident.id),
                    'incident_id': incident.incident_id,
                    'title': incident.title,
                    'description': incident.description,
                    'severity': incident.severity,
                    'date_occurred': incident.date_occurred.isoformat(),
                    'location': incident.location,
                    'fpso': incident.fpso,
                    'work_center': incident.work_center,
                    'injured_persons': incident.injured_persons,
                    'environmental_impact': incident.environmental_impact,
                    'investigation_status': incident.investigation_status
                } for incident in incidents]
                
        except Exception as e:
            logger.error(f"Failed to retrieve safety incidents: {str(e)}")
            return []
    
    # Cache Management
    def get_cache(self, cache_key: str, cache_type: str = None) -> Optional[Any]:
        """Retrieve cached value"""
        try:
            with self.get_session() as session:
                query = session.query(CacheEntry).filter(CacheEntry.cache_key == cache_key)
                
                if cache_type:
                    query = query.filter(CacheEntry.cache_type == cache_type)
                
                cache_entry = query.first()
                
                if cache_entry:
                    # Check TTL
                    if cache_entry.ttl and datetime.utcnow() > cache_entry.ttl:
                        session.delete(cache_entry)
                        return None
                    
                    # Update access statistics
                    cache_entry.accessed_at = datetime.utcnow()
                    cache_entry.access_count += 1
                    
                    return cache_entry.cache_value
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to retrieve cache: {str(e)}")
            return None
    
    def set_cache(self, cache_key: str, cache_value: Any, cache_type: str = None, 
                 ttl_hours: int = 24):
        """Store cached value"""
        try:
            with self.get_session() as session:
                # Remove existing entry
                existing = session.query(CacheEntry).filter(
                    CacheEntry.cache_key == cache_key
                ).first()
                
                if existing:
                    session.delete(existing)
                
                # Create new entry
                ttl = datetime.utcnow() + timedelta(hours=ttl_hours) if ttl_hours else None
                
                cache_entry = CacheEntry(
                    cache_key=cache_key,
                    cache_value=cache_value,
                    cache_type=cache_type,
                    ttl=ttl
                )
                session.add(cache_entry)
                
        except Exception as e:
            logger.error(f"Failed to store cache: {str(e)}")
    
    def clear_expired_cache(self):
        """Clear expired cache entries"""
        try:
            with self.get_session() as session:
                expired_count = session.query(CacheEntry).filter(
                    CacheEntry.ttl < datetime.utcnow()
                ).delete()
                
                logger.info(f"Cleared {expired_count} expired cache entries")
                
        except Exception as e:
            logger.error(f"Failed to clear expired cache: {str(e)}")
    
    # Database Health and Statistics
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            with self.get_session() as session:
                stats = {}
                
                # Table counts
                tables = [
                    ('documents', Document),
                    ('document_chunks', DocumentChunk),
                    ('query_history', QueryHistory),
                    ('safety_incidents', SafetyIncidentDB),
                    ('equipment', EquipmentDB),
                    ('notifications', NotificationDB),
                    ('compliance_items', ComplianceItemDB),
                    ('cache_entries', CacheEntry)
                ]
                
                for table_name, model_class in tables:
                    count = session.query(model_class).count()
                    stats[f'{table_name}_count'] = count
                
                # Database size (PostgreSQL specific)
                try:
                    db_size_result = session.execute(
                        text("SELECT pg_size_pretty(pg_database_size(current_database()))")
                    ).scalar()
                    stats['database_size'] = db_size_result
                except:
                    stats['database_size'] = 'Unknown'
                
                return stats
                
        except Exception as e:
            logger.error(f"Failed to get database stats: {str(e)}")
            return {}
    
    def health_check(self) -> Dict[str, Any]:
        """Perform database health check"""
        try:
            with self.get_session() as session:
                # Test basic connectivity
                session.execute(text("SELECT 1"))
                
                # Check table accessibility
                tables_accessible = []
                for table_name, model_class in [
                    ('documents', Document),
                    ('query_history', QueryHistory),
                    ('cache_entries', CacheEntry)
                ]:
                    try:
                        session.query(model_class).limit(1).all()
                        tables_accessible.append(table_name)
                    except Exception as e:
                        logger.warning(f"Table {table_name} not accessible: {str(e)}")
                
                return {
                    'status': 'healthy',
                    'connection': 'ok',
                    'tables_accessible': tables_accessible,
                    'timestamp': datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Database health check failed: {str(e)}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }