"""
Domain-specific models and utilities for industrial inspection analysis
Comprehensive model definitions, data structures, and utility functions
"""

import re
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Enumerations for industrial classifications
class SafetySeverity(Enum):
    """Safety violation severity levels"""
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"
    OBSERVATION = "observation"

class EquipmentStatus(Enum):
    """Equipment operational status"""
    OPERATIONAL = "operational"
    MAINTENANCE = "maintenance"
    SHUTDOWN = "shutdown"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"

class NotificationType(Enum):
    """Industrial notification types"""
    SAFETY_INCIDENT = "safety_incident"
    EQUIPMENT_FAILURE = "equipment_failure"
    MAINTENANCE_REQUEST = "maintenance_request"
    COMPLIANCE_ISSUE = "compliance_issue"
    ENVIRONMENTAL_EVENT = "environmental_event"
    OPERATIONAL_ISSUE = "operational_issue"
    QUALITY_ISSUE = "quality_issue"

class Priority(Enum):
    """Priority levels for notifications and actions"""
    EMERGENCY = "emergency"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class ComplianceStatus(Enum):
    """Compliance status levels"""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIAL_COMPLIANT = "partial_compliant"
    UNDER_REVIEW = "under_review"

# Data models for industrial entities
@dataclass
class SafetyIncident:
    """Model for safety incidents"""
    incident_id: str
    title: str
    description: str
    severity: SafetySeverity
    date_occurred: datetime
    location: str
    fpso: Optional[str] = None
    work_center: Optional[str] = None
    injured_persons: int = 0
    environmental_impact: bool = False
    root_causes: List[str] = field(default_factory=list)
    corrective_actions: List[str] = field(default_factory=list)
    investigation_status: str = "pending"
    lessons_learned: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'incident_id': self.incident_id,
            'title': self.title,
            'description': self.description,
            'severity': self.severity.value,
            'date_occurred': self.date_occurred.isoformat(),
            'location': self.location,
            'fpso': self.fpso,
            'work_center': self.work_center,
            'injured_persons': self.injured_persons,
            'environmental_impact': self.environmental_impact,
            'root_causes': self.root_causes,
            'corrective_actions': self.corrective_actions,
            'investigation_status': self.investigation_status,
            'lessons_learned': self.lessons_learned
        }

@dataclass
class Equipment:
    """Model for industrial equipment"""
    equipment_id: str
    name: str
    type: str
    manufacturer: str
    model: str
    installation_date: datetime
    location: str
    fpso: Optional[str] = None
    work_center: Optional[str] = None
    status: EquipmentStatus = EquipmentStatus.OPERATIONAL
    criticality: str = "medium"
    last_maintenance: Optional[datetime] = None
    next_maintenance: Optional[datetime] = None
    operating_hours: float = 0.0
    performance_indicators: Dict[str, float] = field(default_factory=dict)
    maintenance_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def calculate_availability(self, period_days: int = 30) -> float:
        """Calculate equipment availability percentage"""
        try:
            # Simplified calculation - in real implementation would use actual downtime data
            if self.status == EquipmentStatus.OPERATIONAL:
                return 95.0 + (np.random.random() * 5)  # 95-100% for operational
            elif self.status == EquipmentStatus.DEGRADED:
                return 80.0 + (np.random.random() * 15)  # 80-95% for degraded
            elif self.status == EquipmentStatus.MAINTENANCE:
                return 70.0 + (np.random.random() * 20)  # 70-90% for maintenance
            else:
                return 0.0  # 0% for shutdown
        except Exception:
            return 0.0
    
    def days_since_maintenance(self) -> Optional[int]:
        """Calculate days since last maintenance"""
        if self.last_maintenance:
            return (datetime.now() - self.last_maintenance).days
        return None
    
    def days_to_next_maintenance(self) -> Optional[int]:
        """Calculate days until next maintenance"""
        if self.next_maintenance:
            return (self.next_maintenance - datetime.now()).days
        return None

@dataclass
class Notification:
    """Model for industrial notifications"""
    notification_id: str
    title: str
    description: str
    type: NotificationType
    priority: Priority
    created_date: datetime
    fpso: Optional[str] = None
    work_center: Optional[str] = None
    equipment_id: Optional[str] = None
    creator: Optional[str] = None
    assigned_to: Optional[str] = None
    status: str = "open"
    estimated_hours: Optional[float] = None
    actual_hours: Optional[float] = None
    cost_estimate: Optional[float] = None
    completion_date: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)
    attachments: List[str] = field(default_factory=list)
    
    def is_overdue(self) -> bool:
        """Check if notification is overdue based on priority"""
        if self.status in ["completed", "closed"]:
            return False
        
        priority_sla = {
            Priority.EMERGENCY: 4,    # 4 hours
            Priority.HIGH: 24,        # 24 hours
            Priority.MEDIUM: 72,      # 72 hours
            Priority.LOW: 168         # 1 week
        }
        
        sla_hours = priority_sla.get(self.priority, 72)
        deadline = self.created_date + timedelta(hours=sla_hours)
        
        return datetime.now() > deadline
    
    def age_in_hours(self) -> float:
        """Calculate notification age in hours"""
        return (datetime.now() - self.created_date).total_seconds() / 3600

@dataclass
class ComplianceItem:
    """Model for compliance requirements and status"""
    item_id: str
    regulation: str
    requirement: str
    description: str
    applicable_locations: List[str]
    status: ComplianceStatus
    last_assessment: datetime
    next_assessment: Optional[datetime] = None
    responsible_party: Optional[str] = None
    evidence_documents: List[str] = field(default_factory=list)
    non_conformities: List[str] = field(default_factory=list)
    corrective_actions: List[str] = field(default_factory=list)
    risk_level: str = "medium"
    
    def is_assessment_due(self) -> bool:
        """Check if compliance assessment is due"""
        if not self.next_assessment:
            return True
        return datetime.now() >= self.next_assessment
    
    def days_until_assessment(self) -> Optional[int]:
        """Calculate days until next assessment"""
        if self.next_assessment:
            return (self.next_assessment - datetime.now()).days
        return None

# Utility classes for data processing
class IndustrialDataProcessor:
    """Processor for industrial data with domain-specific logic"""
    
    def __init__(self):
        self.equipment_patterns = {
            'pump': r'(?i)pump|P-\d+',
            'compressor': r'(?i)compressor|comp|C-\d+',
            'valve': r'(?i)valve|V-\d+',
            'vessel': r'(?i)vessel|tank|T-\d+',
            'heat_exchanger': r'(?i)heat.?exchanger|HX-\d+',
            'turbine': r'(?i)turbine|GT-\d+',
            'generator': r'(?i)generator|GEN-\d+'
        }
        
        self.safety_keywords = [
            'safety', 'hazard', 'risk', 'danger', 'warning', 'violation',
            'incident', 'accident', 'injury', 'near miss', 'unsafe'
        ]
        
        self.priority_keywords = {
            Priority.EMERGENCY: ['emergency', 'critical', 'urgent', 'immediate'],
            Priority.HIGH: ['high', 'important', 'asap', 'priority'],
            Priority.MEDIUM: ['medium', 'normal', 'routine'],
            Priority.LOW: ['low', 'minor', 'cosmetic', 'future']
        }
    
    def classify_notification_type(self, text: str) -> NotificationType:
        """Classify notification type based on text content"""
        text_lower = text.lower()
        
        # Safety-related keywords
        if any(keyword in text_lower for keyword in self.safety_keywords):
            return NotificationType.SAFETY_INCIDENT
        
        # Equipment-related keywords
        equipment_keywords = ['failure', 'malfunction', 'breakdown', 'fault', 'defect']
        if any(keyword in text_lower for keyword in equipment_keywords):
            return NotificationType.EQUIPMENT_FAILURE
        
        # Maintenance keywords
        maintenance_keywords = ['maintenance', 'repair', 'service', 'inspection', 'calibration']
        if any(keyword in text_lower for keyword in maintenance_keywords):
            return NotificationType.MAINTENANCE_REQUEST
        
        # Compliance keywords
        compliance_keywords = ['compliance', 'regulation', 'standard', 'procedure', 'audit']
        if any(keyword in text_lower for keyword in compliance_keywords):
            return NotificationType.COMPLIANCE_ISSUE
        
        # Environmental keywords
        environmental_keywords = ['spill', 'leak', 'emission', 'discharge', 'environmental']
        if any(keyword in text_lower for keyword in environmental_keywords):
            return NotificationType.ENVIRONMENTAL_EVENT
        
        # Default to operational issue
        return NotificationType.OPERATIONAL_ISSUE
    
    def extract_equipment_references(self, text: str) -> List[str]:
        """Extract equipment references from text"""
        equipment_refs = []
        
        for equipment_type, pattern in self.equipment_patterns.items():
            matches = re.findall(pattern, text)
            equipment_refs.extend(matches)
        
        return equipment_refs
    
    def assess_priority(self, text: str, notification_type: NotificationType) -> Priority:
        """Assess priority based on text content and notification type"""
        text_lower = text.lower()
        
        # Check for explicit priority keywords
        for priority, keywords in self.priority_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return priority
        
        # Default priority based on notification type
        type_priority_map = {
            NotificationType.SAFETY_INCIDENT: Priority.HIGH,
            NotificationType.EQUIPMENT_FAILURE: Priority.MEDIUM,
            NotificationType.MAINTENANCE_REQUEST: Priority.MEDIUM,
            NotificationType.COMPLIANCE_ISSUE: Priority.HIGH,
            NotificationType.ENVIRONMENTAL_EVENT: Priority.HIGH,
            NotificationType.OPERATIONAL_ISSUE: Priority.MEDIUM,
            NotificationType.QUALITY_ISSUE: Priority.MEDIUM
        }
        
        return type_priority_map.get(notification_type, Priority.MEDIUM)
    
    def extract_fpso_reference(self, text: str) -> Optional[str]:
        """Extract FPSO reference from text"""
        fpso_patterns = [
            r'FPSO[\s-]?(\w+)',
            r'Platform[\s-]?(\w+)',
            r'Vessel[\s-]?(\w+)'
        ]
        
        for pattern in fpso_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None

class IndustrialAnalytics:
    """Analytics utilities for industrial data"""
    
    @staticmethod
    def calculate_mtbf(failure_times: List[datetime]) -> Optional[float]:
        """Calculate Mean Time Between Failures in hours"""
        if len(failure_times) < 2:
            return None
        
        intervals = []
        for i in range(1, len(failure_times)):
            interval = (failure_times[i] - failure_times[i-1]).total_seconds() / 3600
            intervals.append(interval)
        
        return sum(intervals) / len(intervals)
    
    @staticmethod
    def calculate_mttr(repair_data: List[Dict[str, Any]]) -> Optional[float]:
        """Calculate Mean Time To Repair in hours"""
        if not repair_data:
            return None
        
        repair_times = []
        for repair in repair_data:
            if 'start_time' in repair and 'end_time' in repair:
                start = repair['start_time']
                end = repair['end_time']
                if isinstance(start, str):
                    start = datetime.fromisoformat(start)
                if isinstance(end, str):
                    end = datetime.fromisoformat(end)
                
                repair_time = (end - start).total_seconds() / 3600
                repair_times.append(repair_time)
        
        if not repair_times:
            return None
        
        return sum(repair_times) / len(repair_times)
    
    @staticmethod
    def calculate_availability(uptime_hours: float, total_hours: float) -> float:
        """Calculate equipment availability percentage"""
        if total_hours <= 0:
            return 0.0
        
        return (uptime_hours / total_hours) * 100
    
    @staticmethod
    def trend_analysis(data_points: List[Tuple[datetime, float]], window_size: int = 7) -> Dict[str, Any]:
        """Perform trend analysis on time series data"""
        if len(data_points) < window_size:
            return {'trend': 'insufficient_data', 'confidence': 0.0}
        
        # Sort by date
        sorted_data = sorted(data_points, key=lambda x: x[0])
        
        # Extract values for analysis
        values = [point[1] for point in sorted_data]
        
        # Simple moving average trend
        if len(values) >= window_size:
            recent_avg = np.mean(values[-window_size:])
            previous_avg = np.mean(values[-2*window_size:-window_size]) if len(values) >= 2*window_size else np.mean(values[:-window_size])
            
            if recent_avg > previous_avg * 1.05:  # 5% increase threshold
                trend = 'increasing'
            elif recent_avg < previous_avg * 0.95:  # 5% decrease threshold
                trend = 'decreasing'
            else:
                trend = 'stable'
            
            # Calculate confidence based on data consistency
            std_dev = np.std(values[-window_size:])
            mean_val = np.mean(values[-window_size:])
            cv = std_dev / mean_val if mean_val != 0 else float('inf')
            confidence = max(0.0, 1.0 - cv)  # Lower coefficient of variation = higher confidence
        else:
            trend = 'insufficient_data'
            confidence = 0.0
        
        return {
            'trend': trend,
            'confidence': confidence,
            'recent_average': recent_avg if len(values) >= window_size else None,
            'change_percentage': ((recent_avg - previous_avg) / previous_avg * 100) if len(values) >= 2*window_size and previous_avg != 0 else None
        }

class DomainKnowledgeBase:
    """Knowledge base for industrial domain expertise"""
    
    def __init__(self):
        self.equipment_criticality = {
            'safety_systems': 'critical',
            'production_equipment': 'high',
            'utility_systems': 'medium',
            'auxiliary_equipment': 'low'
        }
        
        self.maintenance_intervals = {
            'daily': 1,
            'weekly': 7,
            'monthly': 30,
            'quarterly': 90,
            'semi_annual': 180,
            'annual': 365
        }
        
        self.regulation_mapping = {
            'OSHA': 'Occupational Safety and Health Administration',
            'EPA': 'Environmental Protection Agency',
            'API': 'American Petroleum Institute',
            'ASME': 'American Society of Mechanical Engineers',
            'ISM': 'International Safety Management',
            'SOLAS': 'Safety of Life at Sea',
            'MARPOL': 'Marine Pollution Prevention'
        }
        
        self.risk_matrix = {
            ('high', 'high'): 'critical',
            ('high', 'medium'): 'high',
            ('high', 'low'): 'medium',
            ('medium', 'high'): 'high',
            ('medium', 'medium'): 'medium',
            ('medium', 'low'): 'low',
            ('low', 'high'): 'medium',
            ('low', 'medium'): 'low',
            ('low', 'low'): 'low'
        }
    
    def get_equipment_criticality(self, equipment_type: str) -> str:
        """Get equipment criticality based on type"""
        equipment_type_lower = equipment_type.lower()
        
        # Safety-critical equipment
        safety_keywords = ['safety', 'emergency', 'fire', 'gas', 'alarm', 'shutdown']
        if any(keyword in equipment_type_lower for keyword in safety_keywords):
            return 'critical'
        
        # Production equipment
        production_keywords = ['pump', 'compressor', 'turbine', 'generator', 'separator']
        if any(keyword in equipment_type_lower for keyword in production_keywords):
            return 'high'
        
        # Utility systems
        utility_keywords = ['cooling', 'heating', 'ventilation', 'lighting', 'instrument']
        if any(keyword in equipment_type_lower for keyword in utility_keywords):
            return 'medium'
        
        return 'low'  # Default for auxiliary equipment
    
    def assess_risk_level(self, probability: str, consequence: str) -> str:
        """Assess risk level based on probability and consequence"""
        return self.risk_matrix.get((probability.lower(), consequence.lower()), 'medium')
    
    def get_recommended_maintenance_interval(self, equipment_type: str, operating_conditions: str = 'normal') -> int:
        """Get recommended maintenance interval in days"""
        base_intervals = {
            'rotating_equipment': 90,
            'static_equipment': 365,
            'instrumentation': 180,
            'electrical': 365,
            'safety_systems': 30
        }
        
        # Adjust based on operating conditions
        condition_multipliers = {
            'severe': 0.5,
            'harsh': 0.7,
            'normal': 1.0,
            'light': 1.5
        }
        
        equipment_category = self._categorize_equipment(equipment_type)
        base_interval = base_intervals.get(equipment_category, 180)
        multiplier = condition_multipliers.get(operating_conditions.lower(), 1.0)
        
        return int(base_interval * multiplier)
    
    def _categorize_equipment(self, equipment_type: str) -> str:
        """Categorize equipment for maintenance planning"""
        equipment_type_lower = equipment_type.lower()
        
        rotating_keywords = ['pump', 'compressor', 'turbine', 'motor', 'fan', 'blower']
        if any(keyword in equipment_type_lower for keyword in rotating_keywords):
            return 'rotating_equipment'
        
        static_keywords = ['vessel', 'tank', 'pipe', 'heat exchanger', 'tower', 'column']
        if any(keyword in equipment_type_lower for keyword in static_keywords):
            return 'static_equipment'
        
        instrument_keywords = ['transmitter', 'indicator', 'controller', 'sensor', 'meter']
        if any(keyword in equipment_type_lower for keyword in instrument_keywords):
            return 'instrumentation'
        
        electrical_keywords = ['electrical', 'motor', 'generator', 'transformer', 'panel']
        if any(keyword in equipment_type_lower for keyword in electrical_keywords):
            return 'electrical'
        
        safety_keywords = ['safety', 'alarm', 'shutdown', 'emergency', 'fire', 'gas']
        if any(keyword in equipment_type_lower for keyword in safety_keywords):
            return 'safety_systems'
        
        return 'general'

# Factory functions for creating domain objects
def create_notification_from_data(data: Dict[str, Any]) -> Notification:
    """Create notification object from data dictionary"""
    processor = IndustrialDataProcessor()
    
    # Extract required fields
    notification_id = data.get('notification_id', f"N{datetime.now().strftime('%Y%m%d%H%M%S')}")
    title = data.get('title', data.get('description', 'Untitled Notification'))
    description = data.get('description', '')
    
    # Classify notification type and priority
    notification_type = processor.classify_notification_type(description)
    priority = processor.assess_priority(description, notification_type)
    
    # Parse dates
    created_date = data.get('created_date')
    if isinstance(created_date, str):
        created_date = datetime.fromisoformat(created_date.replace('Z', '+00:00'))
    elif not isinstance(created_date, datetime):
        created_date = datetime.now()
    
    completion_date = data.get('completion_date')
    if isinstance(completion_date, str):
        completion_date = datetime.fromisoformat(completion_date.replace('Z', '+00:00'))
    
    return Notification(
        notification_id=notification_id,
        title=title,
        description=description,
        type=notification_type,
        priority=priority,
        created_date=created_date,
        fpso=data.get('fpso'),
        work_center=data.get('work_center'),
        equipment_id=data.get('equipment_id'),
        creator=data.get('creator'),
        assigned_to=data.get('assigned_to'),
        status=data.get('status', 'open'),
        estimated_hours=data.get('estimated_hours'),
        actual_hours=data.get('actual_hours'),
        cost_estimate=data.get('cost_estimate'),
        completion_date=completion_date,
        tags=data.get('tags', []),
        attachments=data.get('attachments', [])
    )

def create_equipment_from_data(data: Dict[str, Any]) -> Equipment:
    """Create equipment object from data dictionary"""
    knowledge_base = DomainKnowledgeBase()
    
    equipment_id = data.get('equipment_id', f"E{datetime.now().strftime('%Y%m%d%H%M%S')}")
    name = data.get('name', 'Unknown Equipment')
    equipment_type = data.get('type', 'general')
    
    # Parse installation date
    installation_date = data.get('installation_date')
    if isinstance(installation_date, str):
        installation_date = datetime.fromisoformat(installation_date.replace('Z', '+00:00'))
    elif not isinstance(installation_date, datetime):
        installation_date = datetime.now()
    
    # Parse maintenance dates
    last_maintenance = data.get('last_maintenance')
    if isinstance(last_maintenance, str):
        last_maintenance = datetime.fromisoformat(last_maintenance.replace('Z', '+00:00'))
    
    next_maintenance = data.get('next_maintenance')
    if isinstance(next_maintenance, str):
        next_maintenance = datetime.fromisoformat(next_maintenance.replace('Z', '+00:00'))
    
    # Determine criticality
    criticality = data.get('criticality') or knowledge_base.get_equipment_criticality(equipment_type)
    
    # Parse status
    status_str = data.get('status', 'operational').lower()
    status = EquipmentStatus.OPERATIONAL
    for enum_status in EquipmentStatus:
        if enum_status.value == status_str:
            status = enum_status
            break
    
    return Equipment(
        equipment_id=equipment_id,
        name=name,
        type=equipment_type,
        manufacturer=data.get('manufacturer', 'Unknown'),
        model=data.get('model', 'Unknown'),
        installation_date=installation_date,
        location=data.get('location', 'Unknown'),
        fpso=data.get('fpso'),
        work_center=data.get('work_center'),
        status=status,
        criticality=criticality,
        last_maintenance=last_maintenance,
        next_maintenance=next_maintenance,
        operating_hours=data.get('operating_hours', 0.0),
        performance_indicators=data.get('performance_indicators', {}),
        maintenance_history=data.get('maintenance_history', [])
    )

# Export all classes and functions for easy access
__all__ = [
    'SafetySeverity', 'EquipmentStatus', 'NotificationType', 'Priority', 'ComplianceStatus',
    'SafetyIncident', 'Equipment', 'Notification', 'ComplianceItem',
    'IndustrialDataProcessor', 'IndustrialAnalytics', 'DomainKnowledgeBase',
    'create_notification_from_data', 'create_equipment_from_data'
]
