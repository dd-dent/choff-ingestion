from datetime import datetime
from typing import List, Optional, Dict
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON, Text
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class Conversation(Base):
    """Base model for storing conversations"""
    __tablename__ = "conversations"
    
    id = Column(Integer, primary_key=True)
    title = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # CHOFF Framework Markers
    state_markers = Column(JSON)  # {state:type}
    context_markers = Column(JSON)  # [context:type]
    pattern_markers = Column(JSON)  # &pattern:type&
    
    # Segments relationship
    segments = relationship("ConversationSegment", back_populates="conversation")

class ConversationSegment(Base):
    """Model for storing conversation segments with CHOFF annotations"""
    __tablename__ = "conversation_segments"
    
    id = Column(Integer, primary_key=True)
    conversation_id = Column(Integer, ForeignKey('conversations.id'))
    content = Column(Text)
    embedding = Column(JSON)  # Store embeddings as JSON array
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # PCHOFF Classifications
    content_type = Column(String(50))  # [type:x]
    insight_type = Column(String(50))  # [insight:x]
    implementation_level = Column(String(50))  # [level:x]
    pattern_recognition = Column(String(50))  # &pattern:x@
    source_attribution = Column(String(50))  # {source:x}
    resonance_tracking = Column(String(50))  # &resonance:x@
    
    # Relationships
    conversation = relationship("Conversation", back_populates="segments")
    annotations = relationship("Annotation", back_populates="segment")

class Annotation(Base):
    """Model for storing detailed annotations and metadata"""
    __tablename__ = "annotations"
    
    id = Column(Integer, primary_key=True)
    segment_id = Column(Integer, ForeignKey('conversation_segments.id'))
    annotation_type = Column(String(50))
    content = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    segment = relationship("ConversationSegment", back_populates="annotations")

class EmbeddingCache(Base):
    """Cache for storing computed embeddings"""
    __tablename__ = "embedding_cache"
    
    id = Column(Integer, primary_key=True)
    text_hash = Column(String(64), unique=True)  # SHA-256 hash of text
    embedding = Column(JSON)
    model_name = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)
