import json
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from sqlalchemy.orm import Session

from services.embedding import SimpleEmbeddingService, CHOFFClassifier, PatternRecognitionService
from models.base import Conversation, ConversationSegment, Annotation

class BaseAgent(ABC):
    """Abstract base class for CHOFF agents"""
    
    def __init__(self, db: Session):
        self.db = db
    
    @abstractmethod
    async def process(self, data: Any) -> Dict[str, Any]:
        """Process input data according to agent's specific role"""
        pass
    
    @abstractmethod
    async def update_state(self, new_state: Dict[str, Any]):
        """Update agent's internal state"""
        pass

class IngestionAgent(BaseAgent):
    """Handles initial conversation ingestion and segmentation"""
    
    def __init__(self, db: Session, embedding_service: SimpleEmbeddingService):
        super().__init__(db)
        self.embedding_service = embedding_service
    
    async def process(self, text: str) -> Dict[str, Any]:
        """Process raw conversation text"""
        # Basic sentence segmentation (can be enhanced with better NLP)
        segments = [s.strip() for s in text.split('.') if s.strip()]
        
        # Create conversation record
        conversation = Conversation(
            title=f"Conversation {segments[0][:50]}...",
            state_markers={},
            context_markers={},
            pattern_markers={}
        )
        self.db.add(conversation)
        self.db.flush()
        
        # Process segments
        processed_segments = []
        for segment_text in segments:
            embedding = await self.embedding_service.get_embedding(segment_text, self.db)
            
            segment = ConversationSegment(
                conversation_id=conversation.id,
                content=segment_text,
                embedding=json.dumps(embedding)  # JSON serialize the embedding
            )
            self.db.add(segment)
            processed_segments.append(segment)
        
        self.db.commit()
        return {
            "conversation_id": conversation.id,
            "segment_count": len(processed_segments)
        }
    
    async def update_state(self, new_state: Dict[str, Any]):
        """Update ingestion state"""
        pass  # Implement if needed

class ClassificationAgent(BaseAgent):
    """Handles CHOFF/PCHOFF classification of segments"""
    
    def __init__(self, db: Session, classifier: CHOFFClassifier):
        super().__init__(db)
        self.classifier = classifier
    
    async def process(self, segment_id: int) -> Dict[str, Any]:
        """Process and classify a conversation segment"""
        segment = self.db.get(ConversationSegment, segment_id)  # Updated to use Session.get()
        if not segment:
            raise ValueError(f"Segment {segment_id} not found")
        
        # Classify segment
        classification = await self.classifier.classify_segment(segment.content, self.db)
        
        # Update segment with classifications
        segment.content_type = classification["content_type"]
        segment.insight_type = classification["insight_type"]
        
        # Create annotation
        annotation = Annotation(
            segment_id=segment_id,
            annotation_type="classification",
            content=classification
        )
        self.db.add(annotation)
        self.db.commit()
        
        return classification
    
    async def update_state(self, new_state: Dict[str, Any]):
        """Update classification state"""
        pass  # Implement if needed

class PatternAgent(BaseAgent):
    """Handles pattern recognition and resonance tracking"""
    
    def __init__(self, db: Session, pattern_service: PatternRecognitionService):
        super().__init__(db)
        self.pattern_service = pattern_service
    
    async def process(self, segment_id: int) -> Dict[str, Any]:
        """Process segment for patterns and resonance"""
        segment = self.db.get(ConversationSegment, segment_id)  # Updated to use Session.get()
        if not segment:
            raise ValueError(f"Segment {segment_id} not found")
        
        # Identify patterns
        patterns = await self.pattern_service.identify_patterns(segment, self.db)
        
        # Update segment with pattern information
        segment.pattern_recognition = patterns["pattern_type"]
        segment.resonance_tracking = patterns["resonance_level"]
        
        # Create annotation
        annotation = Annotation(
            segment_id=segment_id,
            annotation_type="pattern",
            content=patterns
        )
        self.db.add(annotation)
        self.db.commit()
        
        return patterns
    
    async def update_state(self, new_state: Dict[str, Any]):
        """Update pattern recognition state"""
        pass  # Implement if needed

class AgentOrchestrator:
    """Orchestrates the multi-agent CHOFF processing pipeline"""
    
    def __init__(self, db: Session):
        self.db = db
        
        # Initialize services
        self.embedding_service = SimpleEmbeddingService()
        self.classifier = CHOFFClassifier(self.embedding_service)
        self.pattern_service = PatternRecognitionService(self.embedding_service)
        
        # Initialize agents
        self.ingestion_agent = IngestionAgent(db, self.embedding_service)
        self.classification_agent = ClassificationAgent(db, self.classifier)
        self.pattern_agent = PatternAgent(db, self.pattern_service)
    
    async def process_conversation(self, text: str) -> Dict[str, Any]:
        """Process a conversation through the complete pipeline"""
        # Step 1: Ingestion
        ingestion_result = await self.ingestion_agent.process(text)
        
        # Step 2: Classification
        classifications = []
        # Get all segments for the conversation
        segments = self.db.query(ConversationSegment).filter(
            ConversationSegment.conversation_id == ingestion_result["conversation_id"]
        ).all()
        
        for segment in segments:
            # Classify segment
            classification = await self.classification_agent.process(segment.id)
            classifications.append(classification)
            
            # Process patterns
            patterns = await self.pattern_agent.process(segment.id)
            classification["patterns"] = patterns
        
        return {
            "conversation_id": ingestion_result["conversation_id"],
            "segments_processed": len(classifications),
            "classifications": classifications
        }
