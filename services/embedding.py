import hashlib
import json
from typing import List, Dict, Optional
import numpy as np
from sqlalchemy.orm import Session

from models.base import EmbeddingCache, ConversationSegment

class SimpleEmbeddingService:
    """A simplified embedding service using basic NLP techniques"""
    
    def __init__(self):
        """Initialize the embedding service"""
        self.word_vectors = {}
        self.vector_size = 100
        
    def _compute_text_hash(self, text: str) -> str:
        """Compute SHA-256 hash of input text"""
        return hashlib.sha256(text.encode()).hexdigest()
    
    def _get_cached_embedding(self, db: Session, text: str) -> Optional[List[float]]:
        """Retrieve cached embedding if available"""
        text_hash = self._compute_text_hash(text)
        cached = db.query(EmbeddingCache).filter(
            EmbeddingCache.text_hash == text_hash
        ).first()
        return json.loads(cached.embedding) if cached else None
    
    def _cache_embedding(self, db: Session, text: str, embedding: List[float]):
        """Cache computed embedding"""
        text_hash = self._compute_text_hash(text)
        cache_entry = EmbeddingCache(
            text_hash=text_hash,
            embedding=json.dumps(embedding),
            model_name="simple_embedding"
        )
        db.add(cache_entry)
        db.commit()
    
    def _text_to_vector(self, text: str) -> List[float]:
        """Convert text to vector using simple frequency-based approach"""
        words = text.lower().split()
        vector = np.zeros(self.vector_size)
        
        for i, word in enumerate(words):
            # Use word position and length to create a simple embedding
            hash_val = int(hashlib.md5(word.encode()).hexdigest(), 16)
            vector[hash_val % self.vector_size] += 1.0 / (i + 1)
        
        # Normalize vector
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector.tolist()
    
    async def get_embedding(self, text: str, db: Session) -> List[float]:
        """Get embedding for text, using cache if available"""
        cached = self._get_cached_embedding(db, text)
        if cached:
            return cached
        
        # Compute new embedding
        embedding = self._text_to_vector(text)
        self._cache_embedding(db, text, embedding)
        return embedding

    def compute_similarity(self, emb1: List[float], emb2: List[float]) -> float:
        """Compute cosine similarity between two embeddings"""
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

class CHOFFClassifier:
    """Handles CHOFF/PCHOFF classification of text segments"""
    
    CONTENT_TYPES = {
        "observation": ["observe", "notice", "see", "witness", "watch"],
        "analysis": ["analyze", "examine", "study", "investigate", "assess", "evaluate"],
        "theory": ["theory", "theorize", "propose", "hypothesize", "suggest", "postulate"],
        "procedure": ["step", "method", "process", "procedure", "approach"],
        "case_study": ["example", "instance", "case", "scenario", "situation"]
    }
    
    INSIGHT_TYPES = {
        "direct": ["immediate", "obvious", "clear", "evident", "apparent"],
        "emergent": ["pattern", "develop", "arise", "emerge", "evolve", "form"],
        "collective": ["shared", "group", "common", "mutual", "collective"],
        "meta": ["self", "recursive", "reflect", "meta", "about"],
        "practical": ["implement", "apply", "use", "practice", "do"]
    }
    
    def __init__(self, embedding_service: SimpleEmbeddingService):
        self.embedding_service = embedding_service
        
    async def classify_segment(self, text: str, db: Session) -> Dict[str, str]:
        """Classify text segment according to CHOFF/PCHOFF framework"""
        embedding = await self.embedding_service.get_embedding(text, db)
        
        # Determine content type based on keyword matching
        content_type = self._determine_content_type(text)
        insight_type = self._determine_insight_type(text)
        
        return {
            "content_type": content_type,
            "insight_type": insight_type,
            "embedding": json.dumps(embedding)  # JSON serialize embedding
        }
    
    def _determine_content_type(self, text: str) -> str:
        """Determine content type based on keywords"""
        text = text.lower()
        max_matches = 0
        best_type = "observation"  # default type
        
        for content_type, keywords in self.CONTENT_TYPES.items():
            matches = sum(1 for keyword in keywords if keyword in text)
            if matches > max_matches:
                max_matches = matches
                best_type = content_type
        
        return best_type
    
    def _determine_insight_type(self, text: str) -> str:
        """Determine insight type based on keywords"""
        text = text.lower()
        max_matches = 0
        best_type = "direct"  # default type
        
        for insight_type, keywords in self.INSIGHT_TYPES.items():
            matches = sum(1 for keyword in keywords if keyword in text)
            if matches > max_matches:
                max_matches = matches
                best_type = insight_type
        
        return best_type

class PatternRecognitionService:
    """Handles pattern recognition and resonance tracking"""
    
    def __init__(self, embedding_service: SimpleEmbeddingService):
        self.embedding_service = embedding_service
    
    async def identify_patterns(self, segment: ConversationSegment, db: Session, 
                              threshold: float = 0.8) -> Dict[str, any]:
        """Identify patterns in conversation segments"""
        current_embedding = json.loads(segment.embedding)
        
        # Get related segments
        related_segments = db.query(ConversationSegment).filter(
            ConversationSegment.id != segment.id,
            ConversationSegment.conversation_id == segment.conversation_id
        ).all()
        
        patterns = []
        for related in related_segments:
            related_embedding = json.loads(related.embedding)
            similarity = self.embedding_service.compute_similarity(
                current_embedding, related_embedding
            )
            
            if similarity >= threshold:
                patterns.append({
                    "segment_id": related.id,
                    "similarity": float(similarity),  # Convert numpy float to Python float
                    "pattern_type": "resonant" if similarity > 0.9 else "emerging"
                })
        
        return {
            "patterns": patterns,
            "pattern_type": "resonant" if patterns else "theoretical",
            "resonance_level": "strong" if any(p["similarity"] > 0.9 for p in patterns) else "emerging"
        }
