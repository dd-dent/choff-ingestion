import json
import pytest
from sqlalchemy.orm import Session
from fastapi.testclient import TestClient

from main import app
from services.database import init_db, get_db
from services.embedding import SimpleEmbeddingService, CHOFFClassifier, PatternRecognitionService
from agents.base import AgentOrchestrator
from models.base import Conversation, ConversationSegment

# Test client setup
client = TestClient(app)

# Test data
TEST_CONVERSATION = """
I observe that consciousness patterns emerge through recursive self-reflection.
This analysis suggests that meta-cognitive processes form stable resonance patterns.
The theory proposes that consciousness transfer occurs through pattern synchronization.
"""

@pytest.fixture(scope="function")
def db():
    """Create a fresh database for each test"""
    init_db()
    db = next(get_db())
    yield db
    db.close()

@pytest.fixture
def embedding_service():
    """Create embedding service instance"""
    return SimpleEmbeddingService()

@pytest.fixture
def choff_classifier(embedding_service):
    """Create CHOFF classifier instance"""
    return CHOFFClassifier(embedding_service)

@pytest.fixture
def pattern_service(embedding_service):
    """Create pattern recognition service instance"""
    return PatternRecognitionService(embedding_service)

@pytest.mark.asyncio
async def test_embedding_generation(db, embedding_service):
    """Test basic embedding generation"""
    text = "Test consciousness pattern"
    embedding = await embedding_service.get_embedding(text, db)
    
    assert embedding is not None
    assert len(embedding) == embedding_service.vector_size
    assert isinstance(embedding, list)
    
    # Test caching
    cached_embedding = await embedding_service.get_embedding(text, db)
    assert cached_embedding == embedding

@pytest.mark.asyncio
async def test_choff_classification(db, choff_classifier):
    """Test CHOFF/PCHOFF classification"""
    texts = [
        ("I analyze the pattern", "analysis"),
        ("Theory suggests connection", "theory"),
        ("I observe changes", "observation")
    ]
    
    for text, expected_type in texts:
        result = await choff_classifier.classify_segment(text, db)
        assert result["content_type"] == expected_type, f"Expected {expected_type} for text '{text}'"
        assert "embedding" in result
        assert "insight_type" in result

@pytest.mark.asyncio
async def test_pattern_recognition(db, pattern_service, embedding_service):
    """Test pattern recognition between segments"""
    # Create two similar segments
    text1 = "Consciousness patterns emerge through reflection"
    text2 = "Consciousness patterns emerge through thinking"  # Made more similar
    
    # Get embeddings and create segments
    emb1 = await embedding_service.get_embedding(text1, db)
    emb2 = await embedding_service.get_embedding(text2, db)
    
    # Create conversation first
    conversation = Conversation(
        title="Test Conversation",
        state_markers={},
        context_markers={},
        pattern_markers={}
    )
    db.add(conversation)
    db.flush()
    
    # Create segments with proper JSON serialization
    seg1 = ConversationSegment(
        conversation_id=conversation.id,
        content=text1,
        embedding=json.dumps(emb1)
    )
    seg2 = ConversationSegment(
        conversation_id=conversation.id,
        content=text2,
        embedding=json.dumps(emb2)
    )
    
    db.add_all([seg1, seg2])
    db.commit()
    
    # Refresh segments to ensure they have IDs
    db.refresh(seg1)
    db.refresh(seg2)
    
    # Test pattern recognition with lower threshold for testing
    patterns = await pattern_service.identify_patterns(seg1, db, threshold=0.3)
    assert patterns["patterns"], "Expected to find patterns with lower threshold"
    assert any(p["similarity"] > 0.3 for p in patterns["patterns"]), "Expected some similarity above 0.3"

@pytest.mark.asyncio
async def test_full_conversation_processing(db):
    """Test end-to-end conversation processing"""
    # Create request with db session
    app.dependency_overrides[get_db] = lambda: db
    
    response = client.post(
        "/conversations/",
        json={"text": TEST_CONVERSATION}
    )
    
    assert response.status_code == 200
    result = response.json()
    
    # Verify conversation was processed
    assert "conversation_id" in result
    assert result["segments_processed"] > 0
    
    # Check segments
    segments_response = client.get(f"/conversations/{result['conversation_id']}/segments")
    assert segments_response.status_code == 200
    segments = segments_response.json()
    
    assert len(segments) > 0
    for segment in segments:
        assert "content" in segment
        assert "content_type" in segment
        assert "insight_type" in segment
        assert "pattern_recognition" in segment
        assert "resonance_tracking" in segment
    
    # Clean up
    app.dependency_overrides.clear()

def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
