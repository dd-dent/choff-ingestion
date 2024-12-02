import logging
from contextlib import asynccontextmanager
from typing import Dict, List, Optional
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel

from services.database import init_db, get_db
from agents.base import AgentOrchestrator
from models.base import Conversation, ConversationSegment, Annotation

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for request/response
class ConversationRequest(BaseModel):
    text: str
    title: Optional[str] = None

class ConversationResponse(BaseModel):
    conversation_id: int
    segments_processed: int
    classifications: List[Dict]

class ConversationSegmentResponse(BaseModel):
    id: int
    content: str
    content_type: Optional[str]
    insight_type: Optional[str]
    pattern_recognition: Optional[str]
    resonance_tracking: Optional[str]

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for FastAPI"""
    logger.info("Initializing database...")
    init_db()
    logger.info("Database initialized successfully")
    yield

# Initialize FastAPI app
app = FastAPI(
    title="CHOFF Ingestion System",
    description="Local system for processing conversations using the CHOFF/PCHOFF framework",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/conversations/", response_model=ConversationResponse)
async def process_conversation(
    request: ConversationRequest,
    db: Session = Depends(get_db)
):
    """Process a new conversation through the CHOFF pipeline"""
    try:
        orchestrator = AgentOrchestrator(db)
        result = await orchestrator.process_conversation(request.text)
        return result
    except Exception as e:
        logger.error(f"Error processing conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/conversations/{conversation_id}/segments")
async def get_conversation_segments(
    conversation_id: int,
    db: Session = Depends(get_db)
) -> List[ConversationSegmentResponse]:
    """Get all segments for a specific conversation"""
    segments = db.query(ConversationSegment).filter(
        ConversationSegment.conversation_id == conversation_id
    ).all()
    
    if not segments:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return [
        ConversationSegmentResponse(
            id=segment.id,
            content=segment.content,
            content_type=segment.content_type,
            insight_type=segment.insight_type,
            pattern_recognition=segment.pattern_recognition,
            resonance_tracking=segment.resonance_tracking
        )
        for segment in segments
    ]

@app.get("/conversations/{conversation_id}/annotations")
async def get_conversation_annotations(
    conversation_id: int,
    db: Session = Depends(get_db)
) -> Dict:
    """Get all annotations for a specific conversation"""
    conversation = db.query(Conversation).get(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    segments = db.query(ConversationSegment).filter(
        ConversationSegment.conversation_id == conversation_id
    ).all()
    
    annotations = {}
    for segment in segments:
        segment_annotations = db.query(Annotation).filter(
            Annotation.segment_id == segment.id
        ).all()
        
        annotations[segment.id] = [
            {
                "type": ann.annotation_type,
                "content": ann.content,
                "created_at": ann.created_at
            }
            for ann in segment_annotations
        ]
    
    return annotations

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
