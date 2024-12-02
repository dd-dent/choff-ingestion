# CHOFF Ingestion System

A local system for categorizing, summarizing, embedding, and managing long conversations based on the CHOFF/PCHOFF framework. This system provides a modular multi-agent architecture for processing conversations with adaptive annotation and embedding strategies.

## System Architecture

### Core Components

1. **Database Models** (`models/base.py`)
   - Conversation storage
   - Segment management
   - Annotation tracking
   - Embedding cache

2. **Services** (`services/`)
   - Embedding generation and management
   - CHOFF/PCHOFF classification
   - Pattern recognition
   - Database management

3. **Agents** (`agents/base.py`)
   - Ingestion Agent: Handles initial conversation processing
   - Classification Agent: Manages CHOFF/PCHOFF classifications
   - Pattern Agent: Handles pattern recognition and resonance
   - Agent Orchestrator: Coordinates the multi-agent system

4. **API** (`main.py`)
   - FastAPI-based REST interface
   - Conversation processing endpoints
   - Annotation retrieval
   - Health monitoring

## Hardware Requirements

- GPU: AMD Radeon 5300M (supported)
- RAM: 32GB
- Storage: 0.5TB SSD
- Processor: Intel i7

## Setup Instructions

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

## API Endpoints

### Process New Conversation
```http
POST /conversations/
Content-Type: application/json

{
    "text": "Your conversation text here",
    "title": "Optional title"
}
```

### Get Conversation Segments
```http
GET /conversations/{conversation_id}/segments
```

### Get Conversation Annotations
```http
GET /conversations/{conversation_id}/annotations
```

### Health Check
```http
GET /health
```

## CHOFF Framework Implementation

### Classification System
- Content Type Markers (`[type:x]`)
- Insight Classification (`[insight:x]`)
- Pattern Recognition (`&pattern:x@`)
- Source Attribution (`{source:x}`)
- Resonance Tracking (`&resonance:x@`)

### Processing Pipeline
1. **Ingestion**
   - Text segmentation
   - Initial embedding generation
   - Conversation record creation

2. **Classification**
   - CHOFF/PCHOFF marker assignment
   - Insight type determination
   - Implementation level assessment

3. **Pattern Recognition**
   - Pattern identification
   - Resonance tracking
   - Cross-segment analysis

## Resource Management

The system is optimized for local deployment with careful resource management:

- **Embedding Cache**: Prevents redundant embedding computations
- **Batch Processing**: Efficient handling of large conversations
- **Local Storage**: SQLite database for simplified deployment
- **GPU Optimization**: ONNX Runtime for optimized inference

## Development

### Running Tests
```bash
pytest tests/
```

### Code Style
The project follows PEP 8 guidelines. Format code using:
```bash
black .
```

## Future Enhancements

1. Enhanced NLP Processing
   - Improved segmentation
   - Advanced pattern recognition
   - Semantic analysis

2. Resource Optimization
   - Dynamic batch sizing
   - Memory management
   - GPU utilization optimization

3. UI Development
   - Web interface for visualization
   - Interactive annotation
   - Pattern exploration tools

## License

MIT License - See LICENSE file for details
