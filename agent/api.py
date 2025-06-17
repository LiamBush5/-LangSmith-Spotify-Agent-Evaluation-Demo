from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
from langsmith import Client

from .music_agent import SpotifyMusicAgent

# Initialize FastAPI app
app = FastAPI(
    title="Spotify Music Concierge API",
    description="AI-powered music recommendation service with LangSmith tracing",
    version="2.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global agent instance and LangSmith client
agent = None
langsmith_client = Client()

# Request/Response models
class MusicQueryRequest(BaseModel):
    """Music query request model"""
    query: str
    thread_id: Optional[str] = None

class FeedbackRequest(BaseModel):
    """User feedback request model"""
    trace_id: str
    feedback: int
    comment: Optional[str] = None

class MusicQueryResponse(BaseModel):
    """Music query response model"""
    response: str
    tool_trajectory: list = []
    reasoning_steps: list = []
    total_tool_calls: int = 0
    unique_tools_used: list = []
    songs_found: int = 0
    songs: list = []
    query: str
    thread_id: Optional[str] = None
    trace_id: Optional[str] = None
    success: bool = True
    error: Optional[str] = None

@app.on_event("startup")
async def startup_event():
    """Initialize agent on startup"""
    global agent
    print("Initializing Spotify Music Concierge Agent...")

    try:
        agent = SpotifyMusicAgent()
        print("Agent initialized successfully!")
        print("Ready to serve music recommendations")
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        raise e

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Spotify Music Concierge API v2.1",
        "status": "running",
        "features": [
            "LangSmith tracing enabled",
            "Brief Spotify DJ responses",
            "Structured Pydantic outputs",
            "Multi-tool music discovery",
            "Evaluation ready"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "agent_ready": agent is not None,
        "version": "2.1.0"
    }

@app.post("/chat", response_model=MusicQueryResponse)
async def chat_music(request: MusicQueryRequest):
    """Main music chat endpoint"""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    try:
        # Get response from agent
        result = agent.analyze_query(request.query, request.thread_id)

        return MusicQueryResponse(
            response=result["response"],
            tool_trajectory=result["tool_trajectory"],
            reasoning_steps=result["reasoning_steps"],
            total_tool_calls=result["total_tool_calls"],
            unique_tools_used=result["unique_tools_used"],
            songs_found=result["songs_found"],
            songs=result.get("songs", []),
            query=result["query"],
            thread_id=result["thread_id"],
            trace_id=result.get("trace_id"),
            success=not result.get("error", False),
            error=result.get("error")
        )

    except Exception as e:
        return MusicQueryResponse(
            response=f"Sorry, I encountered an error: {str(e)}",
            query=request.query,
            thread_id=request.thread_id,
            success=False,
            error=str(e)
        )

@app.post("/evaluate")
async def evaluate_agent(inputs: Dict[str, str]):
    """Evaluation endpoint for LangSmith"""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    from .music_agent import run_spotify_agent_with_project_routing

    try:
        result = run_spotify_agent_with_project_routing(inputs)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Submit user feedback for a response"""
    try:
        # Log feedback to LangSmith
        langsmith_client.create_feedback(
            key="user_feedback",
            score=request.feedback,
            trace_id=request.trace_id,
            comment=request.comment or ("👍 Thumbs up" if request.feedback == 1 else "👎 Thumbs down")
        )

        return {
            "success": True,
            "message": "Feedback submitted successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit feedback: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("Starting Spotify Music Concierge API...")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")