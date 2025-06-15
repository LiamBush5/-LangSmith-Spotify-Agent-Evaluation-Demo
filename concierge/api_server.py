#!/usr/bin/env python3
"""
Music Concierge API Server
FastAPI server with clean endpoints for the music agent
"""

import os
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our agent
from .agent import MusicConciergeAgent
from .schema import MusicResponse, SongInfo

# Initialize FastAPI app
app = FastAPI(
    title="Music Concierge API",
    description="AI-powered music recommendation and discovery service",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global agent storage (in production, use proper session management)
agents: Dict[str, MusicConciergeAgent] = {}

# Request/Response models
class ChatRequest(BaseModel):
    """Chat request model"""
    message: str
    user_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    """Chat response model"""
    message: str
    songs: List[Dict[str, Any]] = []
    response_type: str = "general"
    metadata: Dict[str, Any] = {}
    success: bool = True
    error: Optional[str] = None

# Helper functions
def get_or_create_agent(user_id: str = "default") -> MusicConciergeAgent:
    """Get existing agent or create new one for user"""
    if user_id not in agents:
        agents[user_id] = MusicConciergeAgent(user_id=user_id)
    return agents[user_id]

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    print("🎵 Music Concierge API Server v2.0 Starting...")
    print("✨ Using refactored LCEL-based architecture")
    print("📝 Required environment variables:")
    print("   - OPENAI_API_KEY")
    print("   - SPOTIFY_CLIENT_ID")
    print("   - SPOTIFY_CLIENT_SECRET")
    print("   - TAVILY_API_KEY")

    # Check environment variables
    required_vars = ["OPENAI_API_KEY", "SPOTIFY_CLIENT_ID", "SPOTIFY_CLIENT_SECRET", "TAVILY_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("Please set them in your .env file")
        return

    try:
        # Test agent creation
        test_agent = MusicConciergeAgent(user_id="startup_test")
        print("✅ Music Concierge Agent initialized successfully!")
        print("🚀 Server ready to handle requests")
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Music Concierge API Server v2.0",
        "status": "running",
        "architecture": "LangChain v0.2 + LCEL",
        "features": [
            "Clean tool calling",
            "Structured responses",
            "Memory management",
            "Web search integration",
            "LangSmith tracing"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "architecture": "LCEL",
        "active_users": len(agents)
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint with structured response"""
    try:
        # Get or create agent for user
        agent = get_or_create_agent(request.user_id)

        # Get structured response
        response: MusicResponse = agent.chat(request.message)

        # Convert songs to dicts for JSON serialization
        songs_data = [song.dict() for song in response.songs]

        return ChatResponse(
            message=response.message,
            songs=songs_data,
            response_type=response.response_type,
            metadata=response.metadata or {},
            success=True
        )

    except Exception as e:
        print(f"❌ Chat endpoint error: {e}")

        return ChatResponse(
            message=f"I apologize, but I encountered an error: {str(e)}",
            songs=[],
            response_type="error",
            metadata={"error": str(e)},
            success=False,
            error=str(e)
        )

@app.delete("/memory/{user_id}")
async def clear_memory(user_id: str = "default"):
    """Clear conversation memory for user"""
    try:
        agent = get_or_create_agent(user_id)
        agent.clear_memory()

        return {
            "user_id": user_id,
            "message": "Memory cleared successfully",
            "success": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memory/{user_id}")
async def get_memory_summary(user_id: str = "default"):
    """Get conversation memory summary"""
    try:
        agent = get_or_create_agent(user_id)
        summary = agent.get_memory_summary()

        return {
            "user_id": user_id,
            "memory_summary": summary,
            "success": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/users")
async def get_active_users():
    """Get list of active users"""
    return {
        "active_users": list(agents.keys()),
        "total_users": len(agents),
        "success": True
    }

if __name__ == "__main__":
    import uvicorn

    print("🚀 Starting Music Concierge API Server v2.0...")
    print("📍 Server will be available at: http://127.0.0.1:8000")
    print("📖 API docs will be available at: http://127.0.0.1:8000/docs")
    print("✨ Built with LangChain v0.2 + LCEL")

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="info"
    )