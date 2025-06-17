#!/usr/bin/env python3
"""
Run the Spotify Music Concierge API Server
Updated to use the new clean SpotifyMusicAgent architecture
"""

import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    """Run the API server with the new SpotifyMusicAgent"""
    print("🎵 Spotify Music Concierge API Server v2.1")
    print("🎧 Clean Architecture with LangSmith Tracing")
    print("=" * 60)

    # Check environment variables
    required_vars = [
        "OPENAI_API_KEY",
        "SPOTIFY_CLIENT_ID",
        "SPOTIFY_CLIENT_SECRET",
        "TAVILY_API_KEY"
    ]
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\n💡 Please set these in your .env file")
        print("📋 Required for:")
        print("   - OPENAI_API_KEY: LLM and agent reasoning")
        print("   - SPOTIFY_CLIENT_ID & SECRET: Music data access")
        print("   - TAVILY_API_KEY: Web search for music trends")
        return 1

    print("✅ Environment variables found")
    print("🎵 Using new SpotifyMusicAgent architecture")
    print("📊 LangSmith tracing enabled")
    print("🚀 Starting API server...")

    try:
        # Import the new clean API server
        from agent.simple_api import app
        import uvicorn

        print("📍 Server starting at: http://127.0.0.1:8000")
        print("📖 API docs at: http://127.0.0.1:8000/docs")
        print("🎯 Endpoints:")
        print("   - POST /chat - Music queries with DJ responses")
        print("   - POST /evaluate - LangSmith evaluation endpoint")
        print("   - GET /health - Health check")

        uvicorn.run(
            app,
            host="127.0.0.1",
            port=8000,
            log_level="info"
        )
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure to install dependencies:")
        print("pip install langchain langchain-openai langsmith pydantic fastapi uvicorn python-dotenv")
        return 1
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())