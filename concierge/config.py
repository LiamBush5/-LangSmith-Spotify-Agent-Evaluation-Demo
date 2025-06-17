"""
Configuration for Spotify Music Concierge Agent

Clean configuration following the financial agent pattern.
"""
import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# LangSmith Configuration
LANGSMITH_PROJECT = "spotify-music-concierge"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = LANGSMITH_PROJECT

# Agent Configuration
AGENT_MAX_ITERATIONS = 15
AGENT_MAX_EXECUTION_TIME = 180  # seconds

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

def get_chat_model():
    """Get configured chat model for the agent."""
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
        api_key=OPENAI_API_KEY,
    )

# Validate required environment variables
def validate_config():
    """Validate that all required environment variables are set."""
    required_vars = {
        "OPENAI_API_KEY": OPENAI_API_KEY,
        "SPOTIFY_CLIENT_ID": SPOTIFY_CLIENT_ID,
        "SPOTIFY_CLIENT_SECRET": SPOTIFY_CLIENT_SECRET,
        "TAVILY_API_KEY": TAVILY_API_KEY
    }

    missing_vars = [name for name, value in required_vars.items() if not value]

    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

    return True

# Validate on import
validate_config()
print(f"✅ Configuration loaded for project: {LANGSMITH_PROJECT}")