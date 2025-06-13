"""
Configuration for Financial Agent LangSmith Evaluation

Centralized configuration management for API keys, models, and evaluation settings.
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# LangSmith Configuration
os.environ["LANGSMITH_TRACING_V2"] = "true"
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "financial-agent-dev")
os.environ["LANGSMITH_PROJECT"] = LANGSMITH_PROJECT

# LLM Provider Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()

# Model Configuration
if LLM_PROVIDER == "openai":
    AGENT_MODEL = os.getenv("AGENT_MODEL", "gpt-4")
    EVALUATOR_MODEL = os.getenv("EVALUATOR_MODEL", "gpt-4")
elif LLM_PROVIDER == "gemini":
    AGENT_MODEL = os.getenv("AGENT_MODEL", "gemini-2.0-flash")
    EVALUATOR_MODEL = os.getenv("EVALUATOR_MODEL", "gemini-2.0-flash")
else:
    raise ValueError(f"Unsupported LLM provider: {LLM_PROVIDER}. Use 'openai' or 'gemini'")

# General Configuration
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "4"))
# Dynamic experiment prefix: {AGENT_MODEL}-experiment
EXPERIMENT_PREFIX = os.getenv("EXPERIMENT_PREFIX", f"{AGENT_MODEL}-experiment")
AGENT_MAX_ITERATIONS = int(os.getenv("AGENT_MAX_ITERATIONS", "20"))
AGENT_MAX_EXECUTION_TIME = int(os.getenv("AGENT_MAX_EXECUTION_TIME", "300"))

# Optional cost control
MAX_EXAMPLES = os.getenv("MAX_EXAMPLES")
if MAX_EXAMPLES is not None:
    MAX_EXAMPLES = int(MAX_EXAMPLES)

def get_chat_model():
    """Create the appropriate chat model based on provider."""
    if LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=AGENT_MODEL,
            temperature=TEMPERATURE,
            api_key=OPENAI_API_KEY
        )
    elif LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=AGENT_MODEL,
            temperature=TEMPERATURE,
            google_api_key=GOOGLE_API_KEY
        )

def get_evaluator_model():
    """Create the appropriate evaluator model based on provider."""
    if LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=EVALUATOR_MODEL,
            temperature=TEMPERATURE,
            api_key=OPENAI_API_KEY
        )
    elif LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=EVALUATOR_MODEL,
            temperature=TEMPERATURE,
            google_api_key=GOOGLE_API_KEY
        )

def verify_langsmith_setup():
    """Verify LangSmith configuration and connectivity."""
    if not LANGSMITH_API_KEY:
        print("⚠️  Warning: LANGSMITH_API_KEY not set")
        return False

    try:
        from langsmith import Client
        client = Client()

        # Test connection
        list(client.list_datasets(limit=1))
        print(f"✅ LangSmith connected - Project: {LANGSMITH_PROJECT}")
        return True

    except Exception as e:
        print(f"❌ LangSmith connection failed: {e}")
        return False

# Validate required API keys
def validate_configuration():
    """Validate required configuration based on provider."""
    missing_keys = []

    if LLM_PROVIDER == "openai" and not OPENAI_API_KEY:
        missing_keys.append("OPENAI_API_KEY")
    elif LLM_PROVIDER == "gemini" and not GOOGLE_API_KEY:
        missing_keys.append("GOOGLE_API_KEY")

    if not LANGSMITH_API_KEY:
        missing_keys.append("LANGSMITH_API_KEY")

    if missing_keys:
        raise ValueError(f"Missing required environment variables: {missing_keys}")

# Initialize and validate
validate_configuration()
print(f"🤖 Config loaded - Provider: {LLM_PROVIDER}, Project: {LANGSMITH_PROJECT}")
verify_langsmith_setup()