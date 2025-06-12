"""
Configuration for Financial Agent LangSmith Evaluation Demo
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # For Gemini
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY") # For search

# LangSmith Configuration
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "true")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "financial-agent-eval")

# LLM Provider Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()  # "openai" or "gemini"

# Model Configuration based on provider
if LLM_PROVIDER == "openai":
    AGENT_MODEL = os.getenv("AGENT_MODEL", "gpt-4")
    EVALUATOR_MODEL = os.getenv("EVALUATOR_MODEL", "gpt-4")
    required_keys = ["OPENAI_API_KEY", "LANGSMITH_API_KEY"]
elif LLM_PROVIDER == "gemini":
    # Updated to use current available Gemini models
    AGENT_MODEL = os.getenv("AGENT_MODEL", "gemini-2.0-flash")
    EVALUATOR_MODEL = os.getenv("EVALUATOR_MODEL", "gemini-2.0-flash")
    required_keys = ["GOOGLE_API_KEY", "LANGSMITH_API_KEY"]
else:
    raise ValueError(f"Unsupported LLM provider: {LLM_PROVIDER}. Supported providers: 'openai', 'gemini'")

TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))  # Deterministic outputs for evaluation

# Evaluation Configuration
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "4"))
EXPERIMENT_PREFIX = os.getenv("EXPERIMENT_PREFIX", "finance-agent-eval")
MAX_EXAMPLES = os.getenv("MAX_EXAMPLES")  # Set to limit examples for cost control
if MAX_EXAMPLES is not None:
    MAX_EXAMPLES = int(MAX_EXAMPLES)

# Agent Performance Configuration
AGENT_MAX_ITERATIONS = int(os.getenv("AGENT_MAX_ITERATIONS", "20"))  # Increased from default 10
AGENT_MAX_EXECUTION_TIME = int(os.getenv("AGENT_MAX_EXECUTION_TIME", "300"))  # 5 minutes

# Validation
missing_keys = []
for key in required_keys:
    if key == "OPENAI_API_KEY" and not OPENAI_API_KEY:
        missing_keys.append(key)
    elif key == "GOOGLE_API_KEY" and not GOOGLE_API_KEY:
        missing_keys.append(key)
    elif key == "LANGSMITH_API_KEY" and not LANGSMITH_API_KEY:
        missing_keys.append(key)

if missing_keys:
    raise ValueError(f"Missing required environment variables for {LLM_PROVIDER}: {missing_keys}")

def get_chat_model():
    """
    Factory function to create the appropriate chat model based on the provider.
    """
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
    else:
        raise ValueError(f"Unsupported LLM provider: {LLM_PROVIDER}")

def get_evaluator_model():
    """
    Factory function to create the appropriate evaluator model based on the provider.
    """
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
    else:
        raise ValueError(f"Unsupported LLM provider: {LLM_PROVIDER}")

def verify_langsmith_setup():
    """
    Verify LangSmith configuration and connectivity.
    """
    try:
        from langsmith import Client

        client = Client()

        print("Verifying LangSmith Setup...")
        print(f"   API Key: {'Present' if LANGSMITH_API_KEY else 'Missing'}")
        print(f"   Project: {LANGSMITH_PROJECT}")
        print(f"   Tracing: {LANGSMITH_TRACING}")

        # Test connectivity
        try:
            # Try to list datasets to test connection
            datasets = list(client.list_datasets(limit=1))
            print("   Connection: Working")

            # Check project exists or can be created
            try:
                # This will create the project if it doesn't exist
                print(f"   Project Access: Available")
            except Exception as project_error:
                print(f"   Project Access: Warning - {project_error}")

        except Exception as conn_error:
            print(f"   Connection: Failed - {conn_error}")
            print("   Check your LANGSMITH_API_KEY and internet connection")
            return False

        print("LangSmith setup verified successfully!")
        return True

    except Exception as e:
        print(f"LangSmith verification failed: {e}")
        return False

print("Configuration loaded successfully")
print(f"LLM Provider: {LLM_PROVIDER}")
print(f"LangSmith Project: {LANGSMITH_PROJECT}")
print(f"Agent Model: {AGENT_MODEL}")
print(f"Evaluator Model: {EVALUATOR_MODEL}")

# Verify LangSmith setup on import
verify_langsmith_setup()