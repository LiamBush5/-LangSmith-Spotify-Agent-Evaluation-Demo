"""
Configuration for Financial Agent LangSmith Evaluation Demo
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# LangSmith Configuration
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "true")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "financial-agent-eval")

# Model Configuration
AGENT_MODEL = "gpt-4"
EVALUATOR_MODEL = "gpt-4"
TEMPERATURE = 0  # Deterministic outputs for evaluation

# Evaluation Configuration
MAX_CONCURRENCY = 4
EXPERIMENT_PREFIX = "finance-agent-eval"

# Validation
required_keys = ["OPENAI_API_KEY", "LANGSMITH_API_KEY"]
missing_keys = [key for key in required_keys if not globals()[key]]

if missing_keys:
    raise ValueError(f"Missing required environment variables: {missing_keys}")

print(" Configuration loaded successfully")
print(f" LangSmith Project: {LANGSMITH_PROJECT}")
print(f"Agent Model: {AGENT_MODEL}")
print(f"Evaluator Model: {EVALUATOR_MODEL}")