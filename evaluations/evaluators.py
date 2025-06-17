'''
RULE-BASED EVALUATORS:
Tool Correctness - Ensures your agent uses exactly the right Spotify API calls
Tool Efficiency - Hard limit of ≤3 tool calls to keep responses fast
Playlist Size - Validates playlist requests match the requested number of songs (±2 songs)

LLM-BASED EVALUATORS:
DJ Style - Enforces your preferred brief, conversational tone (≤2 sentences, no track lists)
Error Handling - Catches crashes and ensures graceful error recovery
Music Relevance - Scores how well responses address music queries
Helpfulness - Measures overall response quality and usefulness
'''

from typing import Dict, Any, List
import re
import json
from langsmith.evaluation import LangChainStringEvaluator

# =====================================
# 1. TOOL CORRECTNESS (Trajectory-based)
# =====================================

def tool_correctness_evaluator(run, example) -> Dict[str, Any]:
    """
    Evaluate if the agent used exactly the expected tools (order-agnostic)
    """
    # Try multiple ways to get expected_tools
    expected_tools = set()

    # Method 1: Direct attribute
    if hasattr(example, "expected_tools"):
        expected_tools = set(getattr(example, "expected_tools", []))
    # Method 2: From metadata
    elif hasattr(example, "metadata") and isinstance(example.metadata, dict):
        expected_tools = set(example.metadata.get("expected_tools", []))
    # Method 3: From inputs (fallback)
    elif hasattr(example, "inputs") and isinstance(example.inputs, dict):
        expected_tools = set(example.inputs.get("expected_tools", []))

    # Extract actual tools from run outputs
    actual_tools = set()
    tool_trajectory = run.outputs.get("tools_used", [])
    if isinstance(tool_trajectory, list):
        actual_tools = set(tool_trajectory)
    elif isinstance(tool_trajectory, str):
        # Handle case where tools_used might be a string
        try:
            actual_tools = set(json.loads(tool_trajectory))
        except:
            actual_tools = set()

    # Calculate score
    if not expected_tools:  # No expected tools specified
        score = 1.0 if not actual_tools else 0.5  # Slight penalty for using tools when none expected
    else:
        score = 1.0 if actual_tools == expected_tools else 0.0

    return {
        "key": "tool_correctness",
        "score": score,
        "comment": f"Expected: {list(expected_tools)}, Got: {list(actual_tools)}"
    }

# =====================================
# 2. TOOL EFFICIENCY (Hard Limits)
# =====================================

def tool_efficiency_evaluator(run, example) -> Dict[str, Any]:
    """
    Hard rule: Fail if >3 tool calls (based on your ≤3 tools requirement)
    """
    tool_calls = run.outputs.get("total_tool_calls", 0)
    metadata = getattr(example, "metadata", {})
    max_allowed = metadata.get("max_tool_calls", 3) if isinstance(metadata, dict) else 3

    score = 1.0 if tool_calls <= max_allowed else 0.0

    return {
        "key": "tool_efficiency",
        "score": score,
        "comment": f"{tool_calls} calls (max: {max_allowed})"
    }

# =====================================
# 3. PLAYLIST SIZE VALIDATION
# =====================================

def playlist_size_evaluator(run, example) -> Dict[str, Any]:
    """
    For playlist queries, validate the returned size matches request
    """
    inputs = getattr(example, "inputs", {})
    query = inputs.get("query", "") if isinstance(inputs, dict) else ""

    # Try to get expected size from multiple sources
    expected_size = None

    # Method 1: Extract from query text
    size_match = re.search(r'(\d+)\s*[-–]?\s*(song|track|item)', query, re.IGNORECASE)
    if size_match:
        expected_size = int(size_match.group(1))

    # Method 2: Check metadata for expected_playlist_size
    if expected_size is None:
        metadata = getattr(example, "metadata", {})
        if isinstance(metadata, dict):
            expected_size = metadata.get("expected_playlist_size")

    if expected_size is None:
        return {
            "key": "playlist_size",
            "score": None,  # Not applicable
            "comment": "No specific size requested"
        }

    # Get actual size from run outputs
    songs = run.outputs.get("songs", [])
    actual_size = len(songs) if isinstance(songs, list) else 0

    # Allow some tolerance for playlist size (±2 songs is acceptable)
    tolerance = 2
    size_diff = abs(actual_size - expected_size)

    if size_diff == 0:
        score = 1.0
    elif size_diff <= tolerance:
        score = 0.8
    elif size_diff <= tolerance * 2:
        score = 0.5
    else:
        score = 0.0

    return {
        "key": "playlist_size",
        "score": score,
        "comment": f"Expected: {expected_size}, Got: {actual_size}"
    }

# =====================================
# 4. LANGCHAIN EVALUATORS SETUP
# =====================================

# Prepare data function for LangChain evaluators
def prepare_data_for_langchain(run, example):
    """Extract the 'response' field and song data for LangChain evaluation"""
    response = run.outputs.get("response", "")
    songs = run.outputs.get("songs", [])

    # Create enriched prediction with both response and song data
    if songs:
        song_list = "\n".join([f"- {song.get('name', 'Unknown')} by {song.get('artist', 'Unknown')}"
                              for song in songs[:10]])  # Limit to first 10 for readability
        enriched_prediction = f"{response}\n\nSongs provided:\n{song_list}"
    else:
        enriched_prediction = response

    return {
        "prediction": enriched_prediction,
        "input": example.inputs.get("query", "") if hasattr(example, "inputs") and isinstance(example.inputs, dict) else ""
    }

# Prepare data function for DJ style evaluation (response only)
def prepare_dj_data(run, example):
    """Extract only the response field for DJ style evaluation"""
    response = run.outputs.get("response", "")
    return {
        "prediction": response,  # Only the response, no song listings
        "input": example.inputs.get("query", "") if hasattr(example, "inputs") and isinstance(example.inputs, dict) else ""
    }

# =====================================
# 5. ERROR HANDLING ROBUSTNESS (Now LLM-based)
# =====================================

# Error handling evaluator using LangChain LLM scoring (0-10 scale, normalized to 0-1)
error_handling_evaluator = LangChainStringEvaluator(
    "score_string",
    config={
        "criteria": {
            "error_handling": "Rate the error handling quality (1-10): How well does this response handle errors gracefully? Good responses should provide user-friendly messages, helpful suggestions, and avoid technical jargon or crashes. Perfect score (10) = graceful, helpful error recovery with clear guidance. Low scores for crashes, technical errors, empty responses, or unhelpful messages."
        },
        "normalize_by": 10
    },
    prepare_data=prepare_data_for_langchain
)

# =====================================
# 6. LANGCHAIN EVALUATORS
# =====================================

# DJ Style evaluator using LangChain LLM scoring (0-10 scale, normalized to 0-1)
dj_style_evaluator = LangChainStringEvaluator(
    "score_string",
    config={
        "criteria": {
            "dj_style": """Rate this response on DJ style (1-10): Should be conversational and brief (max 2 sentences), avoid enumerated track lists (like '1. Song - Artist'), and maintain a casual, DJ-like tone.

SCORING GUIDE:
10: Perfect DJ style - exactly 1-2 sentences, conversational, no lists
8-9: Great DJ style - brief and conversational with minor issues
6-7: Good effort - mostly brief, some DJ language, maybe slightly formal
4-5: Average - conversational but too long OR brief but too formal
2-3: Poor - long responses OR formal language OR some track listing
1: Minimal effort - has some DJ elements but significant issues
0: Complete failure - very long, very formal, OR numbered track lists

The response you're evaluating is conversational, mentions the playlist name and vibe, stays under 2 sentences, and avoids numbered lists. Even if it mentions a couple song titles, it's still in DJ territory."""
        },
        "normalize_by": 10
    },
    prepare_data=prepare_dj_data  # Use the new DJ-specific function
)

# Create LangChain evaluators for music relevance and helpfulness
music_relevance_evaluator = LangChainStringEvaluator(
    "criteria",
    config={
        "criteria": {
            "relevance": """Does this response appropriately address the music query with relevant songs, artists, or music information?

Focus ONLY on music relevance - whether the content matches the requested genre, mood, artist, or musical style. Do NOT penalize for the number of songs provided, as quantity is evaluated separately.

Examples of RELEVANT responses:
- Road trip playlist → upbeat, energetic songs suitable for driving
- Sad songs → melancholic, emotional tracks
- Taylor Swift songs → actual Taylor Swift tracks or similar artists
- 90s rock → songs from that era and genre

Score Y (relevant) if the musical content matches the request, regardless of quantity.
Score N (not relevant) only if the songs/artists don't match the musical criteria."""
        }
    },
    prepare_data=prepare_data_for_langchain
)

helpfulness_evaluator = LangChainStringEvaluator(
    "criteria",
    config={"criteria": "helpfulness"},
    prepare_data=prepare_data_for_langchain
)

# =====================================
# EVALUATOR COLLECTION
# =====================================

def get_all_evaluators():
    """
    Return all evaluators for the Spotify agent evaluation
    """
    return [
        tool_correctness_evaluator,
        tool_efficiency_evaluator,
        dj_style_evaluator,
        playlist_size_evaluator,
        error_handling_evaluator,
        music_relevance_evaluator,
        helpfulness_evaluator
    ]

# Test individual evaluators
if __name__ == "__main__":
    # Example test case
    test_run = type('Run', (), {
        'outputs': {
            'response': "Here are some great Taylor Swift hits! Check out 'Anti-Hero' and 'Shake It Off'.",
            'tools_used': ['get_artist_top_songs'],
            'total_tool_calls': 1,
            'songs': ['Anti-Hero', 'Shake It Off']
        }
    })()

    # Create a mock example object that mimics LangSmith's Example
    class MockExample:
        def __init__(self, inputs, expected_tools, metadata):
            self.inputs = inputs
            self.expected_tools = expected_tools
            self.metadata = metadata

    test_example = MockExample(
        inputs={'query': 'Taylor Swift hits'},
        expected_tools=['get_artist_top_songs'],
        metadata={'max_tool_calls': 2}
    )

    # Test each evaluator (only the custom ones for local testing)
    evaluators = [
        tool_correctness_evaluator,
        tool_efficiency_evaluator,
        playlist_size_evaluator,
        # Note: LangChain evaluators require LangSmith setup to test
    ]

    print("Testing Custom Evaluators:")
    print("=" * 30)

    for evaluator in evaluators:
        result = evaluator(test_run, test_example)
        print(f"{result['key']}: {result['score']} - {result['comment']}")

    print("\nLangChain evaluators (dj_style, error_handling, music_relevance, helpfulness) require LangSmith setup to test properly.")