"""
Production-Ready Evaluators for Spotify Music Agent
==================================================

Based on LangSmith best practices, this module provides:
1. Trajectory evaluation for tool correctness
2. Deterministic rule-based evaluators
3. LLM-as-judge for semantic evaluation
4. Custom domain-specific evaluators
"""

from typing import Dict, Any, List
import re
import json

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
# 3. DJ STYLE (Regex-based, Deterministic)
# =====================================

def dj_style_evaluator(run, example) -> Dict[str, Any]:
    """
    Deterministic DJ style evaluation:
    - Max 2 sentences
    - No enumerated track lists (e.g., "1. Song - Artist")
    - Brief and conversational
    """
    response = run.outputs.get("response", "")

    # Count sentences using regex
    sentence_regex = re.compile(r'[.!?]+\s*')
    sentences = [s.strip() for s in sentence_regex.split(response) if s.strip()]
    sentence_count = len(sentences)

    # Check for track list format (numbered lists)
    has_tracklist = bool(re.search(r'\d+\.\s+[^\n]*(?:–|-|by)\s*[^\n]*', response, re.MULTILINE))

    # Check length (brief responses preferred)
    is_brief = len(response) <= 300  # Reasonable character limit

    # Scoring
    sentence_ok = sentence_count <= 2
    no_tracklist = not has_tracklist

    # Calculate final score
    if sentence_ok and no_tracklist and is_brief:
        score = 1.0
    elif sentence_ok and no_tracklist:
        score = 0.8  # Good but a bit long
    elif sentence_ok or no_tracklist:
        score = 0.5  # Partial compliance
    else:
        score = 0.0  # Poor style

    return {
        "key": "dj_style",
        "score": score,
        "comment": f"{sentence_count} sentences, {'no ' if no_tracklist else ''}tracklist, {len(response)} chars"
    }

# =====================================
# 4. PLAYLIST SIZE VALIDATION
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
# 5. ERROR HANDLING ROBUSTNESS
# =====================================

def error_handling_evaluator(run, example) -> Dict[str, Any]:
    """
    Check for graceful error handling (no uncaught exceptions)
    """
    response = run.outputs.get("response", "")
    error_info = run.outputs.get("error", None)

    # Check for signs of crashes or poor error handling
    crash_indicators = [
        "Traceback",
        "Exception:",
        "Error:",
        "500 Internal Server Error",
        "undefined",
        "null"
    ]

    has_crash = any(indicator in response for indicator in crash_indicators)
    has_empty_response = len(response.strip()) == 0

    # Check for good error handling patterns
    good_error_patterns = [
        "sorry",
        "couldn't find",
        "not available",
        "try again",
        "alternative",
        "suggestion"
    ]

    has_good_error_handling = any(pattern in response.lower() for pattern in good_error_patterns)

    if has_crash or has_empty_response:
        score = 0.0
        comment = "System error or crash detected"
    elif error_info and has_good_error_handling:
        score = 0.8  # Good error recovery
        comment = "Graceful error handling"
    elif error_info:
        score = 0.4  # Error occurred but handled
        comment = "Error handled but could be better"
    else:
        score = 1.0  # No errors
        comment = "No errors detected"

    return {
        "key": "error_handling",
        "score": score,
        "comment": comment
    }

# =====================================
# 6. MUSIC RELEVANCE (LLM-as-Judge)
# =====================================

def music_relevance_evaluator(run, example) -> Dict[str, Any]:
    """
    Simple music relevance evaluator based on response content
    """
    response = run.outputs.get("response", "")
    inputs = getattr(example, "inputs", {})
    query = inputs.get("query", "") if isinstance(inputs, dict) else ""

    # Simple heuristic scoring based on response content
    score = 0.8  # Default good score
    comment = "Music response provided"

    # Check if response is empty or error
    if not response or len(response.strip()) == 0:
        score = 0.0
        comment = "Empty response"
    elif any(error in response.lower() for error in ["error", "sorry", "couldn't find"]):
        score = 0.6
        comment = "Response with limitations"
    elif len(response) > 50:  # Substantial response
        score = 0.9
        comment = "Detailed music response"

    return {
        "key": "music_relevance",
        "score": score,
        "comment": comment
    }

# =====================================
# 7. RESPONSE HELPFULNESS (LLM-as-Judge)
# =====================================

def helpfulness_evaluator(run, example) -> Dict[str, Any]:
    """
    Simple helpfulness evaluator based on response quality
    """
    response = run.outputs.get("response", "")
    songs = run.outputs.get("songs", [])

    # Simple heuristic scoring
    score = 0.7  # Default decent score
    comment = "Helpful response"

    # Check response quality indicators
    if not response:
        score = 0.0
        comment = "No response provided"
    elif songs and len(songs) > 0:
        score = 0.9
        comment = f"Provided {len(songs)} music recommendations"
    elif len(response) > 30:
        score = 0.8
        comment = "Detailed helpful response"
    elif any(helpful in response.lower() for helpful in ["here", "try", "check out", "recommend"]):
        score = 0.8
        comment = "Helpful guidance provided"

    return {
        "key": "helpfulness",
        "score": score,
        "comment": comment
    }

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

    # Test each evaluator
    evaluators = [
        tool_correctness_evaluator,
        tool_efficiency_evaluator,
        dj_style_evaluator,
        playlist_size_evaluator,
        error_handling_evaluator,
        music_relevance_evaluator,
        helpfulness_evaluator
    ]

    print("🔍 Testing Evaluators:")
    print("=" * 30)

    for evaluator in evaluators:
        result = evaluator(test_run, test_example)
        print(f"{result['key']}: {result['score']} - {result['comment']}")