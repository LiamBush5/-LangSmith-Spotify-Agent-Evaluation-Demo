"""
Advanced Custom Evaluators for Financial Agent LangSmith Evaluation
Includes LLM-as-judge evaluators for financial accuracy, reasoning, completeness, and trajectory analysis.
"""
import json
from typing import Dict, Any, List, Optional
from langchain_openai import ChatOpenAI
from langsmith.evaluation import evaluate
from langsmith import Client
import config

# Initialize evaluator LLM
evaluator_llm = ChatOpenAI(
    model=config.EVALUATOR_MODEL,
    temperature=0,
    api_key=config.OPENAI_API_KEY
)

def financial_accuracy_evaluator(inputs: Dict[str, Any], outputs: Dict[str, Any], reference_outputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    LLM-as-judge evaluator for financial accuracy.
    Checks if numerical facts, percentages, and financial data are correct.
    """

    question = inputs.get("question", "")
    response = outputs.get("response", "")
    expected_response = reference_outputs.get("response", "") if reference_outputs else ""

    # Create evaluation prompt
    evaluation_prompt = f"""
    You are a financial expert evaluating the accuracy of financial information.

    Question: {question}
    Response: {response}
    Expected Response: {expected_response}

    Evaluate the financial accuracy of the response on a scale of 0-1:
    - 1.0: All financial facts, numbers, and calculations are correct
    - 0.8: Mostly accurate with minor errors
    - 0.6: Generally accurate but some notable errors
    - 0.4: Several inaccuracies in financial data
    - 0.2: Major financial errors
    - 0.0: Completely inaccurate financial information

    Focus on:
    - Numerical accuracy (percentages, ratios, calculations)
    - Financial terminology usage
    - Market data correctness
    - Mathematical calculations

    Respond with a JSON object containing:
    - "score": float between 0 and 1
    - "reasoning": string explaining your evaluation
    """

    try:
        result = evaluator_llm.invoke(evaluation_prompt)
        content = result.content.strip()

        # Try to parse JSON response
        try:
            parsed = json.loads(content)
            score = float(parsed.get("score", 0.0))
            reasoning = parsed.get("reasoning", "No reasoning provided")
        except (json.JSONDecodeError, ValueError):
            # Fallback: extract score from text
            score = 0.5  # Default score
            reasoning = f"Could not parse evaluator response: {content}"

        return {
            "key": "financial_accuracy",
            "score": max(0.0, min(1.0, score)),  # Ensure score is between 0 and 1
            "comment": reasoning
        }

    except Exception as e:
        return {
            "key": "financial_accuracy",
            "score": 0.0,
            "comment": f"Evaluation error: {str(e)}"
        }


def logical_reasoning_evaluator(inputs: Dict[str, Any], outputs: Dict[str, Any], reference_outputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    LLM-as-judge evaluator for logical reasoning quality.
    Assesses the coherence and logical flow of the agent's reasoning.
    """

    question = inputs.get("question", "")
    response = outputs.get("response", "")

    evaluation_prompt = f"""
    You are evaluating the logical reasoning quality of a financial agent's response.

    Question: {question}
    Response: {response}

    Evaluate the logical reasoning on a scale of 0-1:
    - 1.0: Clear, coherent, well-structured reasoning with logical flow
    - 0.8: Good reasoning with minor logical gaps
    - 0.6: Generally logical but some unclear connections
    - 0.4: Some logical issues or unclear reasoning
    - 0.2: Poor logical structure with significant gaps
    - 0.0: Illogical or incoherent reasoning

    Focus on:
    - Logical flow and structure
    - Clear cause-and-effect relationships
    - Appropriate use of evidence
    - Coherent argumentation
    - Step-by-step reasoning clarity

    Respond with a JSON object containing:
    - "score": float between 0 and 1
    - "reasoning": string explaining your evaluation
    """

    try:
        result = evaluator_llm.invoke(evaluation_prompt)
        content = result.content.strip()

        try:
            parsed = json.loads(content)
            score = float(parsed.get("score", 0.0))
            reasoning = parsed.get("reasoning", "No reasoning provided")
        except (json.JSONDecodeError, ValueError):
            score = 0.5
            reasoning = f"Could not parse evaluator response: {content}"

        return {
            "key": "logical_reasoning",
            "score": max(0.0, min(1.0, score)),
            "comment": reasoning
        }

    except Exception as e:
        return {
            "key": "logical_reasoning",
            "score": 0.0,
            "comment": f"Evaluation error: {str(e)}"
        }


def completeness_evaluator(inputs: Dict[str, Any], outputs: Dict[str, Any], reference_outputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    LLM-as-judge evaluator for response completeness.
    Checks if the response comprehensively addresses all aspects of the question.
    """

    question = inputs.get("question", "")
    response = outputs.get("response", "")

    evaluation_prompt = f"""
    You are evaluating how completely a financial agent answered a question.

    Question: {question}
    Response: {response}

    Evaluate the completeness on a scale of 0-1:
    - 1.0: Fully addresses all aspects of the question comprehensively
    - 0.8: Addresses most aspects with minor omissions
    - 0.6: Covers main points but misses some important aspects
    - 0.4: Partial answer with significant gaps
    - 0.2: Minimal coverage of the question
    - 0.0: Does not address the question

    Focus on:
    - Coverage of all question components
    - Depth of analysis where appropriate
    - Addressing implicit requirements
    - Providing sufficient detail
    - Answering sub-questions if present

    Respond with a JSON object containing:
    - "score": float between 0 and 1
    - "reasoning": string explaining your evaluation
    """

    try:
        result = evaluator_llm.invoke(evaluation_prompt)
        content = result.content.strip()

        try:
            parsed = json.loads(content)
            score = float(parsed.get("score", 0.0))
            reasoning = parsed.get("reasoning", "No reasoning provided")
        except (json.JSONDecodeError, ValueError):
            score = 0.5
            reasoning = f"Could not parse evaluator response: {content}"

        return {
            "key": "completeness",
            "score": max(0.0, min(1.0, score)),
            "comment": reasoning
        }

    except Exception as e:
        return {
            "key": "completeness",
            "score": 0.0,
            "comment": f"Evaluation error: {str(e)}"
        }


def hallucination_evaluator(inputs: Dict[str, Any], outputs: Dict[str, Any], reference_outputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    LLM-as-judge evaluator for detecting hallucinations.
    Identifies unsupported claims or fabricated financial information.
    """

    question = inputs.get("question", "")
    response = outputs.get("response", "")

    evaluation_prompt = f"""
    You are evaluating a financial agent's response for hallucinations or unsupported claims.

    Question: {question}
    Response: {response}

    Evaluate for hallucinations on a scale of 0-1 (where 1 = no hallucinations):
    - 1.0: All claims are well-supported or appropriately qualified
    - 0.8: Mostly accurate with minor unsupported details
    - 0.6: Some questionable claims but generally reliable
    - 0.4: Several unsupported or suspicious claims
    - 0.2: Many likely fabricated details
    - 0.0: Significant hallucinations or fabricated information

    Look for:
    - Specific financial data without clear sources
    - Overly precise numbers that seem fabricated
    - Claims about specific companies/events without context
    - Contradictory information within the response
    - Unrealistic financial scenarios

    Respond with a JSON object containing:
    - "score": float between 0 and 1
    - "reasoning": string explaining your evaluation
    """

    try:
        result = evaluator_llm.invoke(evaluation_prompt)
        content = result.content.strip()

        try:
            parsed = json.loads(content)
            score = float(parsed.get("score", 0.0))
            reasoning = parsed.get("reasoning", "No reasoning provided")
        except (json.JSONDecodeError, ValueError):
            score = 0.5
            reasoning = f"Could not parse evaluator response: {content}"

        return {
            "key": "hallucination_detection",
            "score": max(0.0, min(1.0, score)),
            "comment": reasoning
        }

    except Exception as e:
        return {
            "key": "hallucination_detection",
            "score": 0.0,
            "comment": f"Evaluation error: {str(e)}"
        }


def trajectory_evaluator(inputs: Dict[str, Any], outputs: Dict[str, Any], reference_outputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Evaluator for agent tool usage trajectory.
    Analyzes whether the agent used appropriate tools in a logical sequence.
    """

    question = inputs.get("question", "")
    response = outputs.get("response", "")

    # Get expected trajectory from reference outputs
    expected_trajectory = reference_outputs.get("expected_trajectory", []) if reference_outputs else []

    # Extract actual tool usage from outputs - check multiple possible keys
    used_tools = []
    if "unique_tools_used" in outputs:
        used_tools = outputs["unique_tools_used"]
    elif "tool_trajectory" in outputs:
        # Get unique tools from trajectory, preserving order
        seen = set()
        used_tools = []
        for tool in outputs["tool_trajectory"]:
            if tool not in seen:
                used_tools.append(tool)
                seen.add(tool)
    elif "tools_used" in outputs:
        used_tools = outputs["tools_used"]
    else:
        # Fallback: extract from response text
        if "financial_data_api" in response.lower():
            used_tools.append("financial_data_api")
        if "financial_calculator" in response.lower():
            used_tools.append("financial_calculator")
        if "tavily_search" in response.lower() or "search" in response.lower():
            used_tools.append("tavily_search")

    # Calculate trajectory similarity using longest common subsequence
    def lcs_similarity(seq1: List[str], seq2: List[str]) -> float:
        """Calculate similarity based on longest common subsequence."""
        if not seq1 and not seq2:
            return 1.0
        if not seq1 or not seq2:
            return 0.0

        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        lcs_length = dp[m][n]
        max_length = max(len(seq1), len(seq2))
        return lcs_length / max_length if max_length > 0 else 1.0

    # Calculate trajectory score
    if expected_trajectory:
        trajectory_score = lcs_similarity(used_tools, expected_trajectory)
        reasoning = f"Used tools: {used_tools}, Expected: {expected_trajectory}, LCS similarity: {trajectory_score:.2f}"
    else:
        # If no expected trajectory, evaluate based on tool appropriateness
        trajectory_score = 1.0 if used_tools else 0.5
        reasoning = f"Used tools: {used_tools}. No expected trajectory provided."

    return {
        "key": "trajectory_analysis",
        "score": max(0.0, min(1.0, trajectory_score)),
        "comment": reasoning
    }


# List of all available evaluators for easy import
FINANCIAL_EVALUATORS = [
    financial_accuracy_evaluator,
    logical_reasoning_evaluator,
    completeness_evaluator,
    hallucination_evaluator,
    trajectory_evaluator
]

# Quick test functions
def test_evaluators():
    """Test the evaluators with sample data."""
    print("🧪 Testing Custom Evaluators...")

    # Sample run data
    sample_run = type('Run', (), {
        'inputs': {"question": "What is Apple's current stock price?"},
        'outputs': {
            "response": "Apple's current stock price is approximately $150.00 with a market cap of $2.4 trillion.",
            "tool_trajectory": ["financial_data_api"],
            "reasoning_steps": [{"tool": "financial_data_api", "input": "AAPL price", "output": "Current price: $150"}],
            "unique_tools_used": ["financial_data_api"],
            "total_tool_calls": 1
        }
    })()

    sample_example = type('Example', (), {
        'outputs': {
            "response": "Apple's stock price is around $150 with market cap of $2.4T",
            "expected_trajectory": ["financial_data_api"],
            "category": "stock_analysis"
        }
    })()

    evaluators = FINANCIAL_EVALUATORS

    for evaluator in evaluators:
        print(f"\n�� Testing {evaluator.__name__}...")
        try:
            result = evaluator(sample_run.inputs, sample_run.outputs, sample_example.outputs)
            print(f"   Score: {result['score']}")
            print(f"   Comment: {result['comment'][:100]}...")
        except Exception as e:
            print(f"   ❌ Error: {e}")

    print("\n✅ Evaluator testing complete!")

if __name__ == "__main__":
    test_evaluators()