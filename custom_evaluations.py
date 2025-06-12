"""
Advanced Custom Evaluators for Financial Agent LangSmith Evaluation
Includes LLM-as-judge evaluators for financial accuracy, reasoning, completeness, and trajectory analysis.
Enhanced with robust structured output using Pydantic models.
"""
import json
from typing import Dict, Any, List, Optional, Union
from langsmith.evaluation import evaluate
from langsmith import Client
from pydantic import BaseModel, Field, field_validator
import config

# Enhanced Pydantic models for structured output
class EvaluationResponse(BaseModel):
    """Structured response for LLM-as-judge evaluations."""
    score: float = Field(description="Score between 0 and 1", ge=0.0, le=1.0)
    reasoning: str = Field(description="Detailed explanation of the evaluation", min_length=10)

    @field_validator('score')
    @classmethod
    def validate_score(cls, v):
        """Ensure score is properly bounded and rounded."""
        return max(0.0, min(1.0, round(v, 3)))

    @field_validator('reasoning')
    @classmethod
    def validate_reasoning(cls, v):
        """Ensure reasoning is substantive."""
        if len(v.strip()) < 10:
            raise ValueError("Reasoning must be at least 10 characters long")
        return v.strip()

class TrajectoryEvaluationResponse(BaseModel):
    """Specialized response for trajectory evaluation with additional fields."""
    score: float = Field(description="Score between 0 and 1", ge=0.0, le=1.0)
    reasoning: str = Field(description="Detailed explanation of the evaluation", min_length=10)
    used_tools: List[str] = Field(description="List of tools actually used by the agent")
    expected_tools: List[str] = Field(description="List of tools that were expected to be used")
    similarity_metric: float = Field(description="Calculated similarity score", ge=0.0, le=1.0)

    @field_validator('score', 'similarity_metric')
    @classmethod
    def validate_scores(cls, v):
        """Ensure scores are properly bounded and rounded."""
        return max(0.0, min(1.0, round(v, 3)))

class HallucinationEvaluationResponse(BaseModel):
    """Specialized response for hallucination detection with confidence scoring."""
    score: float = Field(description="Score between 0 and 1 (1 = no hallucinations)", ge=0.0, le=1.0)
    reasoning: str = Field(description="Detailed explanation of the evaluation", min_length=10)
    confidence: float = Field(description="Confidence in the evaluation", ge=0.0, le=1.0)
    potential_issues: List[str] = Field(description="List of potential hallucination issues found", default=[])

    @field_validator('score', 'confidence')
    @classmethod
    def validate_scores(cls, v):
        """Ensure scores are properly bounded and rounded."""
        return max(0.0, min(1.0, round(v, 3)))

# Initialize evaluator LLM using factory function
evaluator_llm = config.get_evaluator_model()

# Create structured evaluator LLMs for different response types
standard_evaluator_llm = evaluator_llm.with_structured_output(EvaluationResponse)
trajectory_evaluator_llm = evaluator_llm.with_structured_output(TrajectoryEvaluationResponse)
hallucination_evaluator_llm = evaluator_llm.with_structured_output(HallucinationEvaluationResponse)

def safe_structured_evaluation(llm_func, fallback_score: float = 0.0, evaluator_name: str = "unknown") -> Dict[str, Any]:
    """
    Safely execute a structured LLM evaluation with comprehensive error handling.

    Args:
        llm_func: Function that calls the LLM and returns a Pydantic model
        fallback_score: Score to return if evaluation fails
        evaluator_name: Name of the evaluator for error reporting

    Returns:
        Dictionary with evaluation results
    """
    try:
        result = llm_func()
        return result.model_dump() if hasattr(result, 'model_dump') else result.dict()
    except Exception as e:
        error_msg = f"Structured evaluation failed in {evaluator_name}: {str(e)}"
        print(f"{error_msg}")

        # Return a safe fallback structure
        return {
            "score": fallback_score,
            "reasoning": error_msg,
            "evaluation_error": True
        }

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

    Provide a detailed reasoning for your evaluation.
    """

    def evaluate():
        return standard_evaluator_llm.invoke(evaluation_prompt)

    result = safe_structured_evaluation(evaluate, 0.0, "financial_accuracy")

    return {
        "key": "financial_accuracy",
        "score": result.get("score", 0.0),
        "comment": result.get("reasoning", "Evaluation failed"),
        "evaluation_error": result.get("evaluation_error", False)
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

    Provide a detailed reasoning for your evaluation.
    """

    def evaluate():
        return standard_evaluator_llm.invoke(evaluation_prompt)

    result = safe_structured_evaluation(evaluate, 0.0, "logical_reasoning")

    return {
        "key": "logical_reasoning",
        "score": result.get("score", 0.0),
        "comment": result.get("reasoning", "Evaluation failed"),
        "evaluation_error": result.get("evaluation_error", False)
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

    Provide a detailed reasoning for your evaluation.
    """

    def evaluate():
        return standard_evaluator_llm.invoke(evaluation_prompt)

    result = safe_structured_evaluation(evaluate, 0.0, "completeness")

    return {
        "key": "completeness",
        "score": result.get("score", 0.0),
        "comment": result.get("reasoning", "Evaluation failed"),
        "evaluation_error": result.get("evaluation_error", False)
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

    Provide:
    - A detailed reasoning for your evaluation
    - Your confidence level in this assessment (0-1)
    - A list of any specific potential issues you identified
    """

    def evaluate():
        return hallucination_evaluator_llm.invoke(evaluation_prompt)

    result = safe_structured_evaluation(evaluate, 1.0, "hallucination_detection")  # Default to "no hallucinations" on error

    return {
        "key": "hallucination_detection",
        "score": result.get("score", 1.0),
        "comment": result.get("reasoning", "Evaluation failed"),
        "confidence": result.get("confidence", 0.0),
        "potential_issues": result.get("potential_issues", []),
        "evaluation_error": result.get("evaluation_error", False)
    }


def trajectory_evaluator(inputs: Dict[str, Any], outputs: Dict[str, Any], reference_outputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Enhanced evaluator for agent tool usage trajectory using structured output.
    Analyzes whether the agent used appropriate tools in a logical sequence.
    """

    question = inputs.get("question", "")
    response = outputs.get("response", "")

    # Get expected trajectory from reference outputs (check multiple possible keys)
    expected_trajectory = []
    if reference_outputs:
        expected_trajectory = (
            reference_outputs.get("expected_trajectory", []) or
            reference_outputs.get("expected_tools", []) or
            []
        )

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

    # Calculate basic trajectory score
    similarity_score = lcs_similarity(used_tools, expected_trajectory) if expected_trajectory else (1.0 if used_tools else 0.5)

    # Use LLM for more sophisticated trajectory evaluation
    evaluation_prompt = f"""
    You are evaluating the tool usage trajectory of a financial agent.

    Question: {question}
    Agent's Response: {response}
    Tools Actually Used: {used_tools}
    Expected Tools: {expected_trajectory}
    Calculated Similarity Score: {similarity_score:.3f}

    Evaluate the trajectory on a scale of 0-1 considering:
    - Appropriateness of tools chosen for the task
    - Logical sequence of tool usage
    - Efficiency (not using unnecessary tools)
    - Completeness (using all necessary tools)
    - Overall effectiveness of the tool strategy

    Provide:
    - Your overall trajectory score (0-1)
    - Detailed reasoning for your evaluation
    - List of tools that were actually used
    - List of tools that were expected/should have been used
    - The calculated similarity metric between expected and actual tools
    """

    def evaluate():
        return trajectory_evaluator_llm.invoke(evaluation_prompt)

    result = safe_structured_evaluation(evaluate, similarity_score, "trajectory_analysis")

    return {
        "key": "trajectory_analysis",
        "score": result.get("score", similarity_score),
        "comment": result.get("reasoning", f"Basic similarity score: {similarity_score:.3f}"),
        "used_tools": result.get("used_tools", used_tools),
        "expected_tools": result.get("expected_tools", expected_trajectory),
        "similarity_metric": result.get("similarity_metric", similarity_score),
        "evaluation_error": result.get("evaluation_error", False)
    }


# List of all available evaluators for easy import
FINANCIAL_EVALUATORS = [
    financial_accuracy_evaluator,
    logical_reasoning_evaluator,
    completeness_evaluator,
    hallucination_evaluator,
    trajectory_evaluator
]

def validate_evaluator_health() -> Dict[str, bool]:
    """
    Validate that all evaluators are properly configured and can handle structured output.

    Returns:
        Dictionary mapping evaluator names to their health status
    """
    health_status = {}

    for evaluator in FINANCIAL_EVALUATORS:
        try:
            # Test with minimal inputs
            test_inputs = {"question": "test"}
            test_outputs = {"response": "test response"}

            result = evaluator(test_inputs, test_outputs)

            # Check if result has required fields
            has_key = "key" in result
            has_score = "score" in result and isinstance(result["score"], (int, float))
            has_comment = "comment" in result and isinstance(result["comment"], str)

            health_status[evaluator.__name__] = has_key and has_score and has_comment

        except Exception as e:
            print(f"Evaluator {evaluator.__name__} failed health check: {e}")
            health_status[evaluator.__name__] = False

    return health_status

if __name__ == "__main__":
    print("Enhanced Custom Evaluators with Structured Output")
    print(f"Available evaluators: {[e.__name__ for e in FINANCIAL_EVALUATORS]}")

    # Run health check
    print("Running evaluator health check...")
    health = validate_evaluator_health()

    for name, status in health.items():
        status_emoji = "Healthy" if status else "Unhealthy"
        print(f"   {name}: {status_emoji}")

    healthy_count = sum(health.values())
    total_count = len(health)
    print(f"Health Summary: {healthy_count}/{total_count} evaluators are healthy")

    if healthy_count == total_count:
        print("All evaluators are ready for use!")
    else:
        print("Some evaluators need attention before use.")