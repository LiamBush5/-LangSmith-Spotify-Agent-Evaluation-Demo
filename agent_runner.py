#!/usr/bin/env python3
"""
Financial Agent Test Runner

A professional test runner for financial agent evaluation that routes all traces
to your designated LangSmith project. This provides an alternative to the LangSmith
evaluate() framework when you need consolidated trace visibility.

Usage:
  python agent_runner.py           # Execute all 15 test cases
  python agent_runner.py 5         # Execute 5 test cases (cost control)

Features:
- Direct agent execution bypassing LangSmith evaluate() framework
- All traces consolidated in your target project (financial-agent-dev)
- Comprehensive test coverage across financial analysis categories
- Detailed execution metrics and performance reporting
"""
import config
from financial_agent import run_financial_agent_with_project_routing
from evaluation_dataset import FINANCIAL_EVALUATION_DATASET
import time
from typing import List, Dict, Any

def execute_financial_agent_tests(max_test_cases: int = None) -> List[Dict[str, Any]]:
    """
    Execute financial agent test cases with full trace visibility.

    Args:
        max_test_cases: Maximum number of test cases to execute.
                       If None, executes all available test cases.

    Returns:
        List of test execution results with performance metrics
    """
    print(f"🧪 FINANCIAL AGENT TEST RUNNER")
    print(f"Target Project: {config.LANGSMITH_PROJECT}")
    print(f"="*80)

    # Prepare test cases
    test_cases = FINANCIAL_EVALUATION_DATASET
    if max_test_cases:
        test_cases = test_cases[:max_test_cases]
        print(f"💰 Cost Control: Executing {max_test_cases} out of {len(FINANCIAL_EVALUATION_DATASET)} test cases")
    else:
        print(f"🚀 Executing all {len(FINANCIAL_EVALUATION_DATASET)} test cases")

    execution_results = []

    for test_index, test_case in enumerate(test_cases, 1):
        print(f"\n📊 Test {test_index}/{len(test_cases)}: {test_case['category']}")
        print(f"Query: {test_case['input']}")
        print("-" * 60)

        execution_start_time = time.time()

        # Execute agent test case (traces route to target project)
        agent_result = run_financial_agent_with_project_routing({
            "input": test_case["input"]
        })

        execution_duration = time.time() - execution_start_time

        # Augment result with test metadata
        agent_result.update({
            "test_index": test_index,
            "category": test_case["category"],
            "complexity": test_case["complexity"],
            "expected_tools": test_case["expected_tools"],
            "execution_duration_seconds": execution_duration
        })

        execution_results.append(agent_result)

        print(f"✅ Completed in {execution_duration:.1f}s")
        print(f"Tools Used: {agent_result.get('total_tool_calls', 0)}")
        print(f"Response Preview: {agent_result.get('response', '')[:100]}...")

    print(f"\n🎉 All tests completed successfully!")
    print(f"📈 View all traces in LangSmith project: {config.LANGSMITH_PROJECT}")
    print(f"🔗 LangSmith Console: https://smith.langchain.com → Projects → {config.LANGSMITH_PROJECT}")

    return execution_results

def generate_execution_summary(results: List[Dict[str, Any]]) -> None:
    """
    Generate and display comprehensive test execution summary.

    Args:
        results: List of test execution results
    """
    print(f"\n📊 EXECUTION SUMMARY:")
    print(f"Total Test Cases: {len(results)}")

    # Calculate performance metrics
    total_tool_calls = sum(r.get('total_tool_calls', 0) for r in results)
    average_tools_per_query = total_tool_calls / len(results) if results else 0
    total_execution_time = sum(r.get('execution_duration_seconds', 0) for r in results)

    print(f"Average Tools per Query: {average_tools_per_query:.1f}")
    print(f"Total Execution Time: {total_execution_time:.1f}s")
    print(f"Average Time per Test: {total_execution_time/len(results):.1f}s")

    # Category distribution analysis
    category_distribution = {}
    for result in results:
        category = result.get('category', 'unknown')
        category_distribution[category] = category_distribution.get(category, 0) + 1

    print(f"Category Distribution: {dict(category_distribution)}")

    # Performance insights
    if average_tools_per_query > 4:
        print(f"💡 High tool usage detected - consider query optimization")

    if total_execution_time > 300:  # 5 minutes
        print(f"⏱️  Extended execution time - {total_execution_time/60:.1f} minutes total")

def main():
    """Main execution entry point with professional argument handling."""
    import sys

    # Parse command line arguments
    max_test_cases = None
    if len(sys.argv) > 1:
        try:
            max_test_cases = int(sys.argv[1])
            if max_test_cases <= 0:
                raise ValueError("Test case count must be positive")
        except ValueError as e:
            print("Error: Invalid argument provided")
            print("Usage: python agent_runner.py [max_test_cases]")
            print("  Execute without arguments to run all test cases")
            print("  Specify max_test_cases to limit execution for cost control")
            print("  Example: python agent_runner.py 5")
            sys.exit(1)

    try:
        # Execute test cases
        execution_results = execute_financial_agent_tests(max_test_cases)

        # Generate comprehensive summary
        generate_execution_summary(execution_results)

        print(f"\n✅ Test execution completed successfully!")

    except KeyboardInterrupt:
        print(f"\n⚠️  Test execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test execution failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()