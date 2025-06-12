"""
🎯 Quick Financial Agent Demo for Live Interview Presentation
Runs a single sophisticated example to show agent capabilities in real-time.
"""
import time
from financial_agent import FinancialAgent
from custom_evaluations import FINANCIAL_EVALUATORS
import config

def run_live_demo():
    """Run a single impressive example for live presentation."""

    print("🎯 LangChain Interview Demo: Live Financial Agent Evaluation")
    print("="*70)

    # Choose an impressive example
    demo_query = "Compare Tesla's and Apple's revenue growth over the past 3 years. Calculate their respective CAGRs and determine which has shown more consistent growth."

    print(f"📝 DEMO QUERY:")
    print(f"   {demo_query}")
    print("\n" + "="*70)

    # Initialize agent
    print("🤖 Initializing Financial Agent...")
    agent = FinancialAgent()

    # Run the analysis
    print("\n🔍 Agent Analysis in Progress...")
    print("-" * 50)

    start_time = time.time()
    result = agent.analyze_query(demo_query)
    end_time = time.time()

    # Display results
    print("\n" + "="*70)
    print("📊 AGENT RESPONSE:")
    print("="*70)
    print(result['response'])

    print(f"\n🛠️ TOOL USAGE ANALYSIS:")
    print(f"   Tools Used: {', '.join(result['unique_tools_used'])}")
    print(f"   Tool Sequence: {' → '.join(result['tool_trajectory'])}")
    print(f"   Total Tool Calls: {result['total_tool_calls']}")
    print(f"   Analysis Time: {end_time - start_time:.1f} seconds")

    # Quick evaluation demo
    print(f"\n⚖️ QUICK EVALUATION DEMO:")
    print("-" * 40)

    # Simulate evaluation results
    evaluators = FINANCIAL_EVALUATORS

    # Create mock run object for demonstration
    mock_run = type('Run', (), {
        'inputs': {"question": demo_query},
        'outputs': result
    })()

    mock_example = type('Example', (), {
        'outputs': {
            "expected_trajectory": ["financial_data_api", "financial_calculator", "tavily_search_results_json"],
            "category": "comparative_analysis"
        }
    })()

    print("Evaluating response with custom LLM-as-judge evaluators...")

    for evaluator in evaluators[:3]:  # Show first 3 for time
        try:
            eval_result = evaluator(mock_run.inputs, mock_run.outputs, mock_example.outputs)
            status = "✅" if eval_result['score'] >= 0.7 else "⚠️"
            print(f"   {status} {evaluator.__name__}: {eval_result['score']:.2f}")
        except Exception as e:
            print(f"   ❌ {evaluator.__name__}: Error - {str(e)[:50]}...")

    print(f"\n🎉 DEMO COMPLETE!")
    print("="*70)

    return result

if __name__ == "__main__":
    """
    Quick demo script for interview presentation.
    Shows agent capabilities and evaluation in under 2 minutes.
    """

    print("🚀 Running Quick Demo for Interview Presentation...")
    print("💡 This demonstrates the agent's multi-tool reasoning and evaluation capabilities.\n")

    try:
        result = run_live_demo()

        print(f"\n🎯 KEY DEMO POINTS:")
        print(f"   • Multi-tool agent with ReAct reasoning")
        print(f"   • Real financial data integration")
        print(f"   • Complex calculations and comparisons")
        print(f"   • Custom LLM-as-judge evaluators")
        print(f"   • Trajectory analysis for tool usage")
        print(f"   • Production-ready evaluation framework")

        print(f"\n📈 WHAT THIS SHOWS:")
        print(f"   • Advanced agent architecture beyond simple Q&A")
        print(f"   • Sophisticated evaluation beyond accuracy metrics")
        print(f"   • Financial domain expertise and realistic scenarios")
        print(f"   • LangSmith integration for trace analysis")
        print(f"   • Scalable framework for continuous evaluation")

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("🔧 Check your API keys in config.py or environment variables")

    print(f"\n🔗 Next: Run 'python run_evaluation.py' for full comprehensive evaluation!")
    print("="*70)