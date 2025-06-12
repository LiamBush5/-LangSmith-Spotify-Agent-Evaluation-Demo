"""
🚀 LangSmith Financial Agent Evaluation Demo
Comprehensive evaluation experiment showcasing advanced agent evaluation capabilities.

This script demonstrates:
- Multi-tool financial agent with ReAct reasoning
- Custom LLM-as-judge evaluators
- Trajectory analysis and tool usage evaluation
- Financial accuracy, reasoning, and completeness metrics
- Realistic financial scenarios with complex calculations
"""
import asyncio
import pandas as pd
from typing import Dict, List, Any
import time
from datetime import datetime

from langsmith import Client
from langsmith.evaluation import evaluate

import config
from financial_agent import run_financial_agent
from evaluation_dataset import create_langsmith_dataset, FINANCIAL_EVALUATION_DATASET
from custom_evaluations import FINANCIAL_EVALUATORS

# Initialize LangSmith client
client = Client()

class FinancialAgentEvaluationDemo:
    """
    Main class for running the comprehensive financial agent evaluation.
    """

    def __init__(self):
        self.client = client
        self.experiment_name = f"{config.EXPERIMENT_PREFIX}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.dataset_name = "Financial-Agent-Evaluation-Dataset"
        self.dataset_id = None
        self.evaluators = FINANCIAL_EVALUATORS

        print("🎯 Financial Agent Evaluation Demo Initialized")
        print(f"📊 Experiment Name: {self.experiment_name}")
        print(f"🗂️ Dataset: {self.dataset_name}")
        print(f"⚖️ Evaluators: {[e.__name__ for e in self.evaluators]}")

    def setup_dataset(self) -> str:
        """Create and populate the evaluation dataset."""
        print("\n" + "="*60)
        print("🗂️ SETTING UP EVALUATION DATASET")
        print("="*60)

        try:
            self.dataset_id = create_langsmith_dataset(self.dataset_name)
            print(f"✅ Dataset ready: {self.dataset_id}")
            return self.dataset_id
        except Exception as e:
            print(f"❌ Dataset setup failed: {e}")
            raise

    def run_evaluation_experiment(self) -> Dict[str, Any]:
        """Run the comprehensive evaluation experiment."""
        print("\n" + "="*60)
        print("🔬 RUNNING EVALUATION EXPERIMENT")
        print("="*60)

        print(f"🎯 Target Function: run_financial_agent")
        print(f"🗂️ Dataset: {self.dataset_name}")
        print(f"⚖️ Evaluators: {len(self.evaluators)} custom evaluators")
        print(f"🚀 Max Concurrency: {config.MAX_CONCURRENCY}")

        try:
            # Run the evaluation
            results = evaluate(
                run_financial_agent,  # target as first positional argument
                data=self.dataset_name,
                evaluators=self.evaluators,
                experiment_prefix=config.EXPERIMENT_PREFIX,
                max_concurrency=config.MAX_CONCURRENCY,
                client=self.client
            )

            print(f"\n✅ Evaluation completed!")
            print(f"🔗 View results: {results.experiment_url}")

            return {
                "experiment_url": getattr(results, 'experiment_url', 'URL not available'),
                "experiment_name": getattr(results, 'experiment_name', 'Experiment completed'),
                "results": results
            }

        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            raise

    def analyze_results(self, results) -> pd.DataFrame:
        """Analyze and summarize the evaluation results."""
        print("\n" + "="*60)
        print("📊 ANALYZING EVALUATION RESULTS")
        print("="*60)

        try:
            # Convert results to DataFrame for analysis
            df = results.to_pandas()

            # Calculate summary statistics
            evaluator_columns = [col for col in df.columns if any(eval_name in col for eval_name in
                                [e.__name__ for e in self.evaluators])]

            print("\n📈 EVALUATION SUMMARY:")
            print("-" * 40)

            for eval_name in [e.__name__ for e in self.evaluators]:
                score_cols = [col for col in df.columns if eval_name in col and 'score' in col]
                if score_cols:
                    scores = df[score_cols[0]].dropna()
                    if len(scores) > 0:
                        avg_score = scores.mean()
                        pass_rate = (scores >= 0.7).mean() * 100
                        print(f"  {eval_name:20} | Avg: {avg_score:.3f} | Pass Rate: {pass_rate:.1f}%")

            # Category analysis
            if 'outputs.category' in df.columns:
                print(f"\n📁 PERFORMANCE BY CATEGORY:")
                print("-" * 40)
                category_performance = df.groupby('outputs.category').agg({
                    col: 'mean' for col in evaluator_columns if 'score' in col
                }).round(3)
                print(category_performance)

            # Complexity analysis
            if 'outputs.complexity' in df.columns:
                print(f"\n⚡ PERFORMANCE BY COMPLEXITY:")
                print("-" * 40)
                complexity_performance = df.groupby('outputs.complexity').agg({
                    col: 'mean' for col in evaluator_columns if 'score' in col
                }).round(3)
                print(complexity_performance)

            return df

        except Exception as e:
            print(f"❌ Results analysis failed: {e}")
            return pd.DataFrame()

    def identify_insights(self, df: pd.DataFrame) -> List[str]:
        """Generate insights from the evaluation results."""
        insights = []

        try:
            # Tool usage insights
            if 'outputs.unique_tools_used' in df.columns:
                tool_usage = df['outputs.unique_tools_used'].apply(lambda x: len(x) if x else 0)
                avg_tools = tool_usage.mean()
                insights.append(f"Average tools used per query: {avg_tools:.1f}")

            # Performance insights
            score_columns = [col for col in df.columns if 'score' in col and any(eval_name in col for eval_name in
                            [e.__name__ for e in self.evaluators])]

            if score_columns:
                overall_performance = df[score_columns].mean().mean()
                insights.append(f"Overall agent performance: {overall_performance:.1%}")

                # Find best and worst performing categories
                if 'outputs.category' in df.columns and len(score_columns) > 0:
                    category_scores = df.groupby('outputs.category')[score_columns[0]].mean()
                    best_category = category_scores.idxmax()
                    worst_category = category_scores.idxmin()
                    insights.append(f"Best performing category: {best_category} ({category_scores[best_category]:.2f})")
                    insights.append(f"Most challenging category: {worst_category} ({category_scores[worst_category]:.2f})")

            # Trajectory insights
            trajectory_cols = [col for col in df.columns if 'trajectory_quality' in col and 'score' in col]
            if trajectory_cols:
                trajectory_score = df[trajectory_cols[0]].mean()
                insights.append(f"Tool usage efficiency: {trajectory_score:.1%}")

        except Exception as e:
            insights.append(f"Error generating insights: {e}")

        return insights

    def generate_report(self, results, df: pd.DataFrame, insights: List[str]) -> str:
        """Generate a comprehensive evaluation report."""
        report = f"""
🎯 FINANCIAL AGENT EVALUATION REPORT
{'='*60}

📊 EXPERIMENT DETAILS:
  • Experiment: {self.experiment_name}
          • Dataset: {self.dataset_name} ({len(FINANCIAL_EVALUATION_DATASET)} examples)
  • Evaluators: {len(self.evaluators)} custom LLM-as-judge evaluators
  • Model: {config.AGENT_MODEL}

🔗 LANGSMITH LINKS:
  • Experiment URL: {results['experiment_url']}
  • Project: {config.LANGSMITH_PROJECT}

📈 KEY INSIGHTS:
"""
        for insight in insights:
            report += f"  • {insight}\n"

        report += f"""
⚖️ EVALUATION CRITERIA:
  • Financial Accuracy: Numerical facts and calculations
  • Logical Reasoning: Coherence and soundness of analysis
  • Completeness: All aspects of questions addressed
  • Hallucination Check: No unsupported claims
  • Trajectory Quality: Appropriate tool usage patterns

🛠️ TECHNICAL ARCHITECTURE:
  • ReAct Agent with financial tools (data API, calculator, search)
  • LLM-as-judge evaluators using {config.EVALUATOR_MODEL}
  • Trajectory analysis for tool usage optimization
  • Comprehensive test coverage across financial scenarios

💡 DEMO HIGHLIGHTS:
  • Real-time financial data integration
  • Multi-step reasoning with tool orchestration
  • Advanced evaluation metrics beyond simple accuracy
  • Production-ready evaluation framework
  • Scalable to larger datasets and continuous monitoring

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        return report

    def run_full_demo(self) -> Dict[str, Any]:
        """Run the complete evaluation demo from start to finish."""
        print("\n" + "🚀" + " STARTING COMPREHENSIVE FINANCIAL AGENT EVALUATION DEMO " + "🚀")
        print("="*80)

        start_time = time.time()

        try:
            # Step 1: Setup dataset
            dataset_id = self.setup_dataset()

            # Step 2: Run evaluation experiment
            results = self.run_evaluation_experiment()

            # Step 3: Analyze results
            df = self.analyze_results(results['results'])

            # Step 4: Generate insights
            insights = self.identify_insights(df)

            # Step 5: Create comprehensive report
            report = self.generate_report(results, df, insights)

            # Step 6: Display final summary
            elapsed_time = time.time() - start_time

            print("\n" + "🎉" + " DEMO COMPLETED SUCCESSFULLY! " + "🎉")
            print("="*60)
            print(f"⏱️ Total Runtime: {elapsed_time:.1f} seconds")
            print(f"🔗 View Full Results: {results['experiment_url']}")
            print(f"📊 Examples Evaluated: {len(FINANCIAL_EVALUATION_DATASET)}")
            print(f"⚖️ Evaluation Metrics: {len(self.evaluators)}")

            print("\n📋 EXECUTIVE SUMMARY:")
            for insight in insights[:5]:  # Top 5 insights
                print(f"  • {insight}")

            return {
                "success": True,
                "experiment_url": results['experiment_url'],
                "dataset_id": dataset_id,
                "insights": insights,
                "report": report,
                "runtime": elapsed_time,
                "dataframe": df
            }

        except Exception as e:
            print(f"\n❌ Demo failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "runtime": time.time() - start_time
            }

def quick_agent_test():
    """Quick test of the financial agent before full evaluation."""
    print("\n🧪 QUICK AGENT TEST")
    print("="*40)

    test_query = "What is Apple's current stock price and how has it performed this year?"
    print(f"Query: {test_query}")

    try:
        result = run_financial_agent({"question": test_query})
        print(f"\n✅ Agent Response:")
        print(f"  {result['response'][:200]}...")
        print(f"\n🛠️ Tools Used: {result.get('unique_tools_used', [])}")
        print(f"🔗 Total Tool Calls: {result.get('total_tool_calls', 0)}")
        return True
    except Exception as e:
        print(f"❌ Agent test failed: {e}")
        return False

if __name__ == "__main__":
    """
    🎯 Main execution for the LangSmith Financial Agent Evaluation Demo

    This script showcases:
    1. Advanced financial agent with multiple tools
    2. Comprehensive custom evaluators (LLM-as-judge)
    3. Realistic financial scenarios
    4. Trajectory analysis and tool usage optimization
    5. Production-ready evaluation framework
    """

    print("🎯 LangChain Interview Demo: Financial Agent Evaluation with LangSmith")
    print("="*80)

    # Quick agent test first
    if not quick_agent_test():
        print("❌ Agent test failed. Please check configuration.")
        exit(1)

    # Run full demo
    demo = FinancialAgentEvaluationDemo()
    results = demo.run_full_demo()

    if results["success"]:
        print(f"\n🎉 Demo completed successfully!")
        print(f"🔗 Share this URL with your interviewer: {results['experiment_url']}")

        # Save report to file
        with open(f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md", "w") as f:
            f.write(results["report"])
        print(f"📄 Report saved to evaluation_report_*.md")

    else:
        print(f"❌ Demo failed: {results['error']}")
        print("🔧 Check your API keys and configuration in config.py")

    print("\n" + "="*80)
    print("🚀 Ready for your LangChain interview presentation!")
    print("="*80)