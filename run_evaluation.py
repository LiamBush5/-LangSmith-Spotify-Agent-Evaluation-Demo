"""
LangSmith Financial Agent Evaluation Demo
Comprehensive evaluation experiment showcasing advanced agent evaluation capabilities.

This script demonstrates:
- Multi-tool financial agent with ReAct reasoning
- Custom LLM-as-judge evaluators
- Trajectory analysis and tool usage evaluation
- Financial accuracy, reasoning, and completeness metrics
- Realistic financial scenarios with complex calculations

Usage:
    python run_evaluation.py              # Run 3 examples (default for cost control)
    python run_evaluation.py --max-examples 5    # Run 5 examples
    python run_evaluation.py --all        # Run all 10 examples
"""
import asyncio
import pandas as pd
from typing import Dict, List, Any
import time
import argparse
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

    def __init__(self, max_examples=None):
        self.client = client
        self.experiment_name = f"{config.EXPERIMENT_PREFIX}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.dataset_name = "Financial-Agent-Evaluation-Dataset"
        self.dataset_id = None
        self.evaluators = FINANCIAL_EVALUATORS
        self.max_examples = max_examples  # New parameter to limit examples

        print("Financial Agent Evaluation Demo Initialized")
        print(f"Experiment Name: {self.experiment_name}")
        print(f"Dataset: {self.dataset_name}")
        print(f"Evaluators: {[e.__name__ for e in self.evaluators]}")
        if self.max_examples:
            print(f"Max Examples: {self.max_examples} (cost control enabled)")

    def setup_dataset(self) -> str:
        """Create and populate the evaluation dataset."""
        print("\n" + "="*60)
        print("SETTING UP EVALUATION DATASET")
        print("="*60)

        try:
            self.dataset_id = create_langsmith_dataset(self.dataset_name, max_examples=self.max_examples)
            examples_count = self.max_examples if self.max_examples else len(FINANCIAL_EVALUATION_DATASET)
            print(f"Dataset ready: {self.dataset_id}")
            print(f"Examples to evaluate: {examples_count}")
            return self.dataset_id
        except Exception as e:
            print(f"Dataset setup failed: {e}")
            raise

    def run_evaluation_experiment(self) -> Dict[str, Any]:
        """Run the comprehensive evaluation experiment."""
        print("\n" + "="*60)
        print("RUNNING EVALUATION EXPERIMENT")
        print("="*60)

        print(f"Target Function: run_financial_agent")
        print(f"Dataset: {self.dataset_name}")
        print(f"Evaluators: {len(self.evaluators)} custom evaluators")
        print(f"Max Concurrency: {config.MAX_CONCURRENCY}")

        try:
            # Prepare data source for evaluation
            data_source = self.dataset_name

            # If cost control is enabled, create a sample of examples
            if self.max_examples:
                print(f"Sampling {self.max_examples} random examples from full dataset")

                # Get all examples from the dataset
                dataset = self.client.read_dataset(dataset_name=self.dataset_name)
                all_examples = list(self.client.list_examples(dataset_id=dataset.id))

                # Randomly sample the requested number of examples
                import random
                random.seed(42)  # For reproducibility
                sampled_examples = random.sample(all_examples, min(self.max_examples, len(all_examples)))

                # Use the sampled examples directly
                data_source = sampled_examples
                print(f"Selected {len(sampled_examples)} examples for evaluation")

            # Run the evaluation
            results = evaluate(
                run_financial_agent,  # target as first positional argument
                data=data_source,
                evaluators=self.evaluators,
                experiment_prefix=config.EXPERIMENT_PREFIX,
                max_concurrency=config.MAX_CONCURRENCY,
                client=self.client
            )

            print(f"\nEvaluation completed!")

            # Try to get experiment URL - handle different LangSmith versions
            experiment_url = 'URL not available'
            experiment_name = 'Experiment completed'

            # Check for various possible attribute names
            if hasattr(results, 'experiment_url'):
                experiment_url = results.experiment_url
            elif hasattr(results, 'experiment_id'):
                experiment_url = f"https://smith.langchain.com/o/{getattr(results, 'project_name', 'project')}/datasets/{getattr(results, 'experiment_id', 'experiment')}"
            elif hasattr(results, '_experiment_name'):
                experiment_name = results._experiment_name
                experiment_url = f"LangSmith project: {config.LANGSMITH_PROJECT}"

            if hasattr(results, 'experiment_name'):
                experiment_name = results.experiment_name
            elif hasattr(results, '_experiment_name'):
                experiment_name = results._experiment_name

            print(f"View results: {experiment_url}")

            return {
                "experiment_url": experiment_url,
                "experiment_name": experiment_name,
                "results": results
            }

        except Exception as e:
            print(f"Evaluation failed: {e}")
            raise

    def analyze_results(self, results) -> pd.DataFrame:
        """Analyze and summarize the evaluation results."""
        print("\n" + "="*60)
        print("ANALYZING EVALUATION RESULTS")
        print("="*60)

        try:
            # Convert results to DataFrame for analysis
            df = results.to_pandas()

            # Calculate summary statistics
            evaluator_columns = [col for col in df.columns if any(eval_name in col for eval_name in
                                [e.__name__ for e in self.evaluators])]

            print("\nEVALUATION SUMMARY:")
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
                print(f"\nPERFORMANCE BY CATEGORY:")
                print("-" * 40)
                category_performance = df.groupby('outputs.category').agg({
                    col: 'mean' for col in evaluator_columns if 'score' in col
                }).round(3)
                print(category_performance)

            # Complexity analysis
            if 'outputs.complexity' in df.columns:
                print(f"\nPERFORMANCE BY COMPLEXITY:")
                print("-" * 40)
                complexity_performance = df.groupby('outputs.complexity').agg({
                    col: 'mean' for col in evaluator_columns if 'score' in col
                }).round(3)
                print(complexity_performance)

            return df

        except Exception as e:
            print(f"Results analysis failed: {e}")
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



    def run_full_demo(self) -> Dict[str, Any]:
        """Run the complete evaluation demo from start to finish."""
        print("\n" + "STARTING COMPREHENSIVE FINANCIAL AGENT EVALUATION DEMO")
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

            # Step 5: Display final summary
            elapsed_time = time.time() - start_time

            print("\n" + "DEMO COMPLETED SUCCESSFULLY!")
            print("="*60)
            print(f"Total Runtime: {elapsed_time:.1f} seconds")
            print(f"View Full Results: {results.get('experiment_url', 'URL not available')}")
            examples_run = self.max_examples if self.max_examples else len(FINANCIAL_EVALUATION_DATASET)
            print(f"Examples Evaluated: {examples_run}")
            print(f"Evaluation Metrics: {len(self.evaluators)}")

            print("\nEXECUTIVE SUMMARY:")
            for insight in insights[:5]:  # Top 5 insights
                print(f"  • {insight}")

            return {
                "success": True,
                "experiment_url": results.get('experiment_url', 'URL not available'),
                "dataset_id": dataset_id,
                "insights": insights,
                "runtime": elapsed_time,
                "dataframe": df
            }

        except Exception as e:
            print(f"\nDemo failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "runtime": time.time() - start_time
            }



if __name__ == "__main__":
    """
    Main execution for the LangSmith Financial Agent Evaluation Demo

    This script showcases:
    1. Advanced financial agent with multiple tools
    2. Comprehensive custom evaluators (LLM-as-judge)
    3. Realistic financial scenarios
    4. Trajectory analysis and tool usage optimization
    5. Production-ready evaluation framework
    """

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Financial Agent Evaluation Demo')
    parser.add_argument('--max-examples', type=int, default=3,
                        help='Maximum number of examples to run (default: 3 for cost control)')
    parser.add_argument('--all', action='store_true',
                        help='Run all examples (ignores --max-examples)')
    args = parser.parse_args()

    # Determine how many examples to run (priority: CLI args > env var > default)
    if args.all:
        MAX_EXAMPLES = None
        print("Running ALL examples (full evaluation)")
    elif hasattr(config, 'MAX_EXAMPLES') and config.MAX_EXAMPLES is not None:
        MAX_EXAMPLES = config.MAX_EXAMPLES
        print(f"COST CONTROL (from env): Running {MAX_EXAMPLES} out of {len(FINANCIAL_EVALUATION_DATASET)} examples")
    else:
        MAX_EXAMPLES = args.max_examples
        print(f"COST CONTROL: Running {MAX_EXAMPLES} out of {len(FINANCIAL_EVALUATION_DATASET)} examples")

    print("LangChain Interview Demo: Financial Agent Evaluation with LangSmith")
    print("="*80)
    print(f"Examples to evaluate: {MAX_EXAMPLES if MAX_EXAMPLES else len(FINANCIAL_EVALUATION_DATASET)}")
    print("="*80)

    # Run demo with cost control
    demo = FinancialAgentEvaluationDemo(max_examples=MAX_EXAMPLES)
    results = demo.run_full_demo()

    if results["success"]:
        print(f"\nDemo completed successfully!")
        print(f"Share this URL with your interviewer: {results.get('experiment_url', 'URL not available')}")

    else:
        print(f"Demo failed: {results['error']}")
        print("Check your API keys and configuration in config.py")

    print("\n" + "="*80)
    print("Complete")
    print("="*80)