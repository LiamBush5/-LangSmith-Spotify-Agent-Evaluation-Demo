"""
Financial Agent Evaluation with LangSmith

Professional evaluation script for the financial agent using LangSmith's evaluation framework.
Includes custom LLM-as-judge evaluators for comprehensive assessment.

Usage:
    python run_evaluation.py                    # Run 3 examples (cost control)
    python run_evaluation.py --max-examples 5   # Run 5 examples
    python run_evaluation.py --all              # Run all examples
"""
import pandas as pd
from typing import Dict, Any
import time
import argparse
from datetime import datetime

# Import config first to set environment variables
import config

# Import LangSmith components
from langsmith import Client
from langsmith.evaluation import evaluate

from financial_agent import run_financial_agent_with_project_routing
from evaluation_dataset import create_langsmith_dataset, FINANCIAL_EVALUATION_DATASET
from custom_evaluations import FINANCIAL_EVALUATORS

# Initialize LangSmith client
client = Client()

def setup_evaluation_dataset(dataset_name: str, max_examples: int = None) -> str:
    """Create and populate the evaluation dataset."""
    print(f"\n📊 Setting up evaluation dataset: {dataset_name}")

    dataset_id = create_langsmith_dataset(dataset_name, max_examples=max_examples)
    examples_count = max_examples if max_examples else len(FINANCIAL_EVALUATION_DATASET)

    print(f"✅ Dataset ready with {examples_count} examples")
    return dataset_id

def run_financial_evaluation(dataset_name: str, max_examples: int = None) -> Dict[str, Any]:
    """Run the financial agent evaluation experiment."""
    print(f"\n🚀 Running evaluation experiment")
    print(f"Target: {run_financial_agent_with_project_routing.__name__}")
    print(f"Evaluators: {len(FINANCIAL_EVALUATORS)}")

    # Prepare data source
    data_source = dataset_name
    if max_examples:
        print(f"💰 Cost control: Sampling {max_examples} examples")
        dataset = client.read_dataset(dataset_name=dataset_name)
        all_examples = list(client.list_examples(dataset_id=dataset.id))

        import random
        random.seed(42)  # Reproducible sampling
        data_source = random.sample(all_examples, min(max_examples, len(all_examples)))

    # Create unique experiment name with timestamp
    experiment_prefix = f"{config.EXPERIMENT_PREFIX}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # Run evaluation
    results = evaluate(
        run_financial_agent_with_project_routing,
        data=data_source,
        evaluators=FINANCIAL_EVALUATORS,
        experiment_prefix=experiment_prefix,
        max_concurrency=config.MAX_CONCURRENCY,
        metadata={
            "evaluation_type": "financial_agent",
            "max_examples": max_examples or "all",
            "agent_version": "v2.1",
            "dataset": dataset_name,
            "timestamp": datetime.now().isoformat()
        }
    )

    print(f"✅ Evaluation completed!")

    # Get experiment info
    experiment_name = getattr(results, 'experiment_name', 'experiment-completed')
    experiment_url = f"https://smith.langchain.com/experiments/{experiment_name}"

    return {
        "results": results,
        "experiment_name": experiment_name,
        "experiment_url": experiment_url
    }

def analyze_evaluation_results(results) -> pd.DataFrame:
    """Analyze evaluation results and display summary."""
    print(f"\n📈 Analyzing results...")

    try:
        df = results.to_pandas()

        # Calculate evaluator performance
        print(f"\n📊 EVALUATION SUMMARY:")
        print("-" * 50)

        for evaluator in FINANCIAL_EVALUATORS:
            eval_name = evaluator.__name__
            score_cols = [col for col in df.columns if eval_name in col and 'score' in col]

            if score_cols:
                scores = df[score_cols[0]].dropna()
                if len(scores) > 0:
                    avg_score = scores.mean()
                    pass_rate = (scores >= 0.7).mean() * 100
                    print(f"  {eval_name:25} | Avg: {avg_score:.3f} | Pass: {pass_rate:.1f}%")

        # Category performance
        if 'outputs.category' in df.columns:
            print(f"\n📂 PERFORMANCE BY CATEGORY:")
            print("-" * 50)

            category_scores = df.groupby('outputs.category').agg({
                col: 'mean' for col in df.columns
                if 'score' in col and any(e.__name__ in col for e in FINANCIAL_EVALUATORS)
            }).round(3)

            for category, scores in category_scores.iterrows():
                avg_score = scores.mean() if not scores.empty else 0
                print(f"  {category:25} | Avg: {avg_score:.3f}")

        return df

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return pd.DataFrame()

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Financial Agent Evaluation')
    parser.add_argument('--max-examples', type=int, default=3,
                        help='Maximum examples to evaluate (default: 3)')
    parser.add_argument('--all', action='store_true',
                        help='Evaluate all examples')
    args = parser.parse_args()

    # Determine number of examples
    if args.all:
        max_examples = None
        print("🚀 Running full evaluation (all examples)")
    else:
        max_examples = args.max_examples
        print(f"💰 Cost control: Running {max_examples} examples")

    print("="*80)
    print("FINANCIAL AGENT EVALUATION")
    print("="*80)

    start_time = time.time()
    # Use persistent dataset, create experiments with timestamps
    dataset_name = "Financial-Agent-Evaluation-Dataset"

    try:
        # Setup dataset
        dataset_id = setup_evaluation_dataset(dataset_name, max_examples)

        # Run evaluation
        eval_results = run_financial_evaluation(dataset_name, max_examples)

        # Analyze results
        df = analyze_evaluation_results(eval_results['results'])

        # Summary
        runtime = time.time() - start_time
        examples_evaluated = max_examples if max_examples else len(FINANCIAL_EVALUATION_DATASET)

        print(f"\n🎉 EVALUATION COMPLETE")
        print("="*50)
        print(f"Runtime: {runtime:.1f}s")
        print(f"Examples: {examples_evaluated}")
        print(f"Evaluators: {len(FINANCIAL_EVALUATORS)}")
        print(f"Results: {eval_results['experiment_url']}")

        return {
            "success": True,
            "experiment_url": eval_results['experiment_url'],
            "runtime": runtime
        }

    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    main()