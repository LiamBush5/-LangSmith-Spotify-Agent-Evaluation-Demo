import os
import sys
from datetime import datetime
from typing import Dict, Any
from langsmith import Client
from langsmith import wrappers

# Add parent directory to path so we can import agent module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our Spotify agent, dataset, and evaluators
from agent.music_agent import run_spotify_agent_with_project_routing
from dataset import get_evaluation_dataset, get_dataset_stats
from evaluators import get_all_evaluators

# Initialize LangSmith client
client = Client()

# Configuration
DEFAULT_DATASET_NAME = "Golden-Spotify-Agent-Dataset"
EXPERIMENT_PREFIX = "spotify-agent-experiment"

class SpotifyAgentEvaluation:
    """Production evaluation for Spotify music agent"""

    def __init__(self):
        self.client = client
        self.dataset_id = None
        self.dataset_name = None

    def _prompt_dataset_choice(self) -> str:
        """Prompt user to choose from available datasets"""
        print("\nChoose Dataset:")

        try:
            # Get all available datasets
            datasets = list(self.client.list_datasets())

            if not datasets:
                print("No datasets found. Creating default dataset...")
                return DEFAULT_DATASET_NAME

            # Filter for potentially relevant datasets (optional)
            relevant_datasets = datasets  # Show all datasets

            print("0. Create new dataset")
            for i, dataset in enumerate(relevant_datasets[:10], 1):  # Limit to 10 for readability
                created_date = dataset.created_at.strftime("%Y-%m-%d") if dataset.created_at else "Unknown"
                example_count = "Unknown"
                try:
                    examples = list(self.client.list_examples(dataset_id=dataset.id, limit=1))
                    total_examples = len(list(self.client.list_examples(dataset_id=dataset.id)))
                    example_count = str(total_examples)
                except:
                    pass

                print(f"{i}. {dataset.name} (ID: {str(dataset.id)[:8]}..., Examples: {example_count}, Created: {created_date})")

            while True:
                choice = input(f"\nEnter your choice (0-{len(relevant_datasets)}): ").strip()

                if choice == "0":
                    custom_name = input("Enter new dataset name (or press Enter for default): ").strip()
                    selected_name = custom_name if custom_name else DEFAULT_DATASET_NAME
                    print(f"Selected: Create new dataset '{selected_name}'")
                    return selected_name

                try:
                    choice_idx = int(choice) - 1
                    if 0 <= choice_idx < len(relevant_datasets):
                        selected_dataset = relevant_datasets[choice_idx]
                        print(f"Selected: {selected_dataset.name}")
                        self.dataset_id = selected_dataset.id
                        return selected_dataset.name
                except ValueError:
                    pass

                print("Invalid choice, please try again.")

        except Exception as e:
            print(f"Error retrieving datasets: {e}")
            print(f"Defaulting to: {DEFAULT_DATASET_NAME}")
            return DEFAULT_DATASET_NAME

    def create_dataset(self, dataset_name: str = None) -> str:
        """Create or update evaluation dataset"""
        if not dataset_name:
            dataset_name = DEFAULT_DATASET_NAME

        self.dataset_name = dataset_name

        print("Setting up Evaluation Dataset")
        print("=" * 40)

        # If dataset_id was already set from selection, use existing dataset
        if self.dataset_id:
            print(f"Using selected dataset: {self.dataset_id}")
            return self.dataset_id

        # Load comprehensive test cases
        test_cases = get_evaluation_dataset()
        stats = get_dataset_stats(test_cases)

        print(f"Dataset: {dataset_name}")
        print(f"Test cases: {stats['total_cases']}")
        print(f"Categories: {len(stats['categories'])}")
        print(f"Difficulties: {', '.join(stats['difficulties'].keys())}")
        print()

        try:
            # Create or get existing dataset
            existing_datasets = list(self.client.list_datasets(dataset_name=dataset_name))
            if existing_datasets:
                dataset = existing_datasets[0]
                print(f"Using existing dataset: {dataset.id}")

                # Check if dataset already has examples
                existing_examples = list(self.client.list_examples(dataset_id=dataset.id, limit=1))
                if existing_examples:
                    print(f"Dataset already contains examples, skipping example creation")
                    self.dataset_id = dataset.id
                    return dataset.id
            else:
                dataset = self.client.create_dataset(
                    dataset_name=dataset_name,
                    description=f"Evaluation dataset for Spotify music agent - {dataset_name}"
                )
                print(f"Created new dataset: {dataset.id}")

            # Add examples to dataset (only for new datasets or empty existing ones)
            examples = []
            for case in test_cases:
                # Merge expected_tools into metadata for LangSmith
                metadata = case["metadata"].copy()
                metadata["expected_tools"] = case["expected_tools"]

                example = self.client.create_example(
                    dataset_id=dataset.id,
                    inputs=case["inputs"],
                    outputs=case["outputs"],
                    metadata=metadata
                )
                examples.append(example)

            print(f"Dataset ready with {len(examples)} examples")
            self.dataset_id = dataset.id
            return dataset.id

        except Exception as e:
            print(f"Dataset setup failed: {e}")
            raise

    def run_evaluation(self, filter_metadata: dict = None, model: str = "gpt-4o-mini", split_name: str = "full") -> Dict[str, Any]:
        """Run comprehensive evaluation using LangSmith SDK"""
        print("\nRunning Evaluation")
        print("=" * 30)

        if not self.dataset_id:
            raise ValueError("Dataset not created. Run create_dataset() first.")

        # Get evaluators
        evaluators = get_all_evaluators()
        print(f"Using {len(evaluators)} evaluators")
        print(f"Model: {model}")
        print(f"Split: {split_name}")

        # Define target function with model selection
        def target_function(inputs: dict) -> dict:
            """Target function that calls our Spotify agent with specified model"""
            query = inputs.get("query", "")
            # TODO: Add model parameter to your agent function when available
            result = run_spotify_agent_with_project_routing({"input": query})

            return {
                "response": result.get("response", ""),
                "songs": result.get("songs", []),
                "total_tool_calls": result.get("total_tool_calls", 0),
                "tools_used": result.get("unique_tools_used", [])
            }

        try:
            # Determine data source (full dataset or filtered)
            if filter_metadata:
                data_source = self.client.list_examples(
                    dataset_id=self.dataset_id,
                    metadata=filter_metadata
                )
                filter_desc = f" (filtered: {filter_metadata})"
            elif split_name == "smoke-test":
                # For smoke test, get all examples and sample 3
                all_examples = list(self.client.list_examples(dataset_id=self.dataset_id))
                data_source = all_examples[:3]
                filter_desc = f" (smoke test: 3 examples)"
                print(f"Smoke test: Selected {len(data_source)} examples from {len(all_examples)} total")
            else:
                data_source = self.dataset_id
                filter_desc = ""

            print(f"Running on: {self.dataset_name}{filter_desc}")

            # Create experiment name with format: model-split-timestamp
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            experiment_name = f"{model}-{split_name}-{timestamp}"

            print(f"Experiment: {experiment_name}")

            # Run evaluation
            experiment_results = self.client.evaluate(
                target_function,
                data=data_source,
                evaluators=evaluators,
                experiment_prefix=experiment_name,
                max_concurrency=2,
                metadata={
                    "evaluation_type": "production_evaluation",
                    "model": model,
                    "split": split_name,
                    "agent_version": "v2.1",
                    "timestamp": datetime.now().isoformat(),
                    "evaluator_count": len(evaluators),
                    "filter_metadata": filter_metadata or "none"
                }
            )

            print(f"Evaluation completed!")
            print(f"View results: {experiment_results}")

            return {
                "experiment_results": experiment_results,
                "experiment_name": experiment_name,
                "status": "completed",
                "dataset_id": self.dataset_id,
                "model": model,
                "split": split_name,
                "filter_applied": filter_metadata
            }

        except Exception as e:
            print(f"Evaluation failed: {e}")
            return {
                "experiment_results": None,
                "status": "failed",
                "error": str(e),
                "dataset_id": self.dataset_id,
                "model": model,
                "split": split_name
            }

    def run_complete_evaluation(self) -> Dict[str, Any]:
        """Run complete evaluation workflow with interactive choices"""
        print("Spotify Agent - Production Evaluation")
        print("=" * 50)

        # Step 1: Setup dataset
        try:
            dataset_name = self._prompt_dataset_choice()
            self.create_dataset(dataset_name)
        except Exception as e:
            print(f"Setup failed: {e}")
            return {"status": "setup_failed", "error": str(e)}

        # Step 2: Interactive configuration
        model_choice = self._prompt_model_choice()
        split_choices = self._prompt_split_choices()

        # Step 3: Run evaluations based on choices
        all_results = []

        for split_config in split_choices:
            print(f"\nRunning evaluation: {split_config['name']}")
            print("=" * 60)

            results = self.run_evaluation(
                filter_metadata=split_config.get('filter'),
                model=model_choice,
                split_name=split_config['name']
            )

            if results["status"] == "completed":
                all_results.append(results)
                print(f"{split_config['name']} evaluation completed!")
            else:
                print(f"{split_config['name']} evaluation failed!")

        # Step 4: Summary
        if all_results:
            self._print_summary(all_results)
            return {"status": "completed", "results": all_results}
        else:
            return {"status": "failed", "error": "No evaluations completed successfully"}

    def _prompt_model_choice(self) -> str:
        """Prompt user to choose model"""
        print("\nChoose Model:")
        print("1. GPT-4o (OpenAI)")
        print("2. GPT-4o-mini (OpenAI)")
        print("3. Gemini-1.5-Pro (Google)")
        print("4. Gemini-1.5-Flash (Google)")
        print("5. Claude-3.5-Sonnet (Anthropic)")

        while True:
            choice = input("\nEnter your choice (1-5): ").strip()

            model_map = {
                "1": "gpt-4o",
                "2": "gpt-4o-mini",
                "3": "gemini-1.5-pro",
                "4": "gemini-1.5-flash",
                "5": "claude-3.5-sonnet"
            }

            if choice in model_map:
                selected_model = model_map[choice]
                print(f"Selected: {selected_model}")
                return selected_model
            else:
                print("Invalid choice, please try again.")

    def _prompt_split_choices(self) -> list:
        """Prompt user to choose evaluation splits"""
        print("\nChoose Evaluation Splits:")
        print("1. All splits (complete evaluation)")
        print("2. Quick smoke test (3 examples)")
        print("3. Easy tests only (quick iteration)")
        print("4. Hard tests only (stress test)")
        print("5. Playlist tests only")
        print("6. Complex query tests only")
        print("7. Event-based tests only")
        print("8. Custom selection (multiple splits)")

        choice = input("\nEnter your choice (1-8): ").strip()

        split_configs = {
            "1": [{"name": "all-tests", "filter": None}],
            "2": [{"name": "smoke-test", "filter": None, "sample": 3}],
            "3": [{"name": "easy-tests", "filter": {"difficulty": "easy"}}],
            "4": [{"name": "hard-tests", "filter": {"difficulty": "hard"}}],
            "5": [{"name": "playlist-tests", "filter": {"category": "playlist_creation"}}],
            "6": [{"name": "complex-query-tests", "filter": {"category": "complex_query"}}],
            "7": [{"name": "event-tests", "filter": {"category": "event_search"}}]
        }

        if choice in split_configs:
            selected = split_configs[choice]
            print(f"Selected: {selected[0]['name']}")
            return selected
        elif choice == "8":
            return self._custom_split_selection()
        else:
            print("Invalid choice, defaulting to smoke test")
            return split_configs["2"]

    def _custom_split_selection(self) -> list:
        """Allow user to select multiple splits"""
        print("\nCustom Split Selection:")
        print("Select multiple options (comma-separated, e.g., 1,3,5):")
        print("1. Easy tests")
        print("2. Medium tests")
        print("3. Hard tests")
        print("4. Basic search tests")
        print("5. Genre discovery tests")
        print("6. Mood-based tests")
        print("7. Playlist creation tests")
        print("8. Complex query tests")
        print("9. Event search tests")
        print("10. Edge case tests")

        choices = input("\nEnter choices: ").strip().split(",")

        option_map = {
            "1": {"name": "easy-tests", "filter": {"difficulty": "easy"}},
            "2": {"name": "medium-tests", "filter": {"difficulty": "medium"}},
            "3": {"name": "hard-tests", "filter": {"difficulty": "hard"}},
            "4": {"name": "basic-search-tests", "filter": {"category": "basic_search"}},
            "5": {"name": "genre-discovery-tests", "filter": {"category": "genre_discovery"}},
            "6": {"name": "mood-based-tests", "filter": {"category": "mood_based"}},
            "7": {"name": "playlist-creation-tests", "filter": {"category": "playlist_creation"}},
            "8": {"name": "complex-query-tests", "filter": {"category": "complex_query"}},
            "9": {"name": "event-search-tests", "filter": {"category": "event_search"}},
            "10": {"name": "edge-case-tests", "filter": {"category": "edge_case"}}
        }

        selected_splits = []
        for choice in choices:
            choice = choice.strip()
            if choice in option_map:
                selected_splits.append(option_map[choice])

        if not selected_splits:
            print("No valid choices, defaulting to smoke test")
            return [{"name": "smoke-test", "filter": None, "sample": 3}]

        print(f"Selected {len(selected_splits)} splits: {[s['name'] for s in selected_splits]}")
        return selected_splits

    def _print_summary(self, results):
        """Print evaluation summary"""
        dataset_stats = get_dataset_stats(get_evaluation_dataset())

        print("\nEvaluation Summary")
        print("=" * 50)

        # Handle single result or multiple results
        if isinstance(results, list):
            print(f"Completed {len(results)} evaluations")
            for i, result in enumerate(results, 1):
                print(f"\n{i}. {result['experiment_name']}")
                print(f"   Model: {result['model']}")
                print(f"   Split: {result['split']}")
                print(f"   Status: {result['status']}")
        else:
            print(f"Dataset: {self.dataset_name}")
            print(f"Test Cases: {dataset_stats['total_cases']}")
            print(f"Evaluators: 7 production evaluators")
            print(f"Status: {results['status']}")
            if 'experiment_name' in results:
                print(f"Experiment: {results['experiment_name']}")

        print("\nDataset Coverage:")
        for category, count in dataset_stats['categories'].items():
            print(f"  • {category}: {count} cases")

        print(f"\nView detailed results in LangSmith UI")

def main():
    """Main entry point for evaluation"""
    evaluation = SpotifyAgentEvaluation()
    results = evaluation.run_complete_evaluation()

    if results["status"] == "completed":
        print("\nEvaluation Complete!")
        return True
    else:
        print(f"\nEvaluation Failed: {results.get('error', 'Unknown error')}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)