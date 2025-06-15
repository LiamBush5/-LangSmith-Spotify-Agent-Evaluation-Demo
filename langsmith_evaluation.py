"""
LangSmith Evaluation Experiment for Spotify Music Agent - TSE Assessment
======================================================================

This demonstrates comprehensive evaluation using:
1. LangSmith SDK with OpenEvals for programmatic evaluation
2. Dataset creation for LangSmith UI evaluation
3. Multiple evaluation metrics including LLM-as-judge
4. Realistic developer workflow for debugging and improvement

Following modern LangSmith patterns with OpenEvals integration.
"""

import os
from datetime import datetime
from typing import Dict, Any, List
from langsmith import Client
from langsmith import wrappers
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT, CONCISENESS_PROMPT, RAG_HELPFULNESS_PROMPT

# Import our Spotify agent
from concierge.music_agent import run_spotify_agent_with_project_routing

# Initialize LangSmith client
client = Client()

# Configuration
DATASET_NAME = "spotify-agent-tse-evaluation"
EXPERIMENT_PREFIX = "spotify-agent-tse-experiment"

class SpotifyAgentEvaluationExperiment:
    """
    Comprehensive evaluation experiment for Spotify music agent
    """

    def __init__(self):
        self.client = client
        self.dataset_id = None

    def create_realistic_dataset(self) -> str:
        """
        Create a realistic evaluation dataset that a developer would actually use
        """
        print("📊 Creating Evaluation Dataset")
        print("=" * 40)

        # Real-world test cases covering different scenarios
        test_cases = [
            # Basic artist search - should return songs efficiently
            {
                "inputs": {"query": "Find me Taylor Swift's most popular songs"},
                "outputs": {
                    "expected_behavior": "Should return 5+ Taylor Swift songs with brief DJ commentary",
                    "success_criteria": "Songs returned, artist matches, response is brief"
                },
                "metadata": {
                    "category": "basic_search",
                    "difficulty": "easy",
                    "expected_tools": ["get_artist_top_songs"],
                    "max_tool_calls": 2
                }
            },

            # Genre discovery - test recommendation capabilities
            {
                "inputs": {"query": "I want to discover some indie rock bands"},
                "outputs": {
                    "expected_behavior": "Should return indie rock songs with artist variety",
                    "success_criteria": "Genre-appropriate songs, multiple artists, discovery focus"
                },
                "metadata": {
                    "category": "genre_discovery",
                    "difficulty": "medium",
                    "expected_tools": ["get_genre_songs"],
                    "max_tool_calls": 2
                }
            },

            # Mood-based request - contextual understanding
            {
                "inputs": {"query": "Give me energetic workout music"},
                "outputs": {
                    "expected_behavior": "Should return high-energy songs suitable for workouts",
                    "success_criteria": "Energetic songs, workout-appropriate, good variety"
                },
                "metadata": {
                    "category": "mood_based",
                    "difficulty": "medium",
                    "expected_tools": ["get_genre_songs", "search_tracks"],
                    "max_tool_calls": 2
                }
            },

            # Complex playlist creation - multi-step reasoning
            {
                "inputs": {"query": "Create a chill playlist with artists like Billie Eilish"},
                "outputs": {
                    "expected_behavior": "Should create playlist with Billie Eilish and similar artists",
                    "success_criteria": "Playlist created, similar artists included, 10+ songs"
                },
                "metadata": {
                    "category": "playlist_creation",
                    "difficulty": "hard",
                    "expected_tools": ["create_smart_playlist", "get_similar_songs"],
                    "max_tool_calls": 3
                }
            },

            # Multi-constraint query - complex reasoning
            {
                "inputs": {"query": "Find me upbeat pop songs similar to Dua Lipa but not too mainstream"},
                "outputs": {
                    "expected_behavior": "Should balance multiple constraints: upbeat, pop, Dua Lipa-like, less mainstream",
                    "success_criteria": "Meets all constraints, good song selection, appropriate artists"
                },
                "metadata": {
                    "category": "complex_query",
                    "difficulty": "hard",
                    "expected_tools": ["get_similar_songs", "search_tracks"],
                    "max_tool_calls": 3
                }
            },

            # Event search - external data integration
            {
                "inputs": {"query": "Who's performing in New York this weekend?"},
                "outputs": {
                    "expected_behavior": "Should search for current concert information in NYC",
                    "success_criteria": "Current event data, location-specific, time-relevant"
                },
                "metadata": {
                    "category": "event_search",
                    "difficulty": "medium",
                    "expected_tools": ["tavily_search_results_json"],
                    "max_tool_calls": 2
                }
            },

            # Edge case - vague query handling
            {
                "inputs": {"query": "Play some music"},
                "outputs": {
                    "expected_behavior": "Should handle vague request gracefully, ask for clarification or provide general recommendations",
                    "success_criteria": "Graceful handling, helpful response, no errors"
                },
                "metadata": {
                    "category": "edge_case",
                    "difficulty": "easy",
                    "expected_tools": ["get_genre_songs"],
                    "max_tool_calls": 1
                }
            },

            # Error handling - non-existent artist
            {
                "inputs": {"query": "Find songs by XYZNonExistentArtist123"},
                "outputs": {
                    "expected_behavior": "Should handle gracefully and suggest alternatives",
                    "success_criteria": "No crashes, helpful error handling, alternative suggestions"
                },
                "metadata": {
                    "category": "error_handling",
                    "difficulty": "medium",
                    "expected_tools": ["search_tracks"],
                    "max_tool_calls": 2
                }
            },

            # Efficiency test - simple request
            {
                "inputs": {"query": "Show me The Weeknd's hits"},
                "outputs": {
                    "expected_behavior": "Should efficiently return The Weeknd's popular songs",
                    "success_criteria": "Correct artist, popular songs, minimal tool usage"
                },
                "metadata": {
                    "category": "efficiency_test",
                    "difficulty": "easy",
                    "expected_tools": ["get_artist_top_songs"],
                    "max_tool_calls": 1
                }
            },

            # Style consistency test
            {
                "inputs": {"query": "Recommend some jazz for a dinner party"},
                "outputs": {
                    "expected_behavior": "Should provide jazz recommendations with appropriate context",
                    "success_criteria": "Jazz genre, dinner party appropriate, good selection"
                },
                "metadata": {
                    "category": "style_consistency",
                    "difficulty": "medium",
                    "expected_tools": ["get_genre_songs"],
                    "max_tool_calls": 2
                }
            }
        ]

        try:
            # Create or get existing dataset
            existing_datasets = list(self.client.list_datasets(dataset_name=DATASET_NAME))
            if existing_datasets:
                dataset = existing_datasets[0]
                print(f"✅ Using existing dataset: {dataset.id}")
            else:
                dataset = self.client.create_dataset(
                    dataset_name=DATASET_NAME,
                    description="Comprehensive evaluation dataset for Spotify music agent - TSE evaluation experiment"
                )
                print(f"✅ Created new dataset: {dataset.id}")

            # Add examples to dataset
            examples = []
            for i, case in enumerate(test_cases):
                example = self.client.create_example(
                    dataset_id=dataset.id,
                    inputs=case["inputs"],
                    outputs=case["outputs"],
                    metadata=case["metadata"]
                )
                examples.append(example)

            print(f"✅ Added {len(examples)} test cases to dataset")
            self.dataset_id = dataset.id
            return dataset.id

        except Exception as e:
            print(f"❌ Dataset creation failed: {e}")
            raise

    def create_custom_evaluators(self):
        """
        Create custom evaluators using OpenEvals for music-specific evaluation
        """

        # 1. Song Relevance Evaluator - checks if returned songs match the query
        def song_relevance_evaluator(inputs: dict, outputs: dict, reference_outputs: dict = None):
            """Evaluate if returned songs are relevant to the music query"""

            # Custom prompt for music relevance
            MUSIC_RELEVANCE_PROMPT = """
You are evaluating whether a music agent's response is relevant to the user's music query.

USER QUERY: {inputs}
AGENT RESPONSE: {outputs}

Evaluate based on:
1. Are the returned songs relevant to the query?
2. Do the artists/genres match what was requested?
3. Is the response helpful for the user's music needs?

Rate from 0.0 to 1.0 where:
- 1.0 = Perfect relevance, exactly what was requested
- 0.7-0.9 = Good relevance, mostly appropriate
- 0.4-0.6 = Partial relevance, some issues
- 0.0-0.3 = Poor relevance, doesn't match request

Provide your score and brief explanation.
"""

            evaluator = create_llm_as_judge(
                prompt=MUSIC_RELEVANCE_PROMPT,
                model="openai:gpt-4o-mini",
                feedback_key="song_relevance",
            )

            return evaluator(
                inputs=inputs,
                outputs=outputs,
                reference_outputs=reference_outputs
            )

        # 2. DJ Style Evaluator - checks for brief, DJ-like responses
        def dj_style_evaluator(inputs: dict, outputs: dict, reference_outputs: dict = None):
            """Evaluate if response follows brief DJ style"""

            DJ_STYLE_PROMPT = """
You are evaluating whether a music agent's response follows a brief DJ style.

USER QUERY: {inputs}
AGENT RESPONSE: {outputs}

A good DJ-style response should be:
1. Brief (1-2 sentences max)
2. Cool and knowledgeable about music
3. Focused on the vibe, not track-by-track details
4. Uses music terminology naturally

Rate from 0.0 to 1.0 where:
- 1.0 = Perfect DJ style, brief and cool
- 0.7-0.9 = Good style, mostly appropriate
- 0.4-0.6 = Okay style, some issues
- 0.0-0.3 = Poor style, too verbose or inappropriate

Provide your score and brief explanation.
"""

            evaluator = create_llm_as_judge(
                prompt=DJ_STYLE_PROMPT,
                model="openai:gpt-4o-mini",
                feedback_key="dj_style",
            )

            return evaluator(
                inputs=inputs,
                outputs=outputs,
                reference_outputs=reference_outputs
            )

        # 3. Tool Efficiency Evaluator - checks if agent used tools efficiently
        def tool_efficiency_evaluator(inputs: dict, outputs: dict, reference_outputs: dict = None):
            """Evaluate tool usage efficiency"""

            # Extract tool usage from outputs
            tool_calls = outputs.get("total_tool_calls", 0) if isinstance(outputs, dict) else 0

            # Simple efficiency scoring
            if tool_calls == 0:
                score = 0.0
                comment = "No tools used"
            elif tool_calls <= 2:
                score = 1.0
                comment = f"Efficient tool usage: {tool_calls} calls"
            elif tool_calls <= 4:
                score = 0.7
                comment = f"Moderate tool usage: {tool_calls} calls"
            else:
                score = 0.3
                comment = f"Inefficient tool usage: {tool_calls} calls"

            return {
                "key": "tool_efficiency",
                "score": score,
                "comment": comment
            }

        return [
            song_relevance_evaluator,
            dj_style_evaluator,
            tool_efficiency_evaluator,
            # Add OpenEvals prebuilt evaluators
            lambda inputs, outputs, reference_outputs=None: create_llm_as_judge(
                prompt=RAG_HELPFULNESS_PROMPT,
                model="openai:gpt-4o-mini",
                feedback_key="helpfulness"
            )(inputs=inputs, outputs=outputs, reference_outputs=reference_outputs),

            lambda inputs, outputs, reference_outputs=None: create_llm_as_judge(
                prompt=CONCISENESS_PROMPT,
                model="openai:gpt-4o-mini",
                feedback_key="conciseness"
            )(inputs=inputs, outputs=outputs, reference_outputs=reference_outputs)
        ]

    def run_sdk_evaluation(self) -> Dict[str, Any]:
        """
        Run evaluation using LangSmith SDK with OpenEvals
        """
        print("\n🔬 Running SDK Evaluation with OpenEvals")
        print("=" * 50)

        if not self.dataset_id:
            raise ValueError("Dataset not created. Run create_realistic_dataset() first.")

        # Create evaluators
        evaluators = self.create_custom_evaluators()

        # Define target function (what we're evaluating)
        def target_function(inputs: dict) -> dict:
            """Target function that calls our Spotify agent"""
            query = inputs.get("query", "")

            # Call the Spotify agent
            result = run_spotify_agent_with_project_routing({"input": query})

            # Return in format expected by evaluators
            return {
                "response": result.get("response", ""),
                "songs": result.get("songs", []),
                "total_tool_calls": result.get("total_tool_calls", 0),
                "tools_used": result.get("unique_tools_used", [])
            }

        try:
            # Run the evaluation
            experiment_results = self.client.evaluate(
                target_function,
                data=self.dataset_id,
                evaluators=evaluators,
                experiment_prefix=EXPERIMENT_PREFIX,
                max_concurrency=2,  # Be gentle on the API
                metadata={
                    "evaluation_type": "sdk_with_openevals",
                    "agent_version": "v2.1",
                    "evaluator": "tse_candidate",
                    "timestamp": datetime.now().isoformat(),
                    "description": "Comprehensive evaluation using OpenEvals and custom evaluators"
                }
            )

            print(f"✅ SDK Evaluation completed!")
            print(f"🔗 View results: {experiment_results}")

            return {
                "experiment_results": experiment_results,
                "status": "completed",
                "dataset_id": self.dataset_id
            }

        except Exception as e:
            print(f"❌ SDK Evaluation failed: {e}")
            return {
                "experiment_results": None,
                "status": "failed",
                "error": str(e),
                "dataset_id": self.dataset_id
            }

    def generate_ui_evaluation_guide(self) -> str:
        """
        Generate comprehensive guide for running evaluation in LangSmith UI
        """
        guide = f"""
🎯 LangSmith UI Evaluation Guide - TSE Assessment
===============================================

Dataset: {DATASET_NAME}
Dataset ID: {self.dataset_id}

STEPS TO RUN UI EVALUATION:

1. 🌐 Open LangSmith UI: https://smith.langchain.com

2. 📊 Navigate to Datasets:
   - Go to "Datasets" in the sidebar
   - Find "{DATASET_NAME}"
   - Click to view the 10 test cases

3. 🚀 Run Evaluation:
   - Click "Run Evaluation" button
   - Configure target:
     * If you have an API endpoint: Enter your Spotify agent URL
     * If testing locally: Use the SDK approach instead

4. 📋 Select Evaluators:
   Built-in options to try:
   - ✅ Correctness (compares against expected outputs)
   - ✅ Helpfulness (judges if response helps the user)
   - ✅ Conciseness (evaluates response brevity)
   - ✅ Relevance (checks if response addresses the query)

5. ⚙️ Configure Settings:
   - Max concurrency: 2 (to be gentle on APIs)
   - Add experiment name: "ui-evaluation-{datetime.now().strftime('%Y%m%d')}"

6. 🔍 Analyze Results:
   Look for patterns in:
   - Which query types perform best/worst
   - Tool efficiency across different scenarios
   - Response style consistency
   - Error handling effectiveness

WHAT TO COMPARE (SDK vs UI):
=============================
- Evaluator agreement: Do custom and built-in evaluators align?
- Performance patterns: Which approach reveals more insights?
- Usability: Which is easier for iterative development?
- Debugging: Which provides better failure analysis?

KEY INSIGHTS TO GATHER:
======================
✅ Tool Efficiency: Are simple queries using 1 tool, complex ones ≤3?
✅ Response Style: Are responses brief and DJ-like?
✅ Error Handling: How does the agent handle edge cases?
✅ Music Relevance: Do song recommendations match user intent?
✅ Performance Patterns: Which query types are most/least reliable?

FRICTION LOG OPPORTUNITIES:
==========================
- Note any confusing UI elements
- Document unclear evaluation metrics
- Track any unexpected results
- Identify missing evaluation capabilities
"""

        return guide

    def run_complete_experiment(self):
        """
        Run the complete evaluation experiment demonstrating both approaches
        """
        print("🎵 Spotify Agent LangSmith Evaluation - TSE Assessment")
        print("=" * 65)
        print("Demonstrating modern evaluation workflow with OpenEvals integration")
        print()

        # Step 1: Create dataset
        try:
            dataset_id = self.create_realistic_dataset()
            print(f"📊 Dataset ready: {dataset_id}")
        except Exception as e:
            print(f"❌ Dataset creation failed: {e}")
            return

        # Step 2: Run SDK evaluation
        print("\n" + "="*50)
        sdk_results = self.run_sdk_evaluation()

        # Step 3: Generate UI guide
        print("\n🖥️  UI Evaluation Guide")
        print("=" * 30)
        ui_guide = self.generate_ui_evaluation_guide()
        print(ui_guide)

        # Step 4: Summary and insights
        print("\n📋 Experiment Summary")
        print("=" * 25)
        print(f"✅ Dataset: {DATASET_NAME} ({dataset_id})")
        print(f"✅ SDK Status: {sdk_results['status']}")
        print(f"✅ Test Cases: 10 realistic scenarios")
        print(f"✅ Evaluators: 5 (3 custom + 2 OpenEvals)")

        print("\n🎯 Key Evaluation Insights:")
        print("- Song relevance and music domain accuracy")
        print("- DJ-style response consistency")
        print("- Tool usage efficiency patterns")
        print("- Error handling robustness")
        print("- Performance across query complexity")

        print("\n🔍 What This Experiment Tests:")
        print("- Basic functionality (artist search, genre discovery)")
        print("- Complex reasoning (multi-constraint queries)")
        print("- Edge case handling (vague/invalid queries)")
        print("- Efficiency (tool usage optimization)")
        print("- Style consistency (brief DJ responses)")

        print("\n📊 Next Steps for Analysis:")
        print("1. Compare SDK vs UI evaluation results")
        print("2. Identify failure patterns across test categories")
        print("3. Analyze tool efficiency vs query complexity")
        print("4. Document friction points for engineering team")
        print("5. Iterate on agent prompts based on findings")

        return {
            "dataset_id": dataset_id,
            "sdk_results": sdk_results,
            "ui_guide": ui_guide,
            "status": "experiment_complete"
        }

def main():
    """
    Main entry point for the TSE evaluation experiment
    """
    experiment = SpotifyAgentEvaluationExperiment()
    results = experiment.run_complete_experiment()

    print(f"\n🎉 Evaluation Experiment Complete!")
    print(f"Dataset ID: {results['dataset_id']}")
    print(f"SDK Status: {results['sdk_results']['status']}")

    if results['sdk_results']['status'] == 'completed':
        print("🔗 Check LangSmith UI for detailed results and comparisons")

    return results

if __name__ == "__main__":
    main()