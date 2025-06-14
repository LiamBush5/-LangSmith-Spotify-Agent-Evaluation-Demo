"""
Music Agent Evaluations - LangSmith Evaluation Framework
"""

import os
from datetime import datetime
from dotenv import load_dotenv

# LangSmith imports
from langsmith import Client, traceable
from langsmith.evaluation import evaluate
from langsmith.schemas import Run, Example

# Load environment variables
load_dotenv()

# Configure LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "music-concierge-demo"

class MusicAgentEvaluator:
    """Evaluator for the music agent using LangSmith"""

    def __init__(self, agent):
        self.agent = agent
        self.client = Client()

    def create_evaluation_dataset(self) -> str:
        """Create a dataset for evaluation"""
        dataset_name = f"music-agent-eval-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        # Sample evaluation examples
        examples = [
            {
                "input": "I love energetic pop music. Can you recommend some upbeat songs?",
                "expected_output": "Should recommend energetic pop songs and save genre preference"
            },
            {
                "input": "Find me songs similar to Taylor Swift",
                "expected_output": "Should use get_similar_music or get_artist_top_songs for Taylor Swift"
            },
            {
                "input": "I'm feeling sad today, suggest some music",
                "expected_output": "Should recommend mood-appropriate music and save mood preference"
            },
            {
                "input": "Search for jazz piano music",
                "expected_output": "Should search for jazz piano and potentially save genre preference"
            },
            {
                "input": "Create a workout playlist",
                "expected_output": "Should recommend high-energy songs suitable for workouts"
            }
        ]

        # Create dataset in LangSmith
        dataset = self.client.create_dataset(
            dataset_name=dataset_name,
            description="Evaluation dataset for music concierge agent"
        )

        # Add examples to dataset
        for example in examples:
            self.client.create_example(
                dataset_id=dataset.id,
                inputs={"input": example["input"]},
                outputs={"expected": example["expected_output"]}
            )

        return dataset_name

    @traceable
    def evaluate_response(self, run: Run, example: Example) -> dict:
        """Custom evaluator for agent responses"""
        if not run.outputs:
            return {"score": 0, "reasoning": "No output generated"}

        response = run.outputs.get("output", "")
        expected = example.outputs.get("expected", "")

        # Simple evaluation criteria
        score = 0
        reasoning_parts = []

        # Check if response is not empty
        if response and len(response) > 10:
            score += 0.3
            reasoning_parts.append("Response generated")

        # Check if response contains music-related content
        music_keywords = ["song", "music", "artist", "album", "spotify", "recommend", "playlist"]
        if any(keyword in response.lower() for keyword in music_keywords):
            score += 0.4
            reasoning_parts.append("Contains music-related content")

        # Check if response is engaging (has emojis or enthusiasm)
        if any(char in response for char in "🎵🎶🎤🎧🎸🎹🥁") or "!" in response:
            score += 0.3
            reasoning_parts.append("Engaging and enthusiastic")

        return {
            "score": score,
            "reasoning": "; ".join(reasoning_parts) if reasoning_parts else "Low quality response"
        }

    def run_evaluation(self, dataset_name: str) -> dict:
        """Run evaluation on the dataset"""
        print(f"🧪 Running evaluation on dataset: {dataset_name}")

        # Run evaluation
        results = evaluate(
            lambda inputs: {"output": self.agent.chat(inputs["input"])},
            data=dataset_name,
            evaluators=[self.evaluate_response],
            experiment_prefix="music-agent"
        )

        return results

def main():
    """Demo evaluation function"""
    print("🧪 Music Agent Evaluator - LangSmith Demo")
    print("=" * 50)

    # Check for OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Please set OPENAI_API_KEY environment variable")
        return

    # Import and initialize agent
    from agent import MusicConciergeAgent
    agent = MusicConciergeAgent(api_key)

    # Initialize evaluator
    evaluator = MusicAgentEvaluator(agent)

    print("✅ Evaluator initialized with LangSmith tracing enabled")
    print("🔗 View results at: https://smith.langchain.com/")

    # Run evaluation
    print("\n🧪 Running LangSmith evaluation...")
    dataset_name = evaluator.create_evaluation_dataset()
    results = evaluator.run_evaluation(dataset_name)
    print(f"✅ Evaluation complete! Dataset: {dataset_name}")
    print(f"📊 Results: {results}")

if __name__ == "__main__":
    main()