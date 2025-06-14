#!/usr/bin/env python3
"""
Music Concierge Agent - LangSmith Demo Script
Comprehensive demonstration of LangSmith features including:
- Tracing and observability
- Dataset creation and management
- Evaluation and scoring
- Memory and conversation tracking
"""

import os
import asyncio
import json
from datetime import datetime
from typing import List, Dict, Any

# Set up environment for LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "music-concierge-demo"

from agent import MusicConciergeAgent
from evaluations import MusicAgentEvaluator
from langsmith import Client
from langsmith.evaluation import evaluate, LangChainStringEvaluator
from langchain.evaluation import load_evaluator

class MusicConciergeDemo:
    """Comprehensive demo showcasing LangSmith capabilities"""

    def __init__(self):
        self.client = Client()
        self.agent = None
        self.evaluator = None

    def setup_agent(self):
        """Initialize the music agent"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        print("🎵 Initializing Music Concierge Agent...")
        self.agent = MusicConciergeAgent(api_key)
        self.evaluator = MusicAgentEvaluator(self.agent)
        print("✅ Agent initialized with LangSmith tracing enabled")

    def demo_basic_conversation(self):
        """Demo 1: Basic conversation with tracing"""
        print("\n" + "="*60)
        print("🎯 DEMO 1: Basic Conversation with LangSmith Tracing")
        print("="*60)

        conversations = [
            "Hi! I love pop music and I'm looking for some upbeat songs to work out to.",
            "Can you find me songs similar to Dua Lipa?",
            "I'm feeling nostalgic today. Suggest some 90s hits.",
            "Create a chill playlist for studying."
        ]

        for i, message in enumerate(conversations, 1):
            print(f"\n🎤 User Message {i}: {message}")
            response = self.agent.chat(message)
            print(f"🤖 Agent Response: {response[:200]}...")
            print(f"🔗 View trace at: https://smith.langchain.com/")

    def demo_memory_and_preferences(self):
        """Demo 2: Memory and preference tracking"""
        print("\n" + "="*60)
        print("🧠 DEMO 2: Memory and Preference Tracking")
        print("="*60)

        # Clear memory first
        self.agent.clear_memory()

        # Build up preferences over conversation
        preference_building_conversation = [
            "I really love electronic dance music, especially house and techno.",
            "My favorite artists are Calvin Harris and David Guetta.",
            "I usually listen to music when I'm working out or driving.",
            "Can you recommend something based on what I've told you?"
        ]

        for i, message in enumerate(preference_building_conversation, 1):
            print(f"\n🎤 Building Preferences {i}: {message}")
            response = self.agent.chat(message)
            print(f"🤖 Response: {response[:150]}...")

        # Show memory summary
        print(f"\n🧠 Memory Summary: {self.agent.get_memory_summary()}")

    def demo_tool_usage(self):
        """Demo 3: Tool usage and function calling"""
        print("\n" + "="*60)
        print("🔧 DEMO 3: Tool Usage and Function Calling")
        print("="*60)

        tool_demo_queries = [
            "Search for songs by The Weeknd",
            "Find me some jazz music for a dinner party",
            "Get me Taylor Swift's most popular songs",
            "I want music similar to Billie Eilish"
        ]

        for query in tool_demo_queries:
            print(f"\n🎤 Tool Demo Query: {query}")
            response = self.agent.chat(query)
            print(f"🤖 Response with tool usage: {response[:200]}...")

    def demo_dataset_creation(self):
        """Demo 4: Dataset creation for evaluation"""
        print("\n" + "="*60)
        print("📊 DEMO 4: Dataset Creation and Management")
        print("="*60)

        # Create evaluation dataset
        dataset_name = self.evaluator.create_evaluation_dataset()
        print(f"✅ Created evaluation dataset: {dataset_name}")

        # List datasets
        datasets = list(self.client.list_datasets())
        print(f"📋 Available datasets: {len(datasets)}")
        for dataset in datasets[-3:]:  # Show last 3
            print(f"  - {dataset.name} ({dataset.example_count} examples)")

        return dataset_name

    def demo_custom_evaluators(self):
        """Demo 5: Custom evaluators and scoring"""
        print("\n" + "="*60)
        print("🧪 DEMO 5: Custom Evaluators and Scoring")
        print("="*60)

        # Create a simple dataset for quick evaluation
        quick_dataset_name = f"quick-eval-{datetime.now().strftime('%H%M%S')}"

        quick_examples = [
            {
                "input": "Recommend upbeat pop songs",
                "expected": "Should provide energetic pop music recommendations"
            },
            {
                "input": "I like rock music",
                "expected": "Should acknowledge rock preference and provide recommendations"
            }
        ]

        # Create quick dataset
        dataset = self.client.create_dataset(
            dataset_name=quick_dataset_name,
            description="Quick evaluation dataset for demo"
        )

        for example in quick_examples:
            self.client.create_example(
                dataset_id=dataset.id,
                inputs={"input": example["input"]},
                outputs={"expected": example["expected"]}
            )

        print(f"📊 Created quick evaluation dataset: {quick_dataset_name}")

        # Run evaluation
        print("🧪 Running evaluation with custom scoring...")
        results = self.evaluator.run_evaluation(quick_dataset_name)
        print(f"✅ Evaluation completed!")

        return quick_dataset_name

    def demo_advanced_evaluation(self):
        """Demo 6: Advanced evaluation with multiple metrics"""
        print("\n" + "="*60)
        print("📈 DEMO 6: Advanced Evaluation with Multiple Metrics")
        print("="*60)

        # Create comprehensive evaluation
        def music_relevance_evaluator(run, example):
            """Custom evaluator for music relevance"""
            if not run.outputs:
                return {"score": 0, "reasoning": "No output"}

            response = run.outputs.get("output", "").lower()
            music_terms = ["song", "artist", "music", "album", "playlist", "genre", "spotify"]

            relevance_score = sum(1 for term in music_terms if term in response) / len(music_terms)

            return {
                "key": "music_relevance",
                "score": relevance_score,
                "reasoning": f"Found {sum(1 for term in music_terms if term in response)} music-related terms"
            }

        def response_helpfulness_evaluator(run, example):
            """Custom evaluator for response helpfulness"""
            if not run.outputs:
                return {"score": 0, "reasoning": "No output"}

            response = run.outputs.get("output", "")

            # Check for helpful elements
            helpful_indicators = [
                len(response) > 50,  # Substantial response
                "recommend" in response.lower(),  # Makes recommendations
                "http" in response,  # Includes links
                any(emoji in response for emoji in "🎵🎶🎤🎧"),  # Engaging
            ]

            helpfulness_score = sum(helpful_indicators) / len(helpful_indicators)

            return {
                "key": "helpfulness",
                "score": helpfulness_score,
                "reasoning": f"Met {sum(helpful_indicators)}/{len(helpful_indicators)} helpfulness criteria"
            }

        # Create evaluation dataset
        eval_dataset_name = f"advanced-eval-{datetime.now().strftime('%H%M%S')}"

        advanced_examples = [
            {"input": "I need workout music", "expected": "High-energy recommendations"},
            {"input": "Suggest sad songs", "expected": "Melancholic music recommendations"},
            {"input": "Find jazz for dinner", "expected": "Smooth jazz recommendations"}
        ]

        dataset = self.client.create_dataset(
            dataset_name=eval_dataset_name,
            description="Advanced evaluation with multiple metrics"
        )

        for example in advanced_examples:
            self.client.create_example(
                dataset_id=dataset.id,
                inputs={"input": example["input"]},
                outputs={"expected": example["expected"]}
            )

        print(f"📊 Running advanced evaluation on: {eval_dataset_name}")

        # Run evaluation with multiple evaluators
        results = evaluate(
            lambda inputs: {"output": self.agent.chat(inputs["input"])},
            data=eval_dataset_name,
            evaluators=[music_relevance_evaluator, response_helpfulness_evaluator],
            experiment_prefix="advanced-music-eval"
        )

        print("✅ Advanced evaluation completed with multiple metrics!")
        return eval_dataset_name

    def demo_streaming_and_async(self):
        """Demo 7: Streaming and async capabilities"""
        print("\n" + "="*60)
        print("⚡ DEMO 7: Streaming and Async Capabilities")
        print("="*60)

        # Note: This is a simplified demo since our current agent doesn't implement streaming
        # In a full implementation, you would use .astream() and .astream_events()

        print("🔄 Simulating async conversation handling...")

        async def async_chat_demo():
            queries = [
                "Quick recommendation for pop music",
                "Find me some rock songs",
                "Suggest electronic music"
            ]

            # Simulate async processing
            for i, query in enumerate(queries, 1):
                print(f"🎤 Async Query {i}: {query}")
                # In real implementation: response = await self.agent.achat(query)
                response = self.agent.chat(query)  # Sync for demo
                print(f"⚡ Async Response {i}: {response[:100]}...")

        # Run async demo
        asyncio.run(async_chat_demo())

    def demo_error_handling_and_fallbacks(self):
        """Demo 8: Error handling and fallback mechanisms"""
        print("\n" + "="*60)
        print("🛡️ DEMO 8: Error Handling and Fallback Mechanisms")
        print("="*60)

        error_test_queries = [
            "Find songs by an artist that doesn't exist: XYZ123FAKE",
            "Search for music in a genre that doesn't exist: quantum-jazz-fusion",
            "",  # Empty query
            "Play music" * 100,  # Very long query
        ]

        for i, query in enumerate(error_test_queries, 1):
            print(f"\n🧪 Error Test {i}: {query[:50]}{'...' if len(query) > 50 else ''}")
            try:
                response = self.agent.chat(query)
                print(f"✅ Handled gracefully: {response[:100]}...")
            except Exception as e:
                print(f"❌ Error occurred: {str(e)[:100]}...")

    def run_complete_demo(self):
        """Run the complete LangSmith demonstration"""
        print("🎵 MUSIC CONCIERGE AGENT - LANGSMITH COMPLETE DEMO")
        print("=" * 80)
        print("This demo showcases all major LangSmith features:")
        print("• Tracing and observability")
        print("• Memory and conversation tracking")
        print("• Tool usage and function calling")
        print("• Dataset creation and management")
        print("• Custom evaluators and scoring")
        print("• Advanced evaluation metrics")
        print("• Error handling and fallbacks")
        print("=" * 80)

        try:
            # Setup
            self.setup_agent()

            # Run all demos
            self.demo_basic_conversation()
            self.demo_memory_and_preferences()
            self.demo_tool_usage()
            dataset_name = self.demo_dataset_creation()
            quick_eval_dataset = self.demo_custom_evaluators()
            advanced_eval_dataset = self.demo_advanced_evaluation()
            self.demo_streaming_and_async()
            self.demo_error_handling_and_fallbacks()

            # Final summary
            print("\n" + "="*80)
            print("🎉 DEMO COMPLETE - LANGSMITH FEATURES SHOWCASED")
            print("="*80)
            print("✅ Tracing: All conversations traced in LangSmith")
            print("✅ Memory: Conversation history and preferences tracked")
            print("✅ Tools: Spotify integration with function calling")
            print("✅ Datasets: Multiple evaluation datasets created")
            print("✅ Evaluation: Custom evaluators and metrics implemented")
            print("✅ Error Handling: Graceful fallback mechanisms")
            print("\n🔗 View all traces and evaluations at: https://smith.langchain.com/")
            print(f"📊 Datasets created: {dataset_name}, {quick_eval_dataset}, {advanced_eval_dataset}")

        except Exception as e:
            print(f"❌ Demo failed: {str(e)}")
            raise

def main():
    """Main entry point for the demo"""
    # Check environment
    required_env_vars = ["OPENAI_API_KEY", "LANGSMITH_API_KEY"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]

    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set these variables before running the demo.")
        return

    # Run demo
    demo = MusicConciergeDemo()
    demo.run_complete_demo()

if __name__ == "__main__":
    main()