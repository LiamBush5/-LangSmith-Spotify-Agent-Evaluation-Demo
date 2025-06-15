#!/usr/bin/env python3
"""
Spotify Music Concierge Agent Demo

Demonstrates the clean agent architecture with LangSmith tracing.
Built following the financial agent patterns for consistency.
"""

from concierge import SpotifyMusicAgent, run_spotify_agent_with_project_routing
import pandas as pd

def main():
    """Run demonstration of Spotify Music Concierge Agent."""
    print("🎵 " + "="*70)
    print("SPOTIFY MUSIC CONCIERGE AGENT DEMO")
    print("Clean Architecture with LangSmith Tracing")
    print("="*72)

    # Demo queries showcasing different capabilities
    demo_queries = [
        "Find me some Taylor Swift songs",
        "Create a workout playlist with pop and electronic music",
        "Who won the Grammy for Best New Artist in 2024 and get their top songs",
        "Get me similar artists to The Weeknd",
        "Find some chill hip-hop tracks"
    ]

    print(f"📝 Running {len(demo_queries)} demonstration queries...")
    print("🔍 Each query will showcase different tool combinations")
    print("📊 All interactions traced in LangSmith for observability")

    agent = SpotifyMusicAgent()

    for i, query in enumerate(demo_queries, 1):
        print(f"\n🎯 Demo {i}/{len(demo_queries)}: {query}")
        print("-" * 60)

        try:
            # Get response from agent
            result = agent.analyze_query(query)

            # Display response (brief like Spotify DJ)
            print(f"🎵 DJ Response: {result['response']}")
            print(f"🔧 Tools Used: {', '.join(result['unique_tools_used'])}")
            print(f"📊 Tool Calls: {result['total_tool_calls']}")
            print(f"🎶 Songs Found: {result['songs_found']}")

            if result.get('error'):
                print(f"❌ Error occurred: {result.get('error')}")

        except Exception as e:
            print(f"❌ Demo {i} failed: {e}")

    print(f"\n{'='*72}")
    print("✅ DEMO COMPLETE")
    print("📈 Check LangSmith dashboard for detailed traces:")
    print("   - Tool execution times")
    print("   - Agent reasoning steps")
    print("   - Structured data outputs")
    print("   - Error handling patterns")

def run_evaluation_demo():
    """Demonstrate evaluation function for LangSmith datasets."""
    print("\n🧪 " + "="*60)
    print("EVALUATION FUNCTION DEMO")
    print("="*62)

    # Sample evaluation inputs
    eval_inputs = [
        {"input": "Find popular songs by Billie Eilish"},
        {"query": "Create a rock playlist"},
        {"question": "What are some good workout songs?"}
    ]

    print(f"📋 Testing evaluation function with {len(eval_inputs)} inputs...")

    for i, inputs in enumerate(eval_inputs, 1):
        print(f"\n📝 Eval {i}: {inputs}")
        try:
            # Use the evaluation wrapper function
            result = run_spotify_agent_with_project_routing(inputs)
            print(f"✅ Success: {len(result.get('response', ''))} chars response")
            print(f"🔧 Tools: {result.get('total_tool_calls', 0)} calls")
            print(f"🎶 Songs: {result.get('songs_found', 0)} found")
        except Exception as e:
            print(f"❌ Eval {i} failed: {e}")

    print(f"\n✅ Evaluation demo complete!")
    print("💡 Use run_spotify_agent_with_project_routing() for LangSmith evaluations")

if __name__ == "__main__":
    main()
    run_evaluation_demo()

    print(f"\n🎯 " + "="*60)
    print("NEXT STEPS")
    print("="*62)
    print("1. 📊 View traces in LangSmith dashboard")
    print("2. 🧪 Create evaluation datasets")
    print("3. 📈 Monitor agent performance over time")
    print("4. 🔧 Optimize tool usage patterns")
    print("5. 🎵 Enhance music discovery algorithms")