#!/usr/bin/env python3
"""
Music Concierge Demo - Refactored Version
Showcases the clean LCEL-based API
"""

import os
from dotenv import load_dotenv
from concierge import Concierge

def main():
    """Demo the refactored music concierge"""
    # Load environment variables
    load_dotenv()

    # Check for required environment variables
    required_vars = ["OPENAI_API_KEY", "SPOTIFY_CLIENT_ID", "SPOTIFY_CLIENT_SECRET"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\n💡 Please copy env.template to .env and fill in your API keys")
        return

    print("🎵 Music Concierge - Refactored Demo")
    print("=" * 50)

    # Initialize the concierge
    try:
        agent = Concierge(user_id="demo_user")
        print("✅ Successfully initialized Music Concierge!")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return

    # Demo interactions
    demo_queries = [
        "Hi! I love rock music and Taylor Swift. Can you recommend some songs?",
        "Create a workout playlist with energetic pop and rock songs",
        "What are some popular songs by The Beatles?",
        "I'm feeling sad today, can you suggest some comforting music?"
    ]

    print("\n🚀 Running Demo Interactions...")

    for i, query in enumerate(demo_queries, 1):
        print(f"\n📝 Demo {i}: {query}")
        print("-" * 60)

        try:
            # Get response from agent
            response = agent.chat(query)

            # Display response
            print(f"🤖 Response: {response.message}")
            print(f"📊 Type: {response.response_type}")

            if response.songs:
                print(f"🎵 Found {len(response.songs)} songs:")
                for j, song in enumerate(response.songs[:3], 1):  # Show first 3
                    print(f"   {j}. {song.name} by {song.artist}")
                    if song.spotify_url:
                        print(f"      🎧 {song.spotify_url}")

                if len(response.songs) > 3:
                    print(f"   ... and {len(response.songs) - 3} more songs")

            if response.metadata:
                tools_used = response.metadata.get("tools_used", [])
                if tools_used:
                    print(f"🔧 Tools used: {', '.join(tools_used)}")

        except Exception as e:
            print(f"❌ Error: {e}")

    # Show memory and preferences
    print(f"\n💭 Memory Summary: {agent.get_memory_summary()}")

    # Interactive mode
    print("\n" + "=" * 50)
    print("🎯 Interactive Mode - Type 'quit' to exit")

    while True:
        try:
            user_input = input("\n🎵 You: ").strip()

            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Thanks for using Music Concierge!")
                break

            if not user_input:
                continue

            # Get response
            response = agent.chat(user_input)
            print(f"\n🤖 {response.message}")

            # Show songs if any
            if response.songs:
                print(f"\n🎵 Songs ({len(response.songs)}):")
                for song in response.songs[:5]:  # Show first 5
                    print(f"   • {song.name} by {song.artist}")
                    if song.spotify_url:
                        print(f"     🎧 {song.spotify_url}")

        except KeyboardInterrupt:
            print("\n👋 Thanks for using Music Concierge!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()