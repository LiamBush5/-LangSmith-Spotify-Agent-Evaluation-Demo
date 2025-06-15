#!/usr/bin/env python3
"""
Modern Music Concierge UI Demo
==============================

This script demonstrates the new modern UI features:
- Thinking process visualization with animations
- Tool call execution with real-time status updates
- Beautiful song cards with play previews
- Voice input simulation
- Streaming responses with typing animations
- Modern gradients and smooth animations

Run this after starting both servers:
1. cd spotify-chat-ui && npm run dev (port 3000)
2. python run_api.py (port 8000)
"""

import time
import webbrowser
import subprocess
import sys
from pathlib import Path

def print_banner():
    print("🎵" + "=" * 60 + "🎵")
    print("      MODERN MUSIC CONCIERGE UI DEMO")
    print("      Featuring AI Thinking + Tool Animations")
    print("🎵" + "=" * 60 + "🎵")
    print()

def check_servers():
    """Check if both servers are running"""
    import requests

    try:
        # Check Next.js frontend
        response = requests.get("http://localhost:3000", timeout=2)
        print("✅ Frontend server (Next.js) is running at http://localhost:3000")
    except:
        print("❌ Frontend server not found. Please run: cd spotify-chat-ui && npm run dev")
        return False

    try:
        # Check Python backend
        response = requests.get("http://localhost:8000/health", timeout=2)
        print("✅ Backend server (FastAPI) is running at http://localhost:8000")
    except:
        print("❌ Backend server not found. Please run: python run_api.py")
        return False

    return True

def show_features():
    """Display the new UI features"""
    features = [
        "🧠 AI Thinking Process - Watch the model reason through your request",
        "🔧 Tool Call Animations - See exactly which tools are being executed",
        "🎵 Interactive Song Cards - Play previews with smooth animations",
        "🎤 Voice Input Simulation - Modern speech-to-text interface",
        "✨ Streaming Responses - Character-by-character typing animation",
        "🎨 Modern Design - Gradient backgrounds and smooth transitions",
        "📱 Responsive Layout - Works beautifully on all screen sizes",
        "🌙 Dark Mode Support - Automatic theme detection",
        "⚡ Performance Optimized - Built with React 19 + Framer Motion",
        "🔄 Real-time Updates - Live tool execution status"
    ]

    print("🚀 NEW FEATURES IN THIS UI:")
    print("-" * 40)
    for feature in features:
        print(f"  {feature}")
        time.sleep(0.3)
    print()

def demo_scenarios():
    """Show example queries to try"""
    scenarios = [
        {
            "title": "🎯 Mood-Based Discovery",
            "query": "I need some upbeat songs to get me motivated for my workout",
            "features": ["Mood analysis tool", "Energy level matching", "Workout-specific curation"]
        },
        {
            "title": "🎨 Artist Exploration",
            "query": "Show me songs similar to Taylor Swift's latest hits",
            "features": ["Artist similarity search", "Recent releases analysis", "Popularity scoring"]
        },
        {
            "title": "📝 Playlist Creation",
            "query": "Create a romantic dinner playlist with soft jazz and acoustic songs",
            "features": ["Theme-based generation", "Genre filtering", "Vibe matching"]
        },
        {
            "title": "🔍 Trending Discovery",
            "query": "What are the most popular songs right now in electronic music?",
            "features": ["Trending analysis", "Genre-specific search", "Popularity metrics"]
        }
    ]

    print("💡 TRY THESE DEMO SCENARIOS:")
    print("-" * 40)
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{i}. {scenario['title']}")
        print(f"   Query: \"{scenario['query']}\"")
        print(f"   Shows: {', '.join(scenario['features'])}")
    print()

def open_ui():
    """Open the UI in the default browser"""
    print("🌐 Opening Modern Music Concierge UI...")
    time.sleep(1)
    webbrowser.open("http://localhost:3000")
    print("✅ UI opened in your default browser!")
    print()

def main():
    print_banner()

    # Check if servers are running
    if not check_servers():
        print("\n❌ Please start both servers first:")
        print("   1. cd spotify-chat-ui && npm run dev")
        print("   2. python run_api.py")
        return

    print()
    show_features()
    demo_scenarios()

    # Open the UI
    open_ui()

    print("🎵 DEMO TIPS:")
    print("-" * 20)
    print("• Toggle 'Show Thinking' to see AI reasoning")
    print("• Watch tool calls execute in real-time")
    print("• Click play buttons to preview songs")
    print("• Try voice input with the microphone button")
    print("• Use quick prompt buttons for inspiration")
    print("• Notice the smooth animations and transitions")
    print()

    print("🚀 The UI is now ready for your musical exploration!")
    print("   Open: http://localhost:3000")
    print()

    # Keep the demo running
    try:
        print("Press Ctrl+C to exit demo...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 Demo ended. Enjoy your modern music concierge!")

if __name__ == "__main__":
    main()