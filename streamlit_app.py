"""
Music Concierge Agent - Streamlit Web Interface
A beautiful web interface showcasing LangSmith integration
"""

import streamlit as st
import os
import json
from datetime import datetime
from agent import MusicConciergeAgent
from evaluations import MusicAgentEvaluator
from langsmith import Client
from dotenv import load_dotenv
load_dotenv()


# Configure Streamlit page
st.set_page_config(
    page_title="🎵 Music Concierge Agent",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1DB954;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .user-message {
        background-color: #f0f2f6;
        border-left: 4px solid #1DB954;
    }
    .agent-message {
        background-color: #e8f4fd;
        border-left: 4px solid #0066cc;
    }
    .sidebar-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'langsmith_client' not in st.session_state:
    st.session_state.langsmith_client = None

def initialize_agent():
    """Initialize the music agent"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("❌ OPENAI_API_KEY environment variable is required")
        return False

    try:
        st.session_state.agent = MusicConciergeAgent(api_key)
        st.session_state.langsmith_client = Client()
        return True
    except Exception as e:
        st.error(f"❌ Failed to initialize agent: {str(e)}")
        return False

def main():
    """Main Streamlit application"""

    # Header
    st.markdown('<h1 class="main-header">🎵 Music Concierge Agent</h1>', unsafe_allow_html=True)
    st.markdown("### *Powered by LangChain + LangSmith + Spotify*")

    # Sidebar
    with st.sidebar:
        st.markdown("## 🎛️ Control Panel")

        # Agent Status
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("### 🤖 Agent Status")

        if st.session_state.agent is None:
            if st.button("🚀 Initialize Agent", type="primary"):
                with st.spinner("Initializing Music Concierge..."):
                    if initialize_agent():
                        st.success("✅ Agent initialized!")
                        st.rerun()
        else:
            st.success("✅ Agent Ready")
            if st.button("🔄 Restart Agent"):
                st.session_state.agent = None
                st.session_state.chat_history = []
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

        # LangSmith Integration
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("### 📊 LangSmith Features")

        if st.session_state.agent:
            col1, col2 = st.columns(2)

            with col1:
                if st.button("🧪 Run Evaluation"):
                    with st.spinner("Running evaluation..."):
                        evaluator = MusicAgentEvaluator(st.session_state.agent)
                        dataset_name = evaluator.create_evaluation_dataset()
                        st.success(f"✅ Dataset created: {dataset_name}")

            with col2:
                if st.button("🧠 View Memory"):
                    memory_summary = st.session_state.agent.get_memory_summary()
                    st.info(f"Memory: {memory_summary}")

            if st.button("🧹 Clear Memory"):
                st.session_state.agent.clear_memory()
                st.session_state.chat_history = []
                st.success("🧹 Memory cleared!")
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

        # Quick Actions
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("### ⚡ Quick Actions")

        quick_prompts = [
            "🎵 Recommend upbeat pop songs",
            "🎸 Find me some rock music",
            "🎷 Suggest jazz for dinner",
            "💃 Create a dance playlist",
            "😴 Chill music for studying"
        ]

        for prompt in quick_prompts:
            if st.button(prompt, key=f"quick_{prompt}"):
                if st.session_state.agent:
                    # Add to chat and process
                    user_input = prompt.split(" ", 1)[1]  # Remove emoji
                    st.session_state.chat_history.append(("user", user_input))

                    with st.spinner("🎵 Finding perfect music..."):
                        response = st.session_state.agent.chat(user_input)
                        st.session_state.chat_history.append(("agent", response))

                    st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

        # LangSmith Links
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("### 🔗 LangSmith Dashboard")
        st.markdown("[View Traces](https://smith.langchain.com/)")
        st.markdown("[View Datasets](https://smith.langchain.com/datasets)")
        st.markdown("[View Evaluations](https://smith.langchain.com/experiments)")
        st.markdown('</div>', unsafe_allow_html=True)

    # Main chat interface
    if st.session_state.agent is None:
        st.info("👈 Please initialize the agent using the sidebar to start chatting!")

        # Show demo information
        st.markdown("## 🎯 What This Demo Showcases")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            ### 🔍 LangSmith Tracing
            - Every conversation is traced
            - Tool usage is monitored
            - Performance metrics tracked
            - Error handling logged
            """)

        with col2:
            st.markdown("""
            ### 🧪 Evaluation & Testing
            - Custom evaluation datasets
            - Multiple scoring metrics
            - Automated testing workflows
            - Performance benchmarking
            """)

        with col3:
            st.markdown("""
            ### 🎵 Music Intelligence
            - Spotify API integration
            - Personalized recommendations
            - Memory of preferences
            - Contextual responses
            """)

        # Show sample interactions
        st.markdown("## 💬 Sample Interactions")

        sample_conversations = [
            {
                "user": "I love energetic pop music for working out",
                "agent": "Perfect! I'll find some high-energy pop tracks that'll keep you motivated. Let me search for upbeat songs with great beats..."
            },
            {
                "user": "Find me songs similar to Taylor Swift",
                "agent": "Great choice! I'll look for artists and songs with similar styles to Taylor Swift. She has such a diverse catalog..."
            },
            {
                "user": "Create a chill playlist for studying",
                "agent": "I'll curate a perfect study playlist with ambient and lo-fi tracks that won't distract from your focus..."
            }
        ]

        for i, conv in enumerate(sample_conversations):
            with st.expander(f"Sample Conversation {i+1}"):
                st.markdown(f"**🎤 User:** {conv['user']}")
                st.markdown(f"**🤖 Agent:** {conv['agent']}")

    else:
        # Chat interface
        st.markdown("## 💬 Chat with Your Music Concierge")

        # Display chat history
        chat_container = st.container()

        with chat_container:
            for role, message in st.session_state.chat_history:
                if role == "user":
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <strong>🎤 You:</strong> {message}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-message agent-message">
                        <strong>🤖 Music Concierge:</strong> {message}
                    </div>
                    """, unsafe_allow_html=True)

        # Chat input
        user_input = st.chat_input("Ask me about music, request recommendations, or create playlists...")

        if user_input:
            # Add user message to history
            st.session_state.chat_history.append(("user", user_input))

            # Get agent response
            with st.spinner("🎵 Thinking about music..."):
                try:
                    response = st.session_state.agent.chat(user_input)
                    st.session_state.chat_history.append(("agent", response))
                except Exception as e:
                    error_msg = f"I apologize, but I encountered an error: {str(e)}"
                    st.session_state.chat_history.append(("agent", error_msg))

            st.rerun()

        # Show current preferences (if any)
        if hasattr(st.session_state.agent, 'user_preferences'):
            with st.expander("🎯 Current Music Preferences"):
                prefs = st.session_state.agent.user_preferences

                col1, col2 = st.columns(2)

                with col1:
                    if prefs.favorite_genres:
                        st.markdown("**Favorite Genres:**")
                        for genre in prefs.favorite_genres:
                            st.markdown(f"• {genre}")

                    if prefs.favorite_artists:
                        st.markdown("**Favorite Artists:**")
                        for artist in prefs.favorite_artists:
                            st.markdown(f"• {artist}")

                with col2:
                    st.markdown(f"**Current Mood:** {prefs.mood}")
                    st.markdown(f"**Energy Level:** {prefs.energy_level}")

                    if prefs.liked_songs:
                        st.markdown(f"**Liked Songs:** {len(prefs.liked_songs)}")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        🎵 Music Concierge Agent | Powered by LangChain + LangSmith + Spotify API<br>
        <a href="https://smith.langchain.com/" target="_blank">View LangSmith Dashboard</a> |
        <a href="https://github.com/langchain-ai/langchain" target="_blank">LangChain GitHub</a>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()