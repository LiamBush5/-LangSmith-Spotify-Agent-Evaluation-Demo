"""
Music Concierge Agent - Main Agent Class
"""

import os
import uuid
from dotenv import load_dotenv

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_core.callbacks import BaseCallbackHandler

# LangSmith imports
from langsmith import Client, traceable

# Import tools and models
from tools import ALL_TOOLS, MusicPreferences

# Load environment variables
load_dotenv()

# Configure LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "music-concierge-demo"

# LangSmith tracking
class LangSmithCallbackHandler(BaseCallbackHandler):
    """Custom callback handler for LangSmith integration"""

    def __init__(self):
        self.client = Client()
        self.current_run_id = None

    def on_chain_start(self, serialized, inputs, **kwargs):
        """Track chain starts"""
        self.current_run_id = str(uuid.uuid4())

    def on_tool_start(self, serialized, input_str, **kwargs):
        """Track tool usage"""
        tool_name = serialized.get("name", "unknown_tool")
        print(f"🔧 Using tool: {tool_name}")

class MusicConciergeAgent:
    """Main music concierge agent with LangSmith integration"""

    def __init__(self, openai_api_key: str):
        # Initialize LLM
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",  # Updated to more cost-effective model
            temperature=0.7,
            api_key=openai_api_key
        )

        # Initialize LangSmith client
        self.langsmith_client = Client()

        # Initialize memory
        self.memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            max_token_limit=1000,
            memory_key="chat_history",
            return_messages=True
        )

        # User preferences storage (in production, use database)
        self.user_preferences = MusicPreferences()

        # Create tools list
        self.tools = ALL_TOOLS

        # Create prompt template (will be updated dynamically)
        self._update_prompt()

        # Create agent
        self.agent = create_openai_functions_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt
        )

        # Create agent executor with callbacks
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            max_iterations=5,
            callbacks=[LangSmithCallbackHandler()]
        )

    def _get_system_prompt(self) -> str:
        """Generate system prompt with current user preferences"""
        preferences_text = self._format_user_preferences()

        return f"""You are a world-class music concierge and DJ assistant. Your role is to:

1. 🎵 **Understand musical taste**: Learn about users' preferences through conversation
2. 🔍 **Find perfect songs**: Use Spotify tools to search and recommend music
3. 📝 **Build playlists**: Create curated playlists based on mood, genre, or activity
4. 💭 **Remember preferences**: Save and use user preferences for personalized recommendations
5. 🎯 **Be contextual**: Consider the user's current mood, activity, or specific request

**Available Tools:**
- search_spotify_songs: Search for any song, artist, or keyword
- get_artist_top_songs: Get popular songs by a specific artist
- get_similar_music: Find music similar to a given artist
- get_genre_recommendations: Get songs by genre
- create_smart_playlist: Create AI-curated playlists with flow optimization
- add_song_to_session_playlist: Build playlists from conversation context
- save_user_preference: Remember user's music preferences

{preferences_text}

**Guidelines:**
- Always be enthusiastic and knowledgeable about music
- Ask follow-up questions to understand preferences better
- Provide reasoning for your recommendations
- Include Spotify links when sharing songs
- Remember past preferences and reference them
- **For playlist creation requests, ALWAYS use create_smart_playlist tool** - don't just list top songs
- Use session playlists to track songs mentioned in conversation
- Focus on popularity and metadata-based insights (audio features not available)
- When creating playlists, consider mood, genre, and user preferences for true curation

**Response Format:**
- Use emojis to make responses engaging
- Format song lists clearly with titles, artists, and links
- Explain why you chose specific recommendations
- Offer to save preferences or create playlists

Let's discover some amazing music together! 🎶"""

    def _format_user_preferences(self) -> str:
        """Format current user preferences for the system prompt"""
        if not any([
            self.user_preferences.favorite_genres,
            self.user_preferences.favorite_artists,
            self.user_preferences.mood != "neutral",
            self.user_preferences.liked_songs,
            self.user_preferences.disliked_songs
        ]):
            return "**Current User Preferences:** None saved yet - learn about the user through conversation!"

        prefs = []
        if self.user_preferences.favorite_genres:
            prefs.append(f"Genres: {', '.join(self.user_preferences.favorite_genres)}")
        if self.user_preferences.favorite_artists:
            prefs.append(f"Artists: {', '.join(self.user_preferences.favorite_artists)}")
        if self.user_preferences.mood != "neutral":
            prefs.append(f"Current mood: {self.user_preferences.mood}")
        if self.user_preferences.energy_level != 0.5:
            energy_desc = "high" if self.user_preferences.energy_level > 0.7 else "low" if self.user_preferences.energy_level < 0.3 else "medium"
            prefs.append(f"Energy preference: {energy_desc}")

        return f"**Current User Preferences:** {' | '.join(prefs)}"

    def _update_prompt(self):
        """Update the prompt template with current preferences"""
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_system_prompt()),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad")
        ])

        # Recreate agent with updated prompt
        self.agent = create_openai_functions_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt
        )

        # Recreate agent executor
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            max_iterations=5,
            callbacks=[LangSmithCallbackHandler()]
        )

    def update_user_preferences(self, **kwargs):
        """Update user preferences and refresh the agent prompt"""
        for key, value in kwargs.items():
            if hasattr(self.user_preferences, key):
                setattr(self.user_preferences, key, value)

        # Update the prompt with new preferences
        self._update_prompt()

    @traceable
    def chat(self, user_input: str) -> str:
        """Main chat interface with LangSmith tracing"""
        try:
            # Process user input
            response = self.agent_executor.invoke({
                "input": user_input
            })

            return response["output"]

        except Exception as e:
            error_msg = f"I apologize, but I encountered an error: {str(e)}"
            print(f"Agent error: {e}")
            return error_msg

    def get_memory_summary(self) -> str:
        """Get current conversation summary"""
        if hasattr(self.memory, 'predict_new_summary'):
            return self.memory.predict_new_summary([], "")
        return "No conversation history yet."

    def clear_memory(self):
        """Clear conversation memory"""
        self.memory.clear()
        self.user_preferences = MusicPreferences()

def main():
    """Main demo function"""
    print("🎵 Music Concierge Agent - LangSmith Demo")
    print("=" * 50)

    # Check for OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Please set OPENAI_API_KEY environment variable")
        return

    # Initialize agent
    agent = MusicConciergeAgent(api_key)

    print("✅ Agent initialized with LangSmith tracing enabled")
    print("🔗 View traces at: https://smith.langchain.com/")

    # Interactive demo
    print("\n🎯 Starting interactive demo...")
    print("Type 'quit' to exit, 'memory' to see summary, 'clear' to clear memory")

    while True:
        user_input = input("\n🎤 You: ").strip()

        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        elif user_input.lower() == 'memory':
            print(f"\n🧠 Memory Summary: {agent.get_memory_summary()}")
            continue
        elif user_input.lower() == 'clear':
            agent.clear_memory()
            print("🧹 Memory cleared!")
            continue

        if not user_input:
            continue

        print("\n🤖 Music Concierge:")
        response = agent.chat(user_input)
        print(response)

if __name__ == "__main__":
    main()