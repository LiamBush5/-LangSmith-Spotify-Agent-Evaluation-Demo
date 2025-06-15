"""
Music Concierge Agent - Main Agent Class
Clean public API using LCEL and modern LangChain patterns
"""

import os
from typing import List
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from .model import create_model_components
from .schema import MusicResponse, SongInfo
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from .model import create_system_prompt

# Configure LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "music-concierge-demo"

class MusicConciergeAgent:
    """Main music concierge agent with clean public API"""

    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        components = create_model_components(user_id)
        self.chain = components["chain"]
        self.memory = components["memory"]
        self.llm = components["llm"]
        self.tools = components["tools"]

        # Create tool lookup dict
        self.tool_dict = {tool.name: tool for tool in self.tools}

    def chat(self, user_input: str) -> MusicResponse:
        """Main chat interface - returns structured response"""
        try:
            # Add user message to memory
            self.memory.chat_memory.add_user_message(user_input)

            # Get AI response
            ai_msg: AIMessage = self.chain.invoke(user_input)

            # If no tool calls, return simple response
            if not ai_msg.tool_calls:
                self.memory.chat_memory.add_ai_message(ai_msg)
                return MusicResponse(
                    message=ai_msg.content,
                    response_type="conversational"
                )

            # Execute tool calls
            tool_msgs: List[ToolMessage] = []
            all_songs: List[SongInfo] = []
            web_search_results = []

            for tool_call in ai_msg.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                if tool_name in self.tool_dict:
                    try:
                        # Execute tool
                        tool = self.tool_dict[tool_name]
                        result = tool.invoke(tool_args)

                        # Create tool message
                        tool_msg = ToolMessage(
                            content=str(result),
                            tool_call_id=tool_call["id"]
                        )
                        tool_msgs.append(tool_msg)

                        # Handle different tool result types
                        if tool_name == "tavily_search_results_json":
                            # Store web search results separately
                            if isinstance(result, list):
                                web_search_results.extend(result)
                            elif isinstance(result, str):
                                # TavilySearch might return a string
                                web_search_results.append({"content": result})
                        else:
                            # Extract songs from music tools
                            songs = self._extract_songs_from_result(result)
                            all_songs.extend(songs)

                    except Exception as e:
                        # Handle tool errors gracefully
                        error_msg = ToolMessage(
                            content=f"Error using {tool_name}: {str(e)}",
                            tool_call_id=tool_call["id"]
                        )
                        tool_msgs.append(error_msg)

            # Get final response with tool results
            final_prompt = ChatPromptTemplate.from_messages([
                ("system", create_system_prompt(self.user_id)),
                ("human", user_input),
                ai_msg,
                *tool_msgs
            ])

            final_response = self.llm.invoke(final_prompt.format_messages())

            # Add messages to memory
            self.memory.chat_memory.add_ai_message(ai_msg)
            for tool_msg in tool_msgs:
                self.memory.chat_memory.add_message(tool_msg)
            self.memory.chat_memory.add_ai_message(final_response)

            # Determine response type
            response_type = self._determine_response_type(ai_msg.tool_calls, all_songs)

            # Build metadata
            metadata = {
                "tools_used": [call["name"] for call in ai_msg.tool_calls],
                "total_songs": len(all_songs)
            }

            # Add web search results to metadata if any
            if web_search_results:
                metadata["web_search_results"] = web_search_results
                metadata["web_sources"] = len(web_search_results)

            return MusicResponse(
                message=final_response.content,
                songs=all_songs,
                response_type=response_type,
                metadata=metadata
            )

        except Exception as e:
            return MusicResponse(
                message=f"I'm sorry, I encountered an error: {str(e)}. Please try again!",
                response_type="error"
            )

    def _extract_songs_from_result(self, result) -> List[SongInfo]:
        """Extract SongInfo objects from tool results"""
        songs = []

        if isinstance(result, list):
            # Handle list of songs
            for item in result:
                if isinstance(item, dict) and "id" in item and "name" in item:
                    try:
                        songs.append(SongInfo(**item))
                    except Exception:
                        # Skip invalid song data
                        continue
        elif isinstance(result, dict):
            # Handle single song or playlist response
            if "songs" in result and isinstance(result["songs"], list):
                # Playlist response
                for song_data in result["songs"]:
                    if isinstance(song_data, dict):
                        try:
                            songs.append(SongInfo(**song_data))
                        except Exception:
                            continue
            elif "id" in result and "name" in result:
                # Single song
                try:
                    songs.append(SongInfo(**result))
                except Exception:
                    pass

        return songs

    def _determine_response_type(self, tool_calls: List, songs: List[SongInfo]) -> str:
        """Determine the type of response based on tools used"""
        if not tool_calls:
            return "conversational"

        tool_names = [call["name"] for call in tool_calls]

        # Check for web search
        if "tavily_search_results_json" in tool_names:
            return "web_search"
        elif "create_smart_playlist" in tool_names:
            return "playlist"
        elif "search_tracks" in tool_names:
            return "search"
        elif any(name in tool_names for name in ["get_artist_top_songs", "get_similar_songs"]):
            return "recommendations"
        elif "get_genre_songs" in tool_names:
            return "genre_exploration"
        else:
            return "general"

    def get_memory_summary(self) -> str:
        """Get a summary of the conversation memory"""
        if hasattr(self.memory, 'buffer'):
            return self.memory.buffer
        return "No conversation history yet."

    def clear_memory(self):
        """Clear conversation memory"""
        self.memory.clear()