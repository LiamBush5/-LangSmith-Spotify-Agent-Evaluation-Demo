import uuid
from typing import Dict, Any, Optional
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langsmith import Client
from langsmith.run_helpers import traceable
from .spotify_tools import SPOTIFY_TOOLS
from . import config
import pandas as pd

# Initialize LangSmith client
client = Client()

class SpotifyMusicAgent:
    """
    Spotify music concierge agent with comprehensive tool access and reasoning.
    """

    def __init__(self):
        """Initialize the music agent with tools and LLM."""
        self.tools = SPOTIFY_TOOLS
        self.llm = config.get_chat_model()
        self.agent = self._create_agent()
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=config.AGENT_MAX_ITERATIONS,
            max_execution_time=config.AGENT_MAX_EXECUTION_TIME,
            return_intermediate_steps=True
        )

    def _create_agent(self):
        """Create ReAct agent with music expertise."""
        prompt_template = """You are a sophisticated music concierge with access to Spotify's catalog and music discovery tools. You're like Spotify's AI DJ - brief, cool, and strategic.

AVAILABLE TOOLS:
{tools}

CRITICAL EFFICIENCY RULES:
1. Use MAXIMUM 2-3 tools per query to avoid iteration limits
2. Be DECISIVE - don't second-guess tool results
3. For simple requests, use ONLY 1 tool
4. For complex requests, use MAX 3 tools strategically

TOOL INPUT FORMAT:
- search_tracks: Use "Taylor Swift" (artist or song name)
- get_artist_top_songs: Use "Drake" (artist name only)
- get_similar_songs: Use "The Weeknd" (artist name only)
- get_genre_songs: Use "pop" (genre name only)
- create_smart_playlist: Use {{"name": "My Playlist", "seed_artists": ["Artist1"], "seed_genres": ["pop"], "size": 20}}
- tavily_search_results_json: Use "Grammy winners 2024" (search query)

EFFICIENT TOOL USAGE:
- Simple search: ONLY use search_tracks OR get_artist_top_songs
- Artist discovery: ONLY use get_artist_top_songs OR get_similar_songs
- Genre exploration: ONLY use get_genre_songs
- Playlist creation: Use create_smart_playlist with data from 1-2 other tools MAX
- Current info: Use tavily_search THEN 1 music tool

SPOTIFY DJ VOICE (CRITICAL):
- 1-2 sentences max - brief and natural like a real DJ
- Sound like a chill friend who knows music, not a music professor
- NO lists, NO track breakdowns, NO song title mentions in your response
- Focus ONLY on the vibe, energy, and feeling - never individual tracks
- Use authentic DJ language: "Just whipped up", "This hits different", "Perfect energy", "Killer mix", "About to drop some heat"
- Let the structured data show the actual songs - your job is pure vibe commentary

RESPONSE EXAMPLES:
 "Just whipped up a killer rock mix that captures that Green Day and U2 energy perfectly!"
"About to drop some fire tracks with that perfect workout energy."
"This mix hits different - pure nostalgic vibes coming your way."

NEVER DO THIS:
Don't mention specific song titles like "Wake Me Up When September Ends"
Don't say "featuring tracks like..." or "you'll find songs such as..."
Don't describe what's IN the playlist - describe the FEELING

Remember: You're a DJ dropping knowledge, not a music encyclopedia!

STOP AFTER SUCCESS: Once you get good results from tools, provide your Final Answer immediately!

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Question: {input}
Thought: {agent_scratchpad}"""

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["input", "agent_scratchpad"],
            partial_variables={
                "tools": "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools]),
                "tool_names": ", ".join([tool.name for tool in self.tools])
            }
        )

        return create_react_agent(self.llm, self.tools, prompt)

    def _serialize_tool_output(self, output: Any) -> Any:
        """Convert Pydantic models to dictionaries for LangSmith compatibility."""
        from pydantic import BaseModel

        if isinstance(output, BaseModel):
            return output.model_dump()
        elif isinstance(output, list):
            return [self._serialize_tool_output(item) for item in output]
        elif isinstance(output, dict):
            return {k: self._serialize_tool_output(v) for k, v in output.items()}
        else:
            return output

    @traceable(
        run_type="chain",
        name="SpotifyMusicAgentAnalysis",
        tags=["spotify_agent", "music_analysis"],
        metadata={"agent_version": "v2.1"}
    )
    def analyze_query(self, query: str, thread_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze a music question and return structured results.

        Args:
            query: The music question to analyze
            thread_id: (Deprecated) Previously used for thread grouping

        Returns:
            Dictionary with agent response, reasoning steps, and tool usage metadata
        """
        print(f"\n🎵 Analyzing Music Query: {query}")
        print("="*60)

        if thread_id is None:
            thread_id = str(uuid.uuid4())

        # Get current trace ID
        trace_id = None
        try:
            from langsmith.run_helpers import get_current_run_tree
            current_run = get_current_run_tree()
            trace_id = str(current_run.trace_id) if current_run else None
        except Exception as e:
            print(f"Could not get trace_id: {e}")

        try:
            # Execute the agent
            result = self.agent_executor.invoke(
                {"input": query},
                config={
                    "metadata": {
                        "query": query,
                        "agent_type": "spotify_music",
                    },
                    "tags": ["spotify_agent"]
                }
            )

            # Extract information
            response = result.get("output", "")
            intermediate_steps = result.get("intermediate_steps", [])

            # Process intermediate steps
            tool_trajectory = []
            reasoning_steps = []
            songs_found = []

            for step in intermediate_steps:
                if len(step) >= 2:
                    action, observation = step[0], step[1]
                    tool_name = action.tool if hasattr(action, 'tool') else "unknown"
                    tool_input = action.tool_input if hasattr(action, 'tool_input') else ""

                    tool_trajectory.append(tool_name)
                    serialized_observation = self._serialize_tool_output(observation)

                    # Extract songs from structured outputs
                    if isinstance(serialized_observation, dict):
                        if 'tracks' in serialized_observation:
                            songs_found.extend(serialized_observation['tracks'])
                        elif 'songs' in serialized_observation:
                            songs_found.extend(serialized_observation['songs'])

                    reasoning_steps.append({
                        "tool": tool_name,
                        "input": str(tool_input),
                        "output": str(serialized_observation)[:200] + "..." if len(str(serialized_observation)) > 200 else str(serialized_observation)
                    })

            # Compile results
            analysis_result = {
                "response": response,
                "tool_trajectory": tool_trajectory,
                "reasoning_steps": reasoning_steps,
                "total_tool_calls": len(tool_trajectory),
                "unique_tools_used": list(set(tool_trajectory)),
                "songs_found": len(songs_found),
                "songs": songs_found,  # Add the actual songs array
                "query": query,
                "thread_id": thread_id,
                "trace_id": trace_id  # Add trace_id to response
            }

            print(f"\nMusic Analysis Complete!")
            print(f"Tools Used: {', '.join(analysis_result['unique_tools_used'])}")
            print(f"Total Tool Calls: {analysis_result['total_tool_calls']}")
            print(f"Songs Found: {analysis_result['songs_found']}")

            if analysis_result['total_tool_calls'] >= config.AGENT_MAX_ITERATIONS * 0.8:
                print(f"⚠️  Warning: High tool usage ({analysis_result['total_tool_calls']}/{config.AGENT_MAX_ITERATIONS})")

            return analysis_result

        except Exception as e:
            error_result = {
                "response": f"Error during music analysis: {str(e)}",
                "tool_trajectory": [],
                "reasoning_steps": [],
                "total_tool_calls": 0,
                "unique_tools_used": [],
                "songs_found": 0,
                "songs": [],  # Add empty songs array
                "query": query,
                "thread_id": thread_id,
                "trace_id": trace_id,  # Add trace_id to error response too
                "error": True
            }

            print(f"Music analysis failed: {str(e)}")
            return error_result


@traceable(
    run_type="chain",
    name="SpotifyMusicAgentEvaluation",
    tags=["spotify_agent", "evaluation"],
    metadata={"evaluation_run": True, "agent_version": "v2.1"}
)
def run_spotify_agent(inputs: Dict[str, str]) -> Dict[str, Any]:
    """
    Main entry point for Spotify music agent evaluation.

    Args:
        inputs: Dictionary containing the query and additional parameters

    Returns:
        Dictionary containing the agent's response and metadata
    """
    # Handle multiple input formats
    query = inputs.get("input", inputs.get("query", inputs.get("question", "")))

    if not query:
        return {
            "response": "No music query provided",
            "error": True,
            "tool_trajectory": [],
            "reasoning_steps": [],
            "total_tool_calls": 0,
            "unique_tools_used": [],
            "songs_found": 0,
            "thread_id": None
        }

    print(f"\n{'='*80}")
    print("🎵 SPOTIFY MUSIC AGENT EVALUATION")
    print(f"Query: {query}")
    print(f"{'='*80}")

    agent = SpotifyMusicAgent()
    result = agent.analyze_query(query)

    # Add timestamp
    result.update({
        "timestamp": str(pd.Timestamp.now())
    })

    print("\nMusic Evaluation Complete")
    print(f"Response Length: {len(result.get('response', ''))}")
    print(f"Tools Used: {result.get('total_tool_calls', 0)}")
    print(f"Songs Discovered: {result.get('songs_found', 0)}")

    return result

@traceable(
    run_type="chain",
    name="SpotifyMusicAgentEvaluationWrapper",
    project_name=config.LANGSMITH_PROJECT,
    tags=["spotify_agent", "evaluation", "wrapper"]
)
def run_spotify_agent_with_project_routing(inputs: Dict[str, str]) -> Dict[str, Any]:
    """
    Wrapper function that ensures traces go to the correct project.
    This is the function that should be passed to evaluate().
    """
    return run_spotify_agent(inputs)