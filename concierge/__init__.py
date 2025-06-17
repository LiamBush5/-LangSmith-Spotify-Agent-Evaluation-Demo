"""
Spotify Music Concierge Package
Clean public API for the music recommendation agent following financial agent patterns
"""

from .music_agent import SpotifyMusicAgent, run_spotify_agent, run_spotify_agent_with_project_routing
from .spotify_tools import SPOTIFY_TOOLS
from .client import WorkingSpotifyClient
from . import config

__all__ = [
    "SpotifyMusicAgent",
    "run_spotify_agent",
    "run_spotify_agent_with_project_routing",
    "SPOTIFY_TOOLS",
    "WorkingSpotifyClient",
    "config",
]

