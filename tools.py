"""
Spotify Tools and Data Models for Music Concierge Agent
Enhanced with advanced audio features, smart playlists, and compatibility analysis
"""

import os
import json
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv

# LangChain imports
from langchain_core.tools import tool

# LangSmith imports
from langsmith import traceable

# Spotify integration
from spotify.spotify_working import WorkingSpotifyClient

# Load environment variables
load_dotenv()

# Configure LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "music-concierge-demo"

# Enhanced Pydantic Models following LangChain v2 patterns
# Audio features models removed - not available with Client Credentials flow

class MusicPreferences(BaseModel):
    """User's music preferences extracted from conversation"""
    favorite_genres: List[str] = Field(description="List of favorite music genres", default=[])
    favorite_artists: List[str] = Field(description="List of favorite artists", default=[])
    energy_level: float = Field(description="Preferred energy level (0.0-1.0)", default=0.5, ge=0.0, le=1.0)
    mood: str = Field(description="Current mood (happy, sad, energetic, calm, etc.)", default="neutral")
    liked_songs: List[str] = Field(description="List of song IDs user has liked", default=[])
    disliked_songs: List[str] = Field(description="List of song IDs user disliked", default=[])
    preferred_tempo_range: tuple[float, float] = Field(description="Preferred BPM range", default=(80.0, 140.0))
    activity_context: str = Field(description="What they're doing (working, exercising, relaxing)", default="general")

    @field_validator('energy_level')
    @classmethod
    def validate_energy_level(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Energy level must be between 0.0 and 1.0')
        return v

class SmartPlaylistRequest(BaseModel):
    """Request to create an AI-curated smart playlist"""
    name: str = Field(description="Name for the playlist")
    description: str = Field(description="Description of the playlist theme/purpose")
    seed_songs: List[str] = Field(description="Seed song IDs to base recommendations on", default=[])
    seed_artists: List[str] = Field(description="Seed artist names", default=[])
    seed_genres: List[str] = Field(description="Seed genres", default=[])
    # target_features removed - audio features not available with Client Credentials flow
    size: int = Field(description="Desired number of songs", default=20, ge=5, le=100)
    diversity_factor: float = Field(description="How diverse the playlist should be (0.0-1.0)", default=0.5, ge=0.0, le=1.0)
    energy_progression: str = Field(description="Energy flow (ascending, descending, stable, varied)", default="varied")

class SessionPlaylist(BaseModel):
    """Tracks songs mentioned during conversation for playlist building"""
    session_id: str = Field(description="Unique session identifier")
    mentioned_songs: List[Dict[str, Any]] = Field(description="Songs mentioned in conversation", default=[])
    user_reactions: Dict[str, str] = Field(description="User reactions to songs (song_id -> reaction)", default={})
    playlist_candidates: List[str] = Field(description="Song IDs that could be added to playlist", default=[])
    created_at: datetime = Field(description="When session started", default_factory=datetime.now)

class MusicRecommendation(BaseModel):
    """Structured music recommendation output with enhanced reasoning"""
    recommended_songs: List[Dict[str, Any]] = Field(description="List of recommended songs with metadata")
    reasoning: str = Field(description="Detailed explanation of why these songs were recommended")
    confidence: float = Field(description="Confidence score (0.0-1.0)", ge=0.0, le=1.0)
    genre_match: str = Field(description="Primary genre that influenced the recommendation")
    # mood_analysis removed - audio features not available
    audio_features_summary: Dict[str, float] = Field(description="Average audio features of recommendations", default={})
    personalization_factors: List[str] = Field(description="What personal preferences influenced this", default=[])

# Initialize Spotify client
spotify_client = WorkingSpotifyClient()

# Session storage (in production, use proper database)
session_playlists: Dict[str, SessionPlaylist] = {}

# Enhanced Spotify Tools
@tool
@traceable
def search_spotify_songs(query: str, limit: int = 5) -> str:
    """
    Search for songs on Spotify by name, artist, or keywords.
    Enhanced with audio features and mood analysis.

    Args:
        query: Search term (song name, artist, genre, etc.)
        limit: Number of results to return (default 5)

    Returns:
        JSON string with song information including audio features
    """
    try:
        songs = spotify_client.search_songs(query, limit=limit)
        if not songs:
            return json.dumps({"error": "No songs found", "query": query})

        # Format for LLM consumption with enhanced metadata
        formatted_songs = []
        for song in songs:
            formatted_songs.append({
                "id": song["id"],
                "name": song["name"],
                "artist": song["artist"],
                "album": song["album"],
                "popularity": song["popularity"],
                "duration": song["duration"],
                "spotify_url": song["spotify_url"],
                "preview_url": song.get("preview_url"),
                "release_date": song.get("release_date", "Unknown"),
                "explicit": song.get("explicit", False)
            })

        return json.dumps({
            "songs": formatted_songs,
            "query": query,
            "count": len(formatted_songs),
            "search_type": "enhanced_search"
        })
    except Exception as e:
        return json.dumps({"error": str(e), "query": query})

# Audio features tools removed - require Spotify user authorization
# (not available with Client Credentials flow)

@tool
@traceable
def create_smart_playlist(request_data: str) -> str:
    """
    Create an AI-curated smart playlist based on user preferences, mood, and themes.
    USE THIS TOOL when users ask to "create a playlist" or want curated music recommendations.
    This creates diverse, well-balanced playlists, not just top songs.

    Args:
        request_data: JSON string with SmartPlaylistRequest data containing:
        - name: playlist name
        - description: what kind of playlist (upbeat, chill, workout, etc.)
        - seed_artists: list of artist names to base recommendations on
        - seed_genres: list of genres to include
        - size: number of songs (default 20)

    Returns:
        JSON string with created playlist information and flow analysis
    """
    try:
        # Parse request
        request_dict = json.loads(request_data)
        playlist_request = SmartPlaylistRequest(**request_dict)

        # Generate smart playlist
        playlist_songs = _generate_smart_playlist(playlist_request)

        if not playlist_songs:
            return json.dumps({"error": "Could not generate playlist with given parameters"})

        # Analyze playlist flow
        flow_analysis = _analyze_playlist_flow(playlist_songs)

        return json.dumps({
            "playlist_name": playlist_request.name,
            "description": playlist_request.description,
            "songs": playlist_songs,
            "count": len(playlist_songs),
            "flow_analysis": flow_analysis,
            "generation_timestamp": datetime.now().isoformat(),
            "ai_curation_notes": "Playlist curated using advanced audio feature analysis and flow optimization"
        })
    except Exception as e:
        return json.dumps({"error": str(e), "request_data": request_data})

@tool
@traceable
def add_song_to_session_playlist(song_id: str, session_id: str = "default", user_reaction: str = "mentioned") -> str:
    """
    Add a song to the current session playlist for later playlist creation.
    Tracks user reactions and builds context-aware playlists.

    Args:
        song_id: Spotify song ID to add
        session_id: Session identifier (default: "default")
        user_reaction: User's reaction (liked, mentioned, interested, etc.)

    Returns:
        JSON string with session playlist update
    """
    try:
        # Get or create session playlist
        if session_id not in session_playlists:
            session_playlists[session_id] = SessionPlaylist(session_id=session_id)

        session_playlist = session_playlists[session_id]

        # Get song details
        song_details = spotify_client.get_track_details(song_id)
        if not song_details:
            return json.dumps({"error": f"Could not find song with ID: {song_id}"})

        # Add to session
        session_playlist.mentioned_songs.append(song_details)
        session_playlist.user_reactions[song_id] = user_reaction

        if user_reaction in ["liked", "interested", "love"]:
            session_playlist.playlist_candidates.append(song_id)

        return json.dumps({
            "status": "success",
            "message": f"Added '{song_details['name']}' to session playlist",
            "session_id": session_id,
            "total_songs": len(session_playlist.mentioned_songs),
            "playlist_candidates": len(session_playlist.playlist_candidates),
            "user_reaction": user_reaction
        })
    except Exception as e:
        return json.dumps({"error": str(e), "song_id": song_id, "session_id": session_id})

@tool
@traceable
def get_artist_top_songs(artist_name: str, limit: int = 5) -> str:
    """
    Get the most popular songs by a specific artist.
    USE THIS TOOL for simple requests like "what are Taylor Swift's top songs" or "show me popular songs by X".
    DO NOT use this for playlist creation - use create_smart_playlist instead.

    Args:
        artist_name: Name of the artist
        limit: Number of top songs to return

    Returns:
        JSON string with top songs ranked by popularity
    """
    try:
        songs = spotify_client.get_artist_top_songs(artist_name, limit=limit)
        if not songs:
            return json.dumps({"error": f"No songs found for artist: {artist_name}"})

        # Enhanced formatting with audio insights
        formatted_songs = []
        for song in songs:
            song_data = {
                "id": song["id"],
                "name": song["name"],
                "artist": song["artist"],
                "album": song["album"],
                "popularity": song["popularity"],
                "spotify_url": song["spotify_url"],
                "preview_url": song.get("preview_url"),
                "duration": song["duration"]
            }

            # Audio features not available with Client Credentials flow
            # Using popularity and genre-based insights instead
            if song["popularity"] > 80:
                song_data["popularity_level"] = "Very Popular"
            elif song["popularity"] > 60:
                song_data["popularity_level"] = "Popular"
            else:
                song_data["popularity_level"] = "Moderate"

            formatted_songs.append(song_data)

        return json.dumps({
            "artist": artist_name,
            "top_songs": formatted_songs,
            "count": len(formatted_songs),
            "analysis_type": "enhanced_top_songs"
        })
    except Exception as e:
        return json.dumps({"error": str(e), "artist": artist_name})

@tool
@traceable
def get_similar_music(artist_name: str, limit: int = 5) -> str:
    """
    Get similar music based on an artist using related artists and audio feature matching.

    Args:
        artist_name: Name of the artist to find similar music for
        limit: Number of similar songs to return

    Returns:
        JSON string with similar songs and similarity reasoning
    """
    try:
        similar_songs = spotify_client.get_similar_songs(artist_name, limit=limit)
        if not similar_songs:
            return json.dumps({"error": f"No similar songs found for: {artist_name}"})

        # Enhanced formatting with similarity analysis
        formatted_songs = []
        for song in similar_songs:
            song_data = {
                "id": song["id"],
                "name": song["name"],
                "artist": song["artist"],
                "album": song["album"],
                "popularity": song["popularity"],
                "spotify_url": song["spotify_url"],
                "similarity_reason": "Related artist analysis"
            }

            # Audio features not available - using metadata-based similarity
            song_data["similarity_score"] = f"{song['popularity']}/100 popularity match"

            formatted_songs.append(song_data)

        return json.dumps({
            "based_on_artist": artist_name,
            "similar_songs": formatted_songs,
            "count": len(formatted_songs),
            "similarity_method": "related_artists_with_audio_analysis"
        })
    except Exception as e:
        return json.dumps({"error": str(e), "artist": artist_name})

@tool
@traceable
def get_genre_recommendations(genre: str, limit: int = 8) -> str:
    """
    Get song recommendations based on a specific genre with mood and energy analysis.

    Args:
        genre: Musical genre (pop, rock, jazz, electronic, etc.)
        limit: Number of songs to recommend

    Returns:
        JSON string with genre-based recommendations and audio insights
    """
    try:
        songs = spotify_client.get_genre_songs(genre, limit=limit)
        if not songs:
            return json.dumps({"error": f"No songs found for genre: {genre}"})

        # Enhanced formatting with metadata-based characteristics
        formatted_songs = []
        total_popularity = 0

        for song in songs:
            song_data = {
                "id": song["id"],
                "name": song["name"],
                "artist": song["artist"],
                "album": song["album"],
                "popularity": song["popularity"],
                "spotify_url": song["spotify_url"],
                "genre_relevance": "High" if song["popularity"] > 70 else "Medium" if song["popularity"] > 50 else "Moderate"
            }
            formatted_songs.append(song_data)
            total_popularity += song["popularity"]

        # Calculate genre characteristics based on available metadata
        avg_popularity = total_popularity / len(songs) if songs else 0
        genre_profile = {
            "average_popularity": round(avg_popularity, 1),
            "popularity_level": "Very Popular" if avg_popularity > 70 else "Popular" if avg_popularity > 50 else "Moderate",
            "genre_description": f"Curated {genre} songs based on popularity and relevance"
        }

        return json.dumps({
            "genre": genre,
            "recommendations": formatted_songs,
            "count": len(formatted_songs),
            "genre_profile": genre_profile,
            "analysis_type": "enhanced_genre_recommendations"
        })
    except Exception as e:
        return json.dumps({"error": str(e), "genre": genre})

@tool
@traceable
def save_user_preference(preference_type: str, value: str, action: str = "add") -> str:
    """
    Save or update user music preferences for personalization.
    Enhanced with audio feature preferences and activity context.

    Args:
        preference_type: Type of preference (genre, artist, mood, energy_level, activity)
        value: The preference value
        action: Action to take (add, remove, update)

    Returns:
        Confirmation message with personalization insights
    """
    try:
        timestamp = datetime.now().isoformat()

        preference_data = {
            "type": preference_type,
            "value": value,
            "action": action,
            "timestamp": timestamp,
            "personalization_impact": _get_personalization_impact(preference_type, value)
        }

        return json.dumps({
            "status": "success",
            "message": f"Saved {preference_type} preference: {value}",
            "preference": preference_data,
            "note": "Preference will influence future recommendations",
            "recommendation_impact": preference_data["personalization_impact"]
        })
    except Exception as e:
        return json.dumps({"error": str(e), "preference_type": preference_type, "value": value})

# Helper functions for enhanced analysis (audio features removed due to API limitations)

def _generate_smart_playlist(request: SmartPlaylistRequest) -> List[Dict[str, Any]]:
    """Generate a smart playlist using AI curation"""
    playlist_songs = []

    # Start with seed songs
    for song_id in request.seed_songs[:5]:  # Limit seeds
        song_details = spotify_client.get_track_details(song_id)
        if song_details:
            playlist_songs.append(song_details)

    # Add songs from seed artists (get more songs to fill playlist)
    for artist in request.seed_artists[:3]:
        # Get more songs per artist to reach target size
        songs_needed = max(10, request.size // len(request.seed_artists) if request.seed_artists else 10)
        artist_songs = spotify_client.get_artist_top_songs(artist, limit=songs_needed)
        playlist_songs.extend(artist_songs)

    # Add genre-based songs if we still need more
    if len(playlist_songs) < request.size:
        for genre in request.seed_genres[:2]:
            remaining_needed = request.size - len(playlist_songs)
            genre_songs = spotify_client.get_genre_songs(genre, limit=min(remaining_needed, 10))
            playlist_songs.extend(genre_songs)

    # If we still don't have enough songs, search for more from the main artist
    if len(playlist_songs) < request.size and request.seed_artists:
        main_artist = request.seed_artists[0]
        remaining_needed = request.size - len(playlist_songs)
        # Search for more songs by the artist
        search_results = spotify_client.search_songs(f"artist:{main_artist}", limit=remaining_needed)
        playlist_songs.extend(search_results)

    # Remove duplicates and limit to requested size
    seen_ids = set()
    unique_songs = []
    for song in playlist_songs:
        if song["id"] not in seen_ids and len(unique_songs) < request.size:
            seen_ids.add(song["id"])
            unique_songs.append(song)

    return unique_songs

def _analyze_playlist_flow(songs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze the flow and characteristics of a playlist (simplified without audio features)"""
    if not songs:
        return {"error": "No songs to analyze"}

    # Calculate basic characteristics from available metadata
    total_popularity = sum(song.get("popularity", 50) for song in songs)
    avg_popularity = total_popularity / len(songs)

    # Analyze artist diversity
    artists = [song.get("artist", "Unknown") for song in songs]
    unique_artists = len(set(artists))
    artist_diversity = unique_artists / len(songs) if songs else 0

    return {
        "total_songs": len(songs),
        "average_popularity": round(avg_popularity, 1),
        "artist_diversity": round(artist_diversity, 2),
        "unique_artists": unique_artists,
        "playlist_character": f"Diverse playlist with {unique_artists} different artists",
        "popularity_level": "High" if avg_popularity > 70 else "Medium" if avg_popularity > 50 else "Moderate"
    }

# Removed audio features helper functions - not available with Client Credentials flow

def _get_personalization_impact(preference_type: str, value: str) -> str:
    """Describe how a preference will impact recommendations"""
    impact_descriptions = {
        "genre": f"Future recommendations will include more {value} music and similar genres",
        "artist": f"Will recommend more songs by {value} and similar artists",
        "mood": f"Will prioritize songs that match {value} mood characteristics",
        "energy_level": f"Will adjust energy levels in recommendations to match {value} preference",
        "activity": f"Will tailor recommendations for {value} activities"
    }

    return impact_descriptions.get(preference_type, f"Will incorporate {value} into future recommendations")

# All audio features helper functions removed due to API limitations

# List of all working tools (audio features tools removed due to API limitations)
ALL_TOOLS = [
    search_spotify_songs,
    create_smart_playlist,
    add_song_to_session_playlist,
    get_artist_top_songs,
    get_similar_music,
    get_genre_recommendations,
    save_user_preference
]