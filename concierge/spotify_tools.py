"""
Spotify Tools for Music Concierge Agent

Professional music tools with structured outputs using Pydantic models.
All tools follow LangChain best practices for reliable agent integration.
"""
import os
import json
import random
from typing import List, Optional
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field
from .client import WorkingSpotifyClient
from . import config

# Pydantic models for structured outputs
class SpotifyTrackData(BaseModel):
    """Individual track information."""
    id: str = Field(description="Spotify track ID")
    name: str = Field(description="Song name")
    artist: str = Field(description="Artist name(s)")
    album: str = Field(description="Album name")
    popularity: int = Field(description="Popularity score (0-100)")
    duration: str = Field(description="Duration in MM:SS format")
    spotify_url: str = Field(description="Spotify URL")
    preview_url: Optional[str] = Field(description="Preview URL if available")
    album_image_url: Optional[str] = Field(description="Album artwork URL")
    formatted_summary: str = Field(description="Human-readable summary")
    error: Optional[str] = None

class TrackSearchResult(BaseModel):
    """Search results for tracks."""
    query: str = Field(description="Original search query")
    total_results: int = Field(description="Number of tracks found")
    tracks: List[SpotifyTrackData] = Field(description="List of tracks")
    formatted_summary: str = Field(description="Human-readable summary")
    error: Optional[str] = None

class ArtistTopSongsResult(BaseModel):
    """Top songs by an artist."""
    artist_name: str = Field(description="Artist name searched")
    total_songs: int = Field(description="Number of songs found")
    songs: List[SpotifyTrackData] = Field(description="List of top songs")
    formatted_summary: str = Field(description="Human-readable summary")
    error: Optional[str] = None

class SmartPlaylistResult(BaseModel):
    """Smart playlist creation result."""
    playlist_name: str = Field(description="Generated playlist name")
    description: str = Field(description="Playlist description")
    total_songs: int = Field(description="Number of songs in playlist")
    songs: List[SpotifyTrackData] = Field(description="List of songs")
    seed_artists: List[str] = Field(description="Artists used as seeds")
    seed_genres: List[str] = Field(description="Genres used as seeds")
    diversity_score: float = Field(description="Playlist diversity score")
    formatted_summary: str = Field(description="Human-readable summary")
    error: Optional[str] = None

class GenreSongsResult(BaseModel):
    """Songs from a specific genre."""
    genre: str = Field(description="Genre searched")
    total_songs: int = Field(description="Number of songs found")
    songs: List[SpotifyTrackData] = Field(description="List of genre songs")
    formatted_summary: str = Field(description="Human-readable summary")
    error: Optional[str] = None

# Initialize Spotify client
_spotify_client = None

def get_spotify_client() -> WorkingSpotifyClient:
    """Get or create Spotify client instance."""
    global _spotify_client
    if _spotify_client is None:
        _spotify_client = WorkingSpotifyClient(
            config.SPOTIFY_CLIENT_ID,
            config.SPOTIFY_CLIENT_SECRET
        )
    return _spotify_client

def _format_track_data(track_dict: dict) -> SpotifyTrackData:
    """Convert track dictionary to SpotifyTrackData model."""
    formatted_summary = f"{track_dict['name']} by {track_dict['artist']} | {track_dict['album']} | Popularity: {track_dict['popularity']}/100"

    return SpotifyTrackData(
        id=track_dict['id'],
        name=track_dict['name'],
        artist=track_dict['artist'],
        album=track_dict['album'],
        popularity=track_dict['popularity'],
        duration=track_dict['duration'],
        spotify_url=track_dict['spotify_url'],
        preview_url=track_dict.get('preview_url'),
        album_image_url=track_dict.get('album_image_url'),
        formatted_summary=formatted_summary
    )

@tool
def search_tracks(query: str, limit: int = 10) -> TrackSearchResult:
    """
    Search for tracks on Spotify.

    Args:
        query: Search query (song name, artist, etc.)
        limit: Number of results to return (default: 10, max: 50)

    Returns:
        Structured track search results with metadata
    """
    try:
        spotify = get_spotify_client()
        raw_results = spotify.search_songs(query, limit=min(limit, 50))

        if not raw_results:
            return TrackSearchResult(
                query=query,
                total_results=0,
                tracks=[],
                formatted_summary=f"No tracks found for query: {query}",
                error="No results found"
            )

        tracks = [_format_track_data(track) for track in raw_results if 'error' not in track]

        formatted_summary = f"Found {len(tracks)} tracks for '{query}'"
        if tracks:
            top_track = tracks[0]
            formatted_summary += f" | Top result: {top_track.name} by {top_track.artist}"

        return TrackSearchResult(
            query=query,
            total_results=len(tracks),
            tracks=tracks,
            formatted_summary=formatted_summary
        )

    except Exception as e:
        return TrackSearchResult(
            query=query,
            total_results=0,
            tracks=[],
            formatted_summary=f"Search failed for '{query}': {str(e)}",
            error=str(e)
        )

@tool
def get_artist_top_songs(artist_name: str, limit: int = 10) -> ArtistTopSongsResult:
    """
    Get top songs by an artist.

    Args:
        artist_name: Name of the artist
        limit: Number of songs to return (default: 10, max: 50)

    Returns:
        Structured artist top songs with popularity and metadata
    """
    try:
        spotify = get_spotify_client()
        raw_results = spotify.get_artist_top_songs(artist_name, limit=min(limit, 50))

        if not raw_results:
            return ArtistTopSongsResult(
                artist_name=artist_name,
                total_songs=0,
                songs=[],
                formatted_summary=f"No top songs found for {artist_name}",
                error="Artist not found"
            )

        songs = [_format_track_data(song) for song in raw_results if 'error' not in song]

        formatted_summary = f"{artist_name} top {len(songs)} songs"
        if songs:
            avg_popularity = sum(song.popularity for song in songs) / len(songs)
            formatted_summary += f" | Avg popularity: {avg_popularity:.1f}/100"

        return ArtistTopSongsResult(
            artist_name=artist_name,
            total_songs=len(songs),
            songs=songs,
            formatted_summary=formatted_summary
        )

    except Exception as e:
        return ArtistTopSongsResult(
            artist_name=artist_name,
            total_songs=0,
            songs=[],
            formatted_summary=f"Failed to get top songs for {artist_name}: {str(e)}",
            error=str(e)
        )

@tool
def get_similar_songs(artist_name: str, limit: int = 10) -> ArtistTopSongsResult:
    """
    Get songs similar to an artist's style.

    Args:
        artist_name: Name of the artist to find similar music to
        limit: Number of songs to return (default: 10, max: 50)

    Returns:
        Structured similar songs results
    """
    try:
        spotify = get_spotify_client()
        raw_results = spotify.get_similar_songs(artist_name, limit=min(limit, 50))

        if not raw_results:
            return ArtistTopSongsResult(
                artist_name=artist_name,
                total_songs=0,
                songs=[],
                formatted_summary=f"No similar songs found for {artist_name}",
                error="No similar artists found"
            )

        songs = [_format_track_data(song) for song in raw_results if 'error' not in song]

        # Get unique artists for diversity metric
        unique_artists = set(song.artist.split(', ')[0] for song in songs)
        diversity_score = len(unique_artists) / len(songs) if songs else 0

        formatted_summary = f"Similar to {artist_name}: {len(songs)} songs from {len(unique_artists)} artists | Diversity: {diversity_score:.2f}"

        return ArtistTopSongsResult(
            artist_name=f"Similar to {artist_name}",
            total_songs=len(songs),
            songs=songs,
            formatted_summary=formatted_summary
        )

    except Exception as e:
        return ArtistTopSongsResult(
            artist_name=artist_name,
            total_songs=0,
            songs=[],
            formatted_summary=f"Failed to get similar songs for {artist_name}: {str(e)}",
            error=str(e)
        )

@tool
def get_genre_songs(genre: str, limit: int = 10) -> GenreSongsResult:
    """
    Get songs from a specific genre.

    Args:
        genre: Genre name (e.g., "pop", "rock", "hip-hop", "electronic")
        limit: Number of songs to return (default: 10, max: 50)

    Returns:
        Structured genre songs with diversity metrics
    """
    try:
        spotify = get_spotify_client()
        raw_results = spotify.get_genre_songs(genre, limit=min(limit, 50))

        if not raw_results:
            return GenreSongsResult(
                genre=genre,
                total_songs=0,
                songs=[],
                formatted_summary=f"No songs found for genre: {genre}",
                error="Genre not found"
            )

        songs = [_format_track_data(song) for song in raw_results if 'error' not in song]

        if songs:
            avg_popularity = sum(song.popularity for song in songs) / len(songs)
            unique_artists = set(song.artist.split(', ')[0] for song in songs)
            formatted_summary = f"{genre.title()} genre: {len(songs)} songs from {len(unique_artists)} artists | Avg popularity: {avg_popularity:.1f}/100"
        else:
            formatted_summary = f"No valid {genre} songs found"

        return GenreSongsResult(
            genre=genre,
            total_songs=len(songs),
            songs=songs,
            formatted_summary=formatted_summary
        )

    except Exception as e:
        return GenreSongsResult(
            genre=genre,
            total_songs=0,
            songs=[],
            formatted_summary=f"Failed to get {genre} songs: {str(e)}",
            error=str(e)
        )

@tool
def create_smart_playlist(query: str) -> SmartPlaylistResult:
    """
    Create a smart playlist based on criteria.

    Args:
        query: JSON string with playlist parameters:
               {"name": "playlist name", "description": "description",
                "seed_artists": ["artist1"], "seed_genres": ["genre1"],
                "size": 20}

    Returns:
        Structured smart playlist with songs and metadata
    """
    try:
        data = json.loads(query)

        playlist_name = data.get("name", "Custom Playlist")
        description = data.get("description", "AI-curated playlist")
        seed_artists = data.get("seed_artists", [])
        seed_genres = data.get("seed_genres", [])
        size = min(data.get("size", 20), 50)

        spotify = get_spotify_client()
        all_songs = []

        # Get songs from seed artists
        for artist in seed_artists[:3]:  # Limit to 3 artists
            artist_songs = spotify.get_artist_top_songs(artist, limit=5)
            all_songs.extend([song for song in artist_songs if 'error' not in song])

        # Get songs from seed genres
        for genre in seed_genres[:2]:  # Limit to 2 genres
            genre_songs = spotify.get_genre_songs(genre, limit=8)
            all_songs.extend([song for song in genre_songs if 'error' not in song])

        # Remove duplicates and shuffle
        seen_ids = set()
        unique_songs = []
        for song in all_songs:
            if song['id'] not in seen_ids:
                seen_ids.add(song['id'])
                unique_songs.append(song)

        random.shuffle(unique_songs)
        playlist_songs = unique_songs[:size]

        # Convert to SpotifyTrackData models
        formatted_songs = [_format_track_data(song) for song in playlist_songs]

        # Calculate diversity
        unique_artists = set(song.artist.split(', ')[0] for song in formatted_songs)
        diversity_score = len(unique_artists) / len(formatted_songs) if formatted_songs else 0

        formatted_summary = f"'{playlist_name}': {len(formatted_songs)} songs | {len(unique_artists)} artists | Diversity: {diversity_score:.2f}"

        return SmartPlaylistResult(
            playlist_name=playlist_name,
            description=description,
            total_songs=len(formatted_songs),
            songs=formatted_songs,
            seed_artists=seed_artists,
            seed_genres=seed_genres,
            diversity_score=diversity_score,
            formatted_summary=formatted_summary
        )

    except Exception as e:
        return SmartPlaylistResult(
            playlist_name="Error",
            description="Failed to create playlist",
            total_songs=0,
            songs=[],
            seed_artists=[],
            seed_genres=[],
            diversity_score=0.0,
            formatted_summary=f"Playlist creation failed: {str(e)}",
            error=str(e)
        )

# Initialize Tavily search tool
try:
    tavily_search = TavilySearch(
        api_key=config.TAVILY_API_KEY,
        max_results=5,
        search_depth="advanced",
        include_answer=True,
        include_raw_content=False,
        include_images=False
    )
    SPOTIFY_TOOLS = [
        search_tracks,
        get_artist_top_songs,
        get_similar_songs,
        get_genre_songs,
        create_smart_playlist,
        tavily_search
    ]
except Exception as e:
    print(f"Warning: Tavily search not available: {e}")
    SPOTIFY_TOOLS = [
        search_tracks,
        get_artist_top_songs,
        get_similar_songs,
        get_genre_songs,
        create_smart_playlist
    ]

print(f"Loaded {len(SPOTIFY_TOOLS)} Spotify tools")