from typing import List, Dict, Optional
from pydantic import BaseModel, Field

class SongInfo(BaseModel):
    """Individual song information"""
    id: str = Field(..., description="Spotify track ID")
    name: str = Field(..., description="Song name")
    artist: str = Field(..., description="Artist name(s)")
    album: Optional[str] = Field("", description="Album name")
    popularity: Optional[int] = Field(0, description="Popularity score (0-100)")
    spotify_url: Optional[str] = Field("", description="Spotify URL")
    duration: Optional[str] = Field("", description="Duration in MM:SS format")
    preview_url: Optional[str] = Field(None, description="Preview URL if available")
    album_image_url: Optional[str] = Field(None, description="Album artwork URL (640x640)")
    album_image_small: Optional[str] = Field(None, description="Small album artwork URL (64x64)")

class MusicResponse(BaseModel):
    """Main response from the music concierge"""
    message: str = Field(..., description="Human-readable response message")
    songs: List[SongInfo] = Field(default=[], description="List of songs if any")
    response_type: str = Field("general", description="Type of response (search, playlist, recommendation, etc.)")
    metadata: Optional[Dict] = Field(default={}, description="Additional metadata")

class SmartPlaylistRequest(BaseModel):
    """Request to create a smart playlist"""
    name: str = Field(..., description="Playlist name")
    description: str = Field(..., description="Playlist description/theme")
    seed_artists: List[str] = Field(default=[], description="Artists to base recommendations on")
    seed_genres: List[str] = Field(default=[], description="Genres to include")
    size: int = Field(default=20, ge=5, le=100, description="Number of songs")
    diversity_factor: float = Field(default=0.5, ge=0.0, le=1.0, description="Playlist diversity")