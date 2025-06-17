import requests
import base64
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import random

class WorkingSpotifyClient:
    """Spotify client that works with current API limitations"""

    def __init__(self, client_id: str, client_secret: str):
        """Initialize with credentials from environment variables"""
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token = None
        self.token_expires_at = None
        self.base_url = "https://api.spotify.com/v1"
        self._get_access_token()

    def _get_access_token(self) -> bool:
        """Get access token using client credentials flow"""
        url = "https://accounts.spotify.com/api/token"
        credentials = f"{self.client_id}:{self.client_secret}"
        encoded_credentials = base64.b64encode(credentials.encode()).decode()

        headers = {
            'Authorization': f'Basic {encoded_credentials}',
            'Content-Type': 'application/x-www-form-urlencoded'
        }

        try:
            response = requests.post(url, headers=headers, data={'grant_type': 'client_credentials'})
            if response.status_code == 200:
                token_data = response.json()
                self.access_token = token_data['access_token']
                expires_in = token_data.get('expires_in', 3600)
                self.token_expires_at = datetime.now() + timedelta(seconds=expires_in)
                return True
            else:
                raise Exception(f"Error getting token: {response.status_code}")
        except Exception as e:
            raise Exception(f"Connection error: {e}")

    def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Make authenticated request to Spotify API"""
        if not self.access_token or datetime.now() >= self.token_expires_at:
            if not self._get_access_token():
                return None

        headers = {'Authorization': f'Bearer {self.access_token}'}
        url = f"{self.base_url}{endpoint}"

        try:
            response = requests.get(url, headers=headers, params=params)
            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"API request failed: {response.status_code}")
        except Exception as e:
            raise Exception(f"Request error: {e}")

    def search_songs(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for songs"""
        params = {'q': query, 'type': 'track', 'limit': min(limit, 50)}
        result = self._make_request('/search', params)

        if result and 'tracks' in result:
            return [self._format_track(track) for track in result['tracks']['items']]
        return []

    def get_artist_top_songs(self, artist_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top songs by a specific artist"""
        artist_id = self._get_artist_id(artist_name)
        if not artist_id:
            return []

        result = self._make_request(f'/artists/{artist_id}/top-tracks', {'market': 'US'})
        if result and 'tracks' in result:
            return [self._format_track(track) for track in result['tracks'][:limit]]
        return []

    def get_similar_songs(self, artist_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get similar songs using related artists"""
        artist_id = self._get_artist_id(artist_name)
        if not artist_id:
            return []

        # Get related artists
        related_result = self._make_request(f'/artists/{artist_id}/related-artists')
        if not related_result or 'artists' not in related_result:
            return []

        similar_songs = []
        for related_artist in related_result['artists'][:5]:
            tracks_result = self._make_request(f'/artists/{related_artist["id"]}/top-tracks', {'market': 'US'})
            if tracks_result and 'tracks' in tracks_result:
                for track in tracks_result['tracks'][:2]:
                    similar_songs.append(self._format_track(track))

        random.shuffle(similar_songs)
        return similar_songs[:limit]

    def get_genre_songs(self, genre: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get songs by genre using search"""
        # Search for songs with genre keywords
        search_queries = [
            f'genre:"{genre}"',
            f'{genre} music',
            f'style:{genre}',
            f'{genre} songs'
        ]

        all_songs = []
        for query in search_queries:
            songs = self.search_songs(query, limit=5)
            all_songs.extend(songs)

        # Remove duplicates
        seen_ids = set()
        unique_songs = []
        for song in all_songs:
            if song['id'] not in seen_ids:
                seen_ids.add(song['id'])
                unique_songs.append(song)

        random.shuffle(unique_songs)
        return unique_songs[:limit]

    def get_featured_playlists(self) -> List[Dict[str, Any]]:
        """Get featured playlists"""
        result = self._make_request('/browse/featured-playlists', {'limit': 10, 'country': 'US'})
        if result and 'playlists' in result:
            playlists = []
            for playlist in result['playlists']['items']:
                playlists.append({
                    'name': playlist['name'],
                    'description': playlist.get('description', ''),
                    'tracks_total': playlist['tracks']['total'],
                    'spotify_url': playlist['external_urls']['spotify'],
                    'id': playlist['id']
                })
            return playlists
        return []

    def get_playlist_tracks(self, playlist_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get tracks from a playlist"""
        result = self._make_request(f'/playlists/{playlist_id}/tracks', {'limit': limit, 'market': 'US'})
        if result and 'items' in result:
            tracks = []
            for item in result['items']:
                if item.get('track') and item['track'].get('type') == 'track':
                    tracks.append(self._format_track(item['track']))
            return tracks
        return []

    def _get_artist_id(self, artist_name: str) -> Optional[str]:
        """Get Spotify artist ID by name"""
        result = self._make_request('/search', {'q': artist_name, 'type': 'artist', 'limit': 1})
        if result and 'artists' in result and result['artists']['items']:
            return result['artists']['items'][0]['id']
        return None

    def _format_track(self, track: Dict[str, Any]) -> Dict[str, Any]:
        """Format track data consistently"""
        # Extract album images
        album_images = track.get('album', {}).get('images', [])
        album_image_url = None
        album_image_small = None

        # Get the best image sizes (Spotify typically returns 640x640, 300x300, 64x64)
        for image in album_images:
            if image.get('height') == 640:  # Large image
                album_image_url = image.get('url')
            elif image.get('height') == 64:  # Small image
                album_image_small = image.get('url')

        # If no exact sizes found, use first available as large, last as small
        if not album_image_url and album_images:
            album_image_url = album_images[0].get('url')
        if not album_image_small and len(album_images) > 1:
            album_image_small = album_images[-1].get('url')
        elif not album_image_small and album_images:
            album_image_small = album_images[0].get('url')

        return {
            'id': track['id'],
            'name': track['name'],
            'artist': ', '.join([artist['name'] for artist in track['artists']]),
            'album': track['album']['name'],
            'duration': self._format_duration(track['duration_ms']),
            'popularity': track['popularity'],
            'spotify_url': track['external_urls']['spotify'],
            'preview_url': track.get('preview_url'),
            'album_image_url': album_image_url,
            'album_image_small': album_image_small
        }

    def _format_duration(self, duration_ms: int) -> str:
        """Convert milliseconds to MM:SS format"""
        seconds = duration_ms // 1000
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{minutes}:{seconds:02d}"