"""
Working Spotify Song Getter - Avoids deprecated endpoints
Uses alternative methods for recommendations since the official endpoint is deprecated
"""

import requests
import base64
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import random

class WorkingSpotifyClient:
    """Spotify client that works with current API limitations"""

    def __init__(self):
        self.client_id = "bf73bea92a5a453f9669976d43ece405"
        self.client_secret = "8b74c43103bb435a90c269e265f72346"
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
                print("✅ Successfully connected to Spotify!")
                return True
            else:
                print(f"❌ Error getting token: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False

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
                print(f"❌ API request failed: {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ Request error: {e}")
            return None

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

    def get_audio_features(self, track_id: str) -> Optional[Dict[str, Any]]:
        """Get audio features for a specific track"""
        result = self._make_request(f'/audio-features/{track_id}')
        if result and result.get('id'):
            return {
                'danceability': result.get('danceability', 0.0),
                'energy': result.get('energy', 0.0),
                'valence': result.get('valence', 0.0),
                'tempo': result.get('tempo', 120.0),
                'acousticness': result.get('acousticness', 0.0),
                'instrumentalness': result.get('instrumentalness', 0.0),
                'liveness': result.get('liveness', 0.0),
                'speechiness': result.get('speechiness', 0.0),
                'loudness': result.get('loudness', -10.0),
                'key': result.get('key', 0),
                'mode': result.get('mode', 1),
                'time_signature': result.get('time_signature', 4)
            }
        return None

    def get_track_details(self, track_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific track"""
        result = self._make_request(f'/tracks/{track_id}', {'market': 'US'})
        if result and result.get('id'):
            formatted_track = self._format_track(result)
            # Add additional details
            formatted_track.update({
                'release_date': result['album'].get('release_date', 'Unknown'),
                'explicit': result.get('explicit', False),
                'track_number': result.get('track_number', 1),
                'disc_number': result.get('disc_number', 1),
                'album_type': result['album'].get('album_type', 'album'),
                'album_image': result['album']['images'][0]['url'] if result['album'].get('images') else None
            })
            return formatted_track
        return None

    def get_multiple_audio_features(self, track_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get audio features for multiple tracks at once"""
        if not track_ids:
            return {}

        # Spotify API allows up to 100 IDs at once
        chunk_size = 100
        all_features = {}

        for i in range(0, len(track_ids), chunk_size):
            chunk = track_ids[i:i + chunk_size]
            ids_param = ','.join(chunk)

            result = self._make_request('/audio-features', {'ids': ids_param})
            if result and 'audio_features' in result:
                for features in result['audio_features']:
                    if features:  # Some tracks might not have features
                        track_id = features['id']
                        all_features[track_id] = {
                            'danceability': features.get('danceability', 0.0),
                            'energy': features.get('energy', 0.0),
                            'valence': features.get('valence', 0.0),
                            'tempo': features.get('tempo', 120.0),
                            'acousticness': features.get('acousticness', 0.0),
                            'instrumentalness': features.get('instrumentalness', 0.0),
                            'liveness': features.get('liveness', 0.0),
                            'speechiness': features.get('speechiness', 0.0),
                            'loudness': features.get('loudness', -10.0),
                            'key': features.get('key', 0),
                            'mode': features.get('mode', 1),
                            'time_signature': features.get('time_signature', 4)
                        }

        return all_features

    def get_artist_details(self, artist_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about an artist"""
        artist_id = self._get_artist_id(artist_name)
        if not artist_id:
            return None

        result = self._make_request(f'/artists/{artist_id}')
        if result and result.get('id'):
            return {
                'id': result['id'],
                'name': result['name'],
                'genres': result.get('genres', []),
                'popularity': result.get('popularity', 0),
                'followers': result.get('followers', {}).get('total', 0),
                'spotify_url': result['external_urls']['spotify'],
                'images': result.get('images', [])
            }
        return None

    def search_with_filters(self, query: str, search_type: str = 'track',
                          market: str = 'US', limit: int = 10,
                          offset: int = 0) -> List[Dict[str, Any]]:
        """Enhanced search with more filter options"""
        params = {
            'q': query,
            'type': search_type,
            'market': market,
            'limit': min(limit, 50),
            'offset': offset
        }

        result = self._make_request('/search', params)

        if result and f'{search_type}s' in result:
            items = result[f'{search_type}s']['items']
            if search_type == 'track':
                return [self._format_track(item) for item in items]
            elif search_type == 'artist':
                return [self._format_artist(item) for item in items]
            elif search_type == 'album':
                return [self._format_album(item) for item in items]

        return []

    def _format_artist(self, artist: Dict[str, Any]) -> Dict[str, Any]:
        """Format artist data consistently"""
        return {
            'id': artist['id'],
            'name': artist['name'],
            'genres': artist.get('genres', []),
            'popularity': artist.get('popularity', 0),
            'followers': artist.get('followers', {}).get('total', 0),
            'spotify_url': artist['external_urls']['spotify'],
            'images': artist.get('images', [])
        }

    def _format_album(self, album: Dict[str, Any]) -> Dict[str, Any]:
        """Format album data consistently"""
        return {
            'id': album['id'],
            'name': album['name'],
            'artist': ', '.join([artist['name'] for artist in album['artists']]),
            'release_date': album.get('release_date', 'Unknown'),
            'total_tracks': album.get('total_tracks', 0),
            'album_type': album.get('album_type', 'album'),
            'spotify_url': album['external_urls']['spotify'],
            'images': album.get('images', [])
        }

    def _get_artist_id(self, artist_name: str) -> Optional[str]:
        """Get Spotify artist ID by name"""
        result = self._make_request('/search', {'q': artist_name, 'type': 'artist', 'limit': 1})
        if result and 'artists' in result and result['artists']['items']:
            return result['artists']['items'][0]['id']
        return None

    def _format_track(self, track: Dict[str, Any]) -> Dict[str, Any]:
        """Format track data consistently"""
        return {
            'name': track['name'],
            'artist': ', '.join([artist['name'] for artist in track['artists']]),
            'album': track['album']['name'],
            'duration': self._format_duration(track['duration_ms']),
            'popularity': track['popularity'],
            'spotify_url': track['external_urls']['spotify'],
            'preview_url': track.get('preview_url'),
            'id': track['id']
        }

    def _format_duration(self, duration_ms: int) -> str:
        """Convert milliseconds to MM:SS format"""
        seconds = duration_ms // 1000
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{minutes}:{seconds:02d}"

    def print_songs(self, songs: List[Dict[str, Any]], title: str = "Songs"):
        """Pretty print song list"""
        if not songs:
            print("🔍 No songs found.")
            return

        print(f"\n🎵 {title} ({len(songs)} found)")
        print("=" * 60)

        for i, song in enumerate(songs, 1):
            print(f"{i:2d}. {song['name']}")
            print(f"    👤 {song['artist']}")
            print(f"    💽 {song['album']}")
            print(f"    ⏱️  {song['duration']} | 📊 Popularity: {song['popularity']}")
            print(f"    🎧 {song['spotify_url']}")
            if song['preview_url']:
                print(f"    🔊 Preview: {song['preview_url']}")
            print()


def main():
    """Main interactive function"""
    print("🎵 Working Spotify Song Getter")
    print("=" * 50)

    spotify = WorkingSpotifyClient()
    if not spotify.access_token:
        print("❌ Failed to connect to Spotify.")
        return

    while True:
        print("\n🎯 What would you like to do?")
        print("1. 🔍 Search for songs")
        print("2. 🎭 Get songs by genre")
        print("3. ⭐ Get artist's top songs")
        print("4. 🔀 Get similar songs (via related artists)")
        print("5. 📋 Browse featured playlists")
        print("6. 🚪 Exit")

        choice = input("\nEnter your choice (1-6): ").strip()

        if choice == '1':
            query = input("🔍 Search for songs: ").strip()
            if query:
                songs = spotify.search_songs(query, limit=10)
                spotify.print_songs(songs, f"Search results for '{query}'")

        elif choice == '2':
            print("\n🎭 Popular genres: pop, rock, hip-hop, jazz, electronic, country, r&b, folk")
            genre = input("Enter a genre: ").strip()
            if genre:
                songs = spotify.get_genre_songs(genre, limit=10)
                spotify.print_songs(songs, f"'{genre}' songs")

        elif choice == '3':
            artist = input("⭐ Enter artist name: ").strip()
            if artist:
                songs = spotify.get_artist_top_songs(artist, limit=10)
                spotify.print_songs(songs, f"Top songs by '{artist}'")

        elif choice == '4':
            artist = input("🔀 Enter artist name for similar songs: ").strip()
            if artist:
                songs = spotify.get_similar_songs(artist, limit=10)
                spotify.print_songs(songs, f"Similar to '{artist}'")

        elif choice == '5':
            playlists = spotify.get_featured_playlists()
            if playlists:
                print("\n📋 Featured Playlists:")
                for i, playlist in enumerate(playlists, 1):
                    print(f"{i:2d}. {playlist['name']} ({playlist['tracks_total']} tracks)")

                try:
                    choice_idx = int(input("\nSelect playlist (number): ")) - 1
                    if 0 <= choice_idx < len(playlists):
                        tracks = spotify.get_playlist_tracks(playlists[choice_idx]['id'], limit=10)
                        spotify.print_songs(tracks, f"'{playlists[choice_idx]['name']}' tracks")
                except ValueError:
                    print("Invalid selection.")

        elif choice == '6':
            print("👋 Thanks for using Spotify Song Getter!")
            break

        else:
            print("❌ Invalid choice. Please try again.")


if __name__ == "__main__":
    main()