# 🎵 Music Concierge Agent - LangChain Spotify Integration

A sophisticated AI-powered music concierge built with LangChain that provides intelligent music recommendations, playlist curation, and Spotify integration. The agent uses advanced tools to search, analyze, and recommend music based on user preferences and conversation context.

## 🚀 Features

### Core Capabilities
- **Intelligent Music Search** - Semantic search across Spotify's catalog
- **Smart Playlist Creation** - AI-curated playlists based on mood, genre, and preferences
- **Artist Discovery** - Top songs, similar artists, and related music recommendations
- **Genre Exploration** - Discover music by genre with relevance scoring
- **Session-Based Playlists** - Track songs mentioned during conversation
- **User Preference Learning** - Save and remember music preferences
- **Conversational Interface** - Natural language music discovery

### Technical Features
- **LangSmith Integration** - Full tracing and observability
- **Pydantic v2 Models** - Structured data validation and output
- **Error Handling** - Robust API error management
- **Memory Management** - Conversation context and user preferences
- **Streamlit UI** - Interactive web interface

## 🛠️ Architecture

### Agent (`agent.py`)
- **LangChain Agent** with OpenAI GPT-4o-mini
- **ConversationSummaryBufferMemory** for context retention
- **Tool Integration** with 7 specialized music tools
- **LangSmith Tracing** for debugging and optimization

### Tools (`tools.py`)
Seven specialized tools for music operations:

1. **`search_spotify_songs`** - Search Spotify catalog
2. **`create_smart_playlist`** - AI-curated playlist generation
3. **`add_song_to_session_playlist`** - Session-based playlist building
4. **`get_artist_top_songs`** - Popular songs by artist
5. **`get_similar_music`** - Related artist recommendations
6. **`get_genre_recommendations`** - Genre-based discovery
7. **`save_user_preference`** - Personalization system

### Spotify Client (`spotify/spotify_working.py`)
- **Client Credentials Flow** for API authentication
- **Rate Limiting** and error handling
- **Comprehensive API Coverage** - search, artists, tracks, genres
- **Data Formatting** - Consistent output structures

## 📋 Requirements

```txt
langchain==0.3.7
langchain-openai==0.2.8
langchain-community==0.3.5
langsmith==0.1.143
pydantic==2.10.3
requests==2.32.3
python-dotenv==1.0.1
streamlit==1.40.1
```

## 🔧 Setup

### 1. Environment Variables
Create a `.env` file:
```env
OPENAI_API_KEY=your_openai_api_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=music-concierge-agent
```

### 2. Spotify API Credentials
The Spotify client uses embedded credentials for demo purposes. For production:
- Register at [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
- Update credentials in `spotify/spotify_working.py`

### 3. Installation
```bash
pip install -r requirements.txt
```

## 🎯 Usage

### Command Line Interface
```bash
python demo_script.py
```

### Streamlit Web Interface
```bash
streamlit run streamlit_app.py
```

### Programmatic Usage
```python
from agent import MusicConciergeAgent

agent = MusicConciergeAgent(api_key="your_openai_key")
response = agent.chat("Create me an upbeat Taylor Swift playlist")
print(response)
```

## 🎼 Tool Details

### 1. Search Spotify Songs
```python
search_spotify_songs(query="Taylor Swift Love Story", limit=5)
```
- Searches Spotify catalog with enhanced metadata
- Returns formatted song data with popularity insights
- Supports complex queries and filters

### 2. Create Smart Playlist
```python
create_smart_playlist(request_data=json.dumps({
    "name": "Upbeat Workout Mix",
    "description": "High-energy songs for exercise",
    "seed_artists": ["Dua Lipa", "The Weeknd"],
    "seed_genres": ["pop", "dance"],
    "size": 20
}))
```
- AI-curated playlists with flow analysis
- Supports multiple seed types (artists, genres, songs)
- Generates diverse, well-balanced playlists

### 3. Session Playlist Management
```python
add_song_to_session_playlist(song_id="track_id", user_reaction="liked")
```
- Tracks songs mentioned during conversation
- Builds context-aware playlists
- Supports user reaction tracking

### 4. Artist Top Songs
```python
get_artist_top_songs(artist_name="Taylor Swift", limit=10)
```
- Returns most popular songs by artist
- Enhanced with popularity levels and metadata
- Market-specific results (US)

### 5. Similar Music Discovery
```python
get_similar_music(artist_name="Green Day", limit=8)
```
- Uses related artists for recommendations
- Metadata-based similarity scoring
- Diverse artist coverage

### 6. Genre Recommendations
```python
get_genre_recommendations(genre="rock", limit=10)
```
- Genre-based music discovery
- Relevance scoring and popularity analysis
- Multiple search strategies for comprehensive results

### 7. User Preference Management
```python
save_user_preference(preference_type="artist", value="Taylor Swift", action="add")
```
- Persistent preference storage
- Personalization impact analysis
- Multiple preference types (genre, artist, mood, energy)

## 🎨 Data Models

### Core Models (Pydantic v2)
- **`MusicPreferences`** - User preference structure
- **`SmartPlaylistRequest`** - Playlist creation parameters
- **`SessionPlaylist`** - Conversation-based playlist tracking
- **`MusicRecommendation`** - Structured recommendation output

### Example Output
```json
{
  "playlist_name": "Upbeat Taylor Swift Playlist",
  "songs": [
    {
      "name": "Cruel Summer",
      "artist": "Taylor Swift",
      "album": "Lover",
      "popularity": 91,
      "spotify_url": "https://open.spotify.com/track/1BxfuPKGuaTgP7aM0Bbdwr",
      "duration": "2:58"
    }
  ],
  "flow_analysis": {
    "average_popularity": 88.0,
    "artist_diversity": 0.67,
    "playlist_character": "High-energy diverse mix"
  }
}
```

## 🔍 Key Improvements Made

### Tool Optimization
- **Removed non-working tools** that required user authorization
- **Enhanced error handling** for API limitations
- **Improved playlist generation** with better song selection algorithms
- **Added comprehensive metadata** analysis

### Agent Enhancement
- **Clear tool usage guidelines** to prevent wrong tool selection
- **Improved prompting** for better playlist vs. search distinction
- **Enhanced memory management** for conversation context
- **Cost optimization** with GPT-4o-mini model

### Code Quality
- **Pydantic v2 patterns** with proper validators
- **LangChain best practices** following official documentation
- **Comprehensive error handling** and logging
- **Clean separation of concerns** between agent, tools, and client

## ⚠️ Current Limitations

### Playlist Creation
- **Simulation Only** - Creates playlist recommendations, not actual Spotify playlists
- **Client Credentials Flow** - Read-only access, cannot modify user accounts
- **No User Authentication** - Cannot access personal Spotify data

### API Constraints
- **No Audio Features** - Advanced audio analysis not available with current auth
- **Rate Limiting** - Spotify API rate limits apply
- **Market Restrictions** - Some content may be region-locked

## 🚀 Future Enhancements

### Real Playlist Creation
- Implement OAuth flow for user authorization
- Add playlist creation and modification endpoints
- Enable saving playlists to user's Spotify account

### Advanced Features
- Audio feature analysis with user auth
- Collaborative filtering recommendations
- Social features and playlist sharing
- Integration with other music services

### UI/UX Improvements
- Enhanced Streamlit interface
- Mobile-responsive design
- Playlist visualization and analytics
- Export functionality (M3U, CSV)

## 📊 Testing

### Tool Testing
All 7 tools have been thoroughly tested:
- ✅ API connectivity and authentication
- ✅ Error handling and edge cases
- ✅ Output format validation
- ✅ Performance and rate limiting

### Agent Testing
- ✅ Tool selection accuracy
- ✅ Conversation flow and memory
- ✅ Response quality and formatting
- ✅ LangSmith tracing integration

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is for educational and demonstration purposes. Spotify API usage subject to Spotify's terms of service.

## 🙏 Acknowledgments

- **LangChain** for the agent framework
- **Spotify Web API** for music data
- **OpenAI** for language model capabilities
- **LangSmith** for observability and tracing

---

Built with ❤️ for music lovers and AI enthusiasts
