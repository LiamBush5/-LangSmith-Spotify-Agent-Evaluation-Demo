# 🎵 Spotify AI Music Agent

A sophisticated AI-powered music concierge that provides intelligent music recommendations and playlist curation through natural conversation. Built with LangChain agents, modern React frontend, and comprehensive Spotify integration.

## 🎯 Project Goal

Create an intelligent music assistant that understands natural language queries and provides personalized music recommendations with the brevity and style of Spotify's AI DJ - delivering quick, knowledgeable commentary about artists and songs without lengthy explanations.

## 🏗️ Architecture Overview

### Backend: Python LangChain Agent
- **Agent Framework**: LangChain with OpenAI GPT-4o-mini
- **Memory**: Conversation context and user preferences
- **Tools**: 6 specialized Spotify integration tools
- **API**: FastAPI server with structured responses
- **Observability**: LangSmith tracing and evaluation

### Frontend: Modern React Application
- **Framework**: Next.js 15 with TypeScript
- **Styling**: Tailwind CSS with Spotify-inspired design
- **Components**: Clean, responsive song cards and chat interface
- **Real-time**: Live chat with streaming responses
- **Integration**: Direct Spotify URL linking for seamless playback

## 🛠️ Agent Tools

The AI agent uses 6 specialized tools to handle different music queries:

### 1. `get_artist_top_songs`
**Purpose**: Retrieve an artist's most popular tracks
```python
# Example: "Show me Taylor Swift's hits"
get_artist_top_songs(artist_name="Taylor Swift", limit=10)
```
**Returns**: Top songs with popularity scores, album info, and Spotify URLs

### 2. `get_genre_songs`
**Purpose**: Discover music by genre or mood
```python
# Example: "I want some chill indie rock"
get_genre_songs(genre="indie rock", limit=10)
```
**Returns**: Genre-appropriate tracks with diversity scoring

### 3. `search_tracks`
**Purpose**: General music search with flexible queries
```python
# Example: "Find songs like Blinding Lights"
search_tracks(query="Blinding Lights The Weeknd", limit=10)
```
**Returns**: Relevant tracks matching search criteria

### 4. `get_similar_songs`
**Purpose**: Find music similar to specific artists
```python
# Example: "Artists similar to Billie Eilish"
get_similar_songs(artist_name="Billie Eilish", limit=8)
```
**Returns**: Related artists and their popular tracks

### 5. `create_smart_playlist`
**Purpose**: Generate curated playlists with AI analysis
```python
# Example: "Create a workout playlist"
create_smart_playlist(name="Workout Mix", description="High energy tracks", size=15)
```
**Returns**: Thoughtfully curated playlist with flow analysis

### 6. `tavily_search_results_json`
**Purpose**: Search for current music events and concerts
```python
# Example: "Who's performing in NYC this weekend?"
tavily_search_results_json(query="concerts New York this weekend")
```
**Returns**: Real-time event information and concert listings

## 🎨 React Frontend Features

### Modern Spotify-Style UI
- **Song Cards**: Compact horizontal layout with album art, track info, and play buttons
- **Responsive Design**: Works seamlessly on desktop and mobile
- **Clean Interface**: Minimal, music-focused design language
- **Direct Integration**: Play buttons link directly to Spotify

### Interactive Chat Experience
- **Natural Conversation**: Chat with the AI about music preferences
- **Real-time Responses**: Streaming responses with loading states
- **Song Display**: Beautiful cards showing recommended tracks
- **Spotify Integration**: One-click access to full tracks

### Key Components
```typescript
// Song card component with Spotify styling
<SongCard
  track={song}
  index={index}
  onPlay={() => window.open(song.spotify_url)}
/>

// Chat interface with streaming responses
<ChatInterface
  messages={messages}
  onSendMessage={handleSendMessage}
  isLoading={isLoading}
/>
```

## 🚀 How It Works

### 1. User Interaction
User sends natural language music query through React chat interface:
- "Find me some upbeat pop songs like Dua Lipa"
- "Create a chill playlist for studying"
- "Who's performing in Philadelphia soon?"

### 2. Agent Processing
LangChain agent analyzes the query and:
- Determines appropriate tool(s) to use
- Executes Spotify API calls through specialized tools
- Processes and formats the response data
- Maintains conversation context for follow-up queries

### 3. Response Generation
Agent provides brief, DJ-style responses with:
- Curated song recommendations
- Quick commentary about artists/tracks
- Structured data for frontend display
- Spotify URLs for immediate playback

### 4. Frontend Display
React frontend renders:
- Clean song cards with album art and metadata
- Play buttons linking directly to Spotify
- Conversational chat history
- Responsive, mobile-friendly interface

## 📊 Agent Capabilities

### Music Discovery
- **Artist Exploration**: Top songs, similar artists, discography insights
- **Genre Discovery**: Curated selections across all music genres
- **Mood-Based Recommendations**: Music matching specific vibes or activities
- **Playlist Generation**: AI-curated playlists with flow analysis

### Conversational Intelligence
- **Context Awareness**: Remembers previous queries and preferences
- **Natural Language**: Understands complex, multi-part music requests
- **Brief Responses**: Spotify DJ-style commentary (1-2 sentences max)
- **Error Handling**: Graceful handling of unclear or invalid queries

### Real-Time Information
- **Concert Search**: Current events and tour information
- **Music News**: Latest releases and industry updates
- **Location-Based**: Local concert and event recommendations

## 🔧 Technical Stack

### Backend
- **Python 3.11+** with FastAPI
- **LangChain** for agent orchestration
- **OpenAI GPT-4o-mini** for natural language processing
- **Spotify Web API** for music data
- **LangSmith** for observability and evaluation
- **Pydantic v2** for data validation

### Frontend
- **Next.js 15** with TypeScript
- **React 18** with modern hooks
- **Tailwind CSS** for styling
- **Lucide React** for icons
- **Responsive design** principles

### Development Tools
- **LangSmith Evaluation** for agent performance testing
- **OpenEvals** for standardized evaluation metrics
- **Git** version control with comprehensive commit history

## 🎵 Example Interactions

### Basic Artist Search
```
User: "Show me The Weeknd's biggest hits"
Agent: "Check out The Weeknd's top hits like 'Blinding Lights,' 'Starboy,' and 'Die For You.' These tracks showcase his signature sound and massive appeal."
```

### Genre Discovery
```
User: "I want to discover some indie rock bands"
Agent: "Dive into the indie rock scene with fresh sounds from The Backseat Lovers, Hozier, and Arctic Monkeys to get your indie fix!"
```

### Event Search
```
User: "Who's performing in New York this weekend?"
Agent: "This weekend in NYC, there are over 66 concerts happening, including festivals like Anti Social Camp. Check Ticketmaster for tickets!"
```

## 🎯 Key Features

### For Music Lovers
- **Personalized Recommendations** based on conversation context
- **Instant Spotify Access** with direct play button integration
- **Diverse Discovery** across genres, moods, and eras
- **Event Information** for live music experiences

### For Developers
- **Modern Architecture** with clean separation of concerns
- **Comprehensive Evaluation** with LangSmith integration
- **Scalable Design** supporting additional music services
- **Professional Codebase** with TypeScript and proper error handling

## 🚀 Getting Started

### Prerequisites
- Node.js 18+ and npm
- Python 3.11+
- OpenAI API key
- LangSmith API key (optional, for tracing)

### Quick Start
```bash
# Clone the repository
git clone <repository-url>
cd spotify-music-agent

# Backend setup
pip install -r requirements.txt
cp .env.example .env  # Add your API keys
python run_api.py

# Frontend setup (new terminal)
cd spotify-agent-demo
npm install
npm run dev
```

Visit `http://localhost:3000` to start chatting with your AI music concierge!

## 📈 Evaluation & Testing

The agent includes comprehensive evaluation using LangSmith and OpenEvals:
- **10 realistic test scenarios** covering different query types
- **5 evaluation metrics** including response quality and tool efficiency
- **Performance tracking** across conversation complexity
- **Continuous improvement** through evaluation insights

## 🎵 Built for Music Discovery

This AI music agent represents the future of music discovery - combining the intelligence of modern language models with the vast catalog of Spotify, delivered through a clean, intuitive interface that makes finding your next favorite song as easy as having a conversation.

---

**Ready to discover your next favorite song?** Start chatting with the AI music concierge today!
