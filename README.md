# Spotify Music Agent

A production-ready AI music agent that provides personalized music recommendations and playlist generation using Spotify's API. Built with LangChain and OpenAI, featuring comprehensive evaluation metrics and a clean web interface.

## Overview

This system implements an AI-powered music concierge that acts as a DJ, providing brief, conversational music recommendations rather than detailed track lists. The agent can search for songs, artists, create playlists, and provide music recommendations based on user queries.

## Key Features

- **AI Music Agent**: LangChain-powered agent with Spotify API integration
- **Evaluation Framework**: 7 production-ready evaluators for agent performance
- **Web Interface**: Next.js frontend for interactive music discovery
- **LangSmith Integration**: Comprehensive tracing and evaluation pipeline
- **RESTful API**: FastAPI backend with standardized endpoints

## Architecture

- **Backend**: Python FastAPI server with LangChain agent
- **Frontend**: Next.js React application
- **AI Model**: OpenAI GPT-4 with custom prompting for DJ-style responses
- **APIs**: Spotify Web API, Tavily search API
- **Evaluation**: Custom evaluators for tool correctness, response style, and efficiency

## Installation

1. Clone the repository
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install frontend dependencies:
   ```bash
   cd spotify-agent-demo
   npm install
   ```

## Configuration

1. Copy the environment template:
   ```bash
   cp .env.example .env
   ```
2. Add your API keys to `.env`:
   - `OPENAI_API_KEY`: OpenAI API access
   - `SPOTIFY_CLIENT_ID` and `SPOTIFY_CLIENT_SECRET`: Spotify Web API
   - `TAVILY_API_KEY`: Web search functionality
   - `LANGSMITH_API_KEY`: Evaluation and tracing

## Running the Application

### Backend API Server

```bash
python run_api.py
```

The API will be available at `http://127.0.0.1:8000` with documentation at `/docs`.

### Frontend Development Server

```bash
cd spotify-agent-demo
npm run dev
```

The web interface will be available at `http://localhost:3000`.

## API Endpoints

- `POST /chat`: Submit music queries and receive AI responses
- `POST /evaluate`: Run evaluation metrics on agent responses
- `GET /health`: Health check endpoint

## Evaluation

The system includes 7 evaluation metrics:

- Tool Correctness: Validates proper API usage
- Tool Efficiency: Ensures optimal performance (≤3 API calls)
- DJ Style: Enforces conversational, brief responses
- Playlist Size: Validates playlist length accuracy
- Error Handling: Tests graceful failure handling
- Music Relevance: Scores response relevance
- Helpfulness: Measures overall response quality

Run evaluations with:

```bash
python evaluations/run.py
```

## License

MIT License
