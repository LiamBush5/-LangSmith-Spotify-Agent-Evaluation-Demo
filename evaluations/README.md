# Spotify Agent Evaluation System

Interactive evaluation framework for the Spotify Music Agent with support for multiple models, dataset splits, and comprehensive reporting.

## 🚀 Quick Start

```bash
cd evaluations
python run.py
```

## 🎯 Interactive Features

### 1. **Model Selection**
Choose from 5 different LLM models:
- **GPT-4o** (OpenAI) - Best quality
- **GPT-4o-mini** (OpenAI) - Fast & cost-effective
- **Gemini-1.5-Pro** (Google) - Advanced reasoning
- **Gemini-1.5-Flash** (Google) - Speed optimized
- **Claude-3.5-Sonnet** (Anthropic) - Excellent writing

### 2. **Split Selection**
Choose evaluation scope:
- **All splits** - Complete 36-test evaluation
- **Quick smoke test** - 3 examples for rapid validation
- **Easy tests only** - 10 quick tests for iteration
- **Hard tests only** - 9 stress tests
- **Playlist tests only** - 4 playlist creation tests
- **Complex query tests** - 5 multi-tool workflows
- **Event-based tests** - 4 real-time search tests
- **Custom selection** - Mix and match multiple splits

### 3. **Experiment Naming**
Auto-generated format: `model-split-timestamp`

Examples:
- `gpt-4o-mini-easy-tests-20250128-143052`
- `gemini-1.5-pro-playlist-tests-20250128-143125`
- `claude-3.5-sonnet-all-tests-20250128-143200`

## 📊 Dataset Structure

**36 total test cases** across:

### Categories (8):
- `basic_search` (5) - Artist/song lookup
- `genre_discovery` (6) - Genre exploration
- `mood_based` (6) - Context-specific music
- `playlist_creation` (4) - Smart playlists
- `complex_query` (5) - Multi-constraint searches
- `event_search` (4) - Concerts/festivals
- `edge_case` (4) - Error handling
- `efficiency_test` (2) - Minimal queries

### Difficulties (3):
- `easy` (10) - Single-tool workflows
- `medium` (17) - 2-tool workflows
- `hard` (9) - 3+ tool workflows

## 🛠️ Evaluators (7)

1. **Tool Correctness** - Trajectory evaluation
2. **Tool Efficiency** - Hard limits (≤3 tools)
3. **DJ Style** - Brief, conversational responses
4. **Playlist Size** - Validates requested sizes
5. **Error Handling** - Graceful failure management
6. **Music Relevance** - Domain-specific accuracy
7. **Helpfulness** - User satisfaction

## 💡 Usage Examples

### Quick Development Iteration
```bash
python run.py
# Choose: 2 (GPT-4o-mini)
# Choose: 3 (Easy tests only)
# Result: gpt-4o-mini-easy-tests-20250128-143052
```

### Stress Testing
```bash
python run.py
# Choose: 1 (GPT-4o)
# Choose: 4 (Hard tests only)
# Result: gpt-4o-hard-tests-20250128-143125
```

### Custom Multi-Split Testing
```bash
python run.py
# Choose: 3 (Gemini-1.5-Pro)
# Choose: 8 (Custom selection)
# Enter: 1,5,7 (Easy + Playlist + Complex)
# Result: Multiple experiments generated
```

## 🔧 Advanced Usage

### Filter by Specific Metadata
```python
from run import SpotifyAgentEvaluation

evaluation = SpotifyAgentEvaluation()
evaluation.create_dataset()

# Run only sized playlist tests
results = evaluation.run_evaluation(
    filter_metadata={"expected_playlist_size": {"$exists": True}},
    model="gpt-4o",
    split_name="sized-playlists"
)
```

### Programmatic Batch Testing
```python
models = ["gpt-4o-mini", "gemini-1.5-flash", "claude-3.5-sonnet"]
splits = [
    {"name": "easy-tests", "filter": {"difficulty": "easy"}},
    {"name": "playlist-tests", "filter": {"category": "playlist_creation"}}
]

for model in models:
    for split in splits:
        evaluation.run_evaluation(
            filter_metadata=split["filter"],
            model=model,
            split_name=split["name"]
        )
```

## 📈 Experiment Results

All results are automatically tracked in LangSmith with:
- **Experiment name** in `model-split-timestamp` format
- **Metadata** including model, split, filters
- **Detailed scoring** across all 7 evaluators
- **Comparison views** for A/B testing
- **Export capabilities** for further analysis

## 🎵 Example Experiment Names

- `gpt-4o-mini-smoke-test-20250128-143052`
- `gemini-1.5-pro-playlist-tests-20250128-143125`
- `claude-3.5-sonnet-hard-tests-20250128-143200`
- `gpt-4o-custom-selection-20250128-143235`

Run `python run.py` to get started with interactive evaluation! 🚀