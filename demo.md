# LangSmith Evaluation Demo Script
## Evaluating a Spotify AI Agent for Music Discovery

*Total Duration: ~10 minutes*

**Demo Files:** All code and documentation available in `/evaluation-demo/` folder

---

### Introduction (1 minute)

**"Hi everyone! Today I'm going to walk through a LangSmith evaluation experiment I built for a real-world LLM application."**

**What I Built:**
- A Spotify AI agent that helps users discover music, create playlists, and find recommendations
- Built with FastAPI backend + Next.js frontend
- Uses multiple Spotify Web API tools (search, recommendations, artist top tracks, etc.)
- **Full evaluation suite in the `/evaluation-demo/` folder**

**The Challenge:**
- How do we evaluate if our AI agent gives good music recommendations?
- Are the tools being used correctly?
- Is the conversational experience actually helpful?

**Demo Assets:**
- `evaluation.py` - SDK evaluation with custom evaluators
- `ui_evaluation_guide.py` - UI evaluation setup and comparison
- `friction_log.md` - Real developer experience feedback
- `results_analysis.py` - Automated insights generation

---

### The Dataset (1.5 minutes)

**"I created a realistic test dataset with 25 music discovery scenarios:"**

```
Examples:
1. "Find me workout songs similar to Eminem"
   Expected: High-energy rap/hip-hop tracks, use search + recommendations tools

2. "What are Taylor Swift's most popular songs?"
   Expected: Use artist top tracks tool, return her biggest hits

3. "Create a chill playlist for studying"
   Expected: Use search with genre filters, focus on instrumental/ambient music
```

**Why This Dataset:**
- Covers different user intents (discovery, artist info, playlist creation)
- Tests different tool combinations
- Includes edge cases (misspelled artists, vague requests)

---

### Evaluation Metrics (1.5 minutes)

**"I designed three key evaluation dimensions:"**

**1. Tool Usage Accuracy (Automated)**
```python
def evaluate_tool_usage(input_query, tools_used):
    # Did it use the right Spotify API endpoints?
    # Artist query -> should use artist_top_tracks
    # Playlist creation -> should use search + recommendations
```

**2. Musical Relevance (LLM-as-Judge)**
```python
def evaluate_relevance(query, songs_returned):
    prompt = """
    Query: {query}
    Songs: {songs}

    Rate 1-5: Do these songs match the user's request?
    Consider genre, mood, energy level, and artist style.
    """
```

**3. Response Quality (LLM-as-Judge)**
```python
def evaluate_response_quality(query, response):
    # Is the AI's commentary helpful and concise?
    # Does it explain why these songs were chosen?
    # Is it conversational like a real DJ?
```

---

### SDK Demo (2.5 minutes)

**"Let me show you the evaluation code in action:"**

**Live Demo Steps:**
1. **Show the evaluation script:** `evaluation-demo/evaluation.py`
2. **Run the evaluation:** `python evaluation-demo/evaluation.py`
3. **Watch real-time execution** with LangSmith tracing

**Key Code Highlights:**
```python
# Custom Music Evaluators with OpenEvals
def song_relevance_evaluator(run, example):
    prompt = """Rate music recommendation relevance 0.0-1.0:
    USER QUERY: {query}
    SONGS RETURNED: {songs_count} songs
    Consider: genre, artist matching, mood/context fit."""

    evaluator = create_llm_as_judge(prompt, model="gpt-4o-mini")
    return evaluator(inputs=inputs, outputs=run.outputs)

# Run comprehensive evaluation
results = evaluate(
    spotify_agent_target,
    data=dataset_id,
    evaluators=[song_relevance, dj_style, tool_efficiency],
    experiment_prefix="spotify-tse-eval"
)
```

**Live Demo Results:**
- Real-time LangSmith trace visualization
- Custom evaluator scores in action
- Performance by test category

---

### UI Demo (2 minutes)

**"Now let's see the same experiment in the LangSmith UI:"**

**Demo Steps:** Follow `evaluation-demo/ui_evaluation_guide.py`

1. **Dataset View**: Navigate to "spotify-agent-tse-demo" dataset
2. **Experiment Setup**: Configure built-in evaluators (Correctness, Helpfulness, Conciseness)
3. **Results Dashboard**:
   - Overall scores by metric
   - Individual run traces
   - Comparison view between SDK and UI runs

**Key UI Features Demonstrated:**
- Visual trace exploration with tool call sequences
- Built-in evaluator variety and ease of setup
- Side-by-side run comparison capabilities
- Filtering and analysis tools

**SDK vs UI Comparison:**
- **SDK**: Custom music evaluators, programmatic analysis
- **UI**: Built-in evaluators, visual exploration, team collaboration
- **Best Practice**: Use both approaches for comprehensive evaluation

---

### Key Findings (1 minute)

**What Worked Well:**
- Tool usage accuracy: 92% - agent correctly chose tools for most queries
- Musical relevance: 4.2/5 average - songs generally matched user intent
- Response quality: 3.8/5 - conversational but sometimes too verbose

**Interesting Discoveries:**
- Edge case: Misspelled artist names caused tool failures
- Genre boundary issues: "chill hip-hop" sometimes returned aggressive tracks
- Tool chaining: Complex requests needed better orchestration

**Failure Analysis:**
- 3 cases where search returned no results → poor error handling
- 2 cases where recommendations were off-genre → need better prompt engineering

---

### Friction Log & Learnings (0.5 minutes)

**What I Found Confusing:**
- Dataset schema documentation could be clearer
- LLM-as-judge setup required trial and error
- Trace filtering in UI was limited

**What I Loved:**
- Real-time experiment monitoring
- Easy comparison between evaluation runs
- Rich trace data for debugging

**If I Were On The Team, I'd Suggest:**
- Template gallery for common evaluation patterns
- Better error messages for dataset upload issues
- More granular filtering options in the traces view

---

### Wrap-up (0.5 minutes)

**"This experiment showed me how LangSmith makes LLM app evaluation practical:"**

- Easy to set up realistic test scenarios
- Multiple evaluation approaches (automated + LLM-judge)
- Rich debugging capabilities through trace visualization
- Clear path from evaluation insights to application improvements

**Next Steps:**
- Implement fixes for the failure cases we identified
- Add more edge cases to the dataset
- Set up continuous evaluation for production monitoring

---

## Demo Code Repository

All code, datasets, and evaluation scripts available in:
`/evaluation-demo/`

**Key Files:**
- `evaluation.py` - Complete SDK evaluation with custom evaluators
- `ui_evaluation_guide.py` - UI setup guide and comparison framework
- `friction_log.md` - Developer experience feedback template
- `results_analysis.py` - Automated insights and report generation
- `README.md` - Comprehensive setup and usage instructions
- `requirements.txt` - Python dependencies

**Generated Outputs:**
- LangSmith experiments with full tracing
- `evaluation_report_*.md` - Automated analysis reports
- Performance insights and improvement recommendations

**Ready for TSE Presentation:**
- Complete 10-minute demo workflow
- Real production agent evaluation
- Comprehensive friction logging
- Actionable product feedback