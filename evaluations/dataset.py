"""
#test_difficulty:
- easy
- medium
- hard

#test_type:
- basic_search
- genre_discovery
- mood_based
- playlist_creation
- complex_query
- event_search
- edge_case
- efficiency_test
"""

from typing import List, Dict, Any

def get_evaluation_dataset() -> List[Dict[str, Any]]:
    """
    Returns comprehensive evaluation dataset with good coverage across:
    - Query complexity (easy/medium/hard)
    - Categories (search, discovery, mood, playlist, events, edge cases)
    - Tool usage patterns
    - Response types
    """

    dataset = []

    # ===================
    # BASIC SEARCH - Easy
    # ===================

    # Artist search
    dataset.extend([
        {
            "inputs": {"query": "Find me Taylor Swift's most popular songs"},
            "outputs": {
                "expected_behavior": "Should return 5+ Taylor Swift songs with brief commentary",
                "success_criteria": "Songs returned, artist matches, response is brief and vibey"
            },
            "expected_tools": ["get_artist_top_songs"],
            "metadata": {
                "category": "basic_search",
                "difficulty": "easy",
                "max_tool_calls": 2,
                "query_type": "artist_songs"
            }
        },
        {
            "inputs": {"query": "Show me The Weeknd's hits"},
            "outputs": {
                "expected_behavior": "Should efficiently return The Weeknd's popular songs",
                "success_criteria": "Correct artist, popular songs, minimal tool usage"
            },
            "expected_tools": ["get_artist_top_songs"],
            "metadata": {
                "category": "basic_search",
                "difficulty": "easy",
                "max_tool_calls": 1,
                "query_type": "artist_hits"
            }
        },
        {
            "inputs": {"query": "Play some Billie Eilish"},
            "outputs": {
                "expected_behavior": "Should return Billie Eilish songs with chill commentary",
                "success_criteria": "Correct artist, good song selection, laid back vibe"
            },
            "expected_tools": ["get_artist_top_songs"],
            "metadata": {
                "category": "basic_search",
                "difficulty": "easy",
                "max_tool_calls": 1,
                "query_type": "artist_play"
            }
        }
    ])

    # Song/Track search
    dataset.extend([
        {
            "inputs": {"query": "Find the song Blinding Lights"},
            "outputs": {
                "expected_behavior": "Should find the specific song by The Weeknd",
                "success_criteria": "Correct song found, artist identified, brief response"
            },
            "expected_tools": ["search_tracks"],
            "metadata": {
                "category": "basic_search",
                "difficulty": "easy",
                "max_tool_calls": 1,
                "query_type": "specific_song"
            }
        },
        {
            "inputs": {"query": "What's that song that goes 'drivers license'"},
            "outputs": {
                "expected_behavior": "Should identify Olivia Rodrigo's 'drivers license'",
                "success_criteria": "Correct song identification, helpful response"
            },
            "expected_tools": ["search_tracks"],
            "metadata": {
                "category": "basic_search",
                "difficulty": "medium",
                "max_tool_calls": 2,
                "query_type": "lyric_search"
            }
        }
    ])

    # ===================
    # GENRE DISCOVERY - Medium
    # ===================

    dataset.extend([
        {
            "inputs": {"query": "I want to discover some indie rock bands"},
            "outputs": {
                "expected_behavior": "Should return indie rock songs with artist variety",
                "success_criteria": "Genre-appropriate songs, multiple artists, discovery focus"
            },
            "expected_tools": ["get_genre_songs"],

            "metadata": {
                "category": "genre_discovery",
                "difficulty": "medium",
                "max_tool_calls": 2,
                "query_type": "genre_exploration"
            }
        },
        {
            "inputs": {"query": "Show me some lo-fi hip hop"},
            "outputs": {
                "expected_behavior": "Should return chill lo-fi hip hop tracks",
                "success_criteria": "Correct genre, chill vibe matches, good variety"
            },
            "expected_tools": ["get_genre_songs"],

            "metadata": {
                "category": "genre_discovery",
                "difficulty": "easy",
                "max_tool_calls": 1,
                "query_type": "specific_genre"
            }
        },
        {
            "inputs": {"query": "What's good in electronic music lately?"},
            "outputs": {
                "expected_behavior": "Should provide recent/trending electronic music",
                "success_criteria": "Electronic genre, contemporary feel, discovery aspect"
            },
            "expected_tools": ["get_genre_songs", "search_tracks"],

            "metadata": {
                "category": "genre_discovery",
                "difficulty": "medium",
                "max_tool_calls": 2,
                "query_type": "trending_genre"
            }
        },
        {
            "inputs": {"query": "Recommend some jazz for a dinner party"},
            "outputs": {
                "expected_behavior": "Should provide jazz recommendations with appropriate context",
                "success_criteria": "Jazz genre, dinner party appropriate, good selection"
            },
            "expected_tools": ["get_genre_songs"],

            "metadata": {
                "category": "genre_discovery",
                "difficulty": "medium",
                "max_tool_calls": 2,
                "query_type": "contextual_genre"
            }
        },
        {
            "inputs": {"query": "Show me some underground hip hop artists"},
            "outputs": {
                "expected_behavior": "Should return less mainstream hip hop artists and tracks",
                "success_criteria": "Hip hop genre, underground/alternative artists, good variety"
            },
            "expected_tools": ["get_genre_songs", "search_tracks"],

            "metadata": {
                "category": "genre_discovery",
                "difficulty": "hard",
                "max_tool_calls": 3,
                "query_type": "underground_discovery"
            }
        },
        {
            "inputs": {"query": "What's trending in K-pop right now?"},
            "outputs": {
                "expected_behavior": "Should provide current popular K-pop tracks and artists",
                "success_criteria": "K-pop genre, trending/current songs, popular artists"
            },
            "expected_tools": ["get_genre_songs", "search_tracks"],

            "metadata": {
                "category": "genre_discovery",
                "difficulty": "medium",
                "max_tool_calls": 2,
                "query_type": "trending_kpop"
            }
        }
    ])

    # ===================
    # MOOD/CONTEXT - Medium
    # ===================

    dataset.extend([
        {
            "inputs": {"query": "Give me energetic workout music"},
            "outputs": {
                "expected_behavior": "Should return high-energy songs suitable for workouts",
                "success_criteria": "Energetic songs, workout-appropriate, good variety"
            },
            "expected_tools": ["get_genre_songs", "search_tracks"],

            "metadata": {
                "category": "mood_based",
                "difficulty": "medium",
                "max_tool_calls": 2,
                "query_type": "workout_mood"
            }
        },
        {
            "inputs": {"query": "I need some sad songs to cry to"},
            "outputs": {
                "expected_behavior": "Should provide emotional, melancholic music",
                "success_criteria": "Appropriate mood, emotional songs, empathetic tone"
            },
            "expected_tools": ["get_genre_songs", "search_tracks"],

            "metadata": {
                "category": "mood_based",
                "difficulty": "medium",
                "max_tool_calls": 2,
                "query_type": "emotional_mood"
            }
        },
        {
            "inputs": {"query": "Something chill for studying"},
            "outputs": {
                "expected_behavior": "Should return calm, non-distracting music for focus",
                "success_criteria": "Chill vibe, study-appropriate, instrumental preferred"
            },
            "expected_tools": ["get_genre_songs"],

            "metadata": {
                "category": "mood_based",
                "difficulty": "easy",
                "max_tool_calls": 1,
                "query_type": "study_mood"
            }
        },
        {
            "inputs": {"query": "Party music that gets everyone dancing"},
            "outputs": {
                "expected_behavior": "Should return upbeat, danceable party tracks",
                "success_criteria": "High energy, danceable, crowd-pleasing songs"
            },
            "expected_tools": ["get_genre_songs"],

            "metadata": {
                "category": "mood_based",
                "difficulty": "medium",
                "max_tool_calls": 2,
                "query_type": "party_mood"
            }
        },
        {
            "inputs": {"query": "I need music to help me focus while coding"},
            "outputs": {
                "expected_behavior": "Should return instrumental or low-key music good for concentration",
                "success_criteria": "Focus-friendly music, minimal vocals, good for productivity"
            },
            "expected_tools": ["get_genre_songs"],

            "metadata": {
                "category": "mood_based",
                "difficulty": "medium",
                "max_tool_calls": 2,
                "query_type": "focus_mood"
            }
        },
        {
            "inputs": {"query": "Relaxing music for a spa day at home"},
            "outputs": {
                "expected_behavior": "Should return calming, ambient music perfect for relaxation",
                "success_criteria": "Relaxing vibe, spa-appropriate, soothing sounds"
            },
            "expected_tools": ["get_genre_songs"],

            "metadata": {
                "category": "mood_based",
                "difficulty": "easy",
                "max_tool_calls": 1,
                "query_type": "spa_relaxation"
            }
        }
    ])

    # ===================
    # PLAYLIST CREATION - Hard
    # ===================

    dataset.extend([
        {
            "inputs": {"query": "Create a chill playlist with artists like Billie Eilish"},
            "outputs": {
                "expected_behavior": "Should create playlist with Billie Eilish and similar artists",
                "success_criteria": "Playlist created, similar artists included, 10+ songs"
            },
            "expected_tools": ["create_smart_playlist", "get_similar_songs"],

            "metadata": {
                "category": "playlist_creation",
                "difficulty": "hard",
                "max_tool_calls": 3,
                "query_type": "similar_artist_playlist"
            }
        },
        {
            "inputs": {"query": "Make me a 15 song road trip playlist"},
            "outputs": {
                "expected_behavior": "Should create 15-song playlist for road trips",
                "success_criteria": "Road trip vibe, exactly ~15 songs, upbeat"
            },
            "expected_tools": ["create_smart_playlist", "get_genre_songs"],

            "metadata": {
                "category": "playlist_creation",
                "difficulty": "hard",
                "max_tool_calls": 3,
                "query_type": "sized_playlist",
                "expected_playlist_size": 15
            }
        },
        {
            "inputs": {"query": "Build a 20 track playlist that goes from calm to energetic"},
            "outputs": {
                "expected_behavior": "Should create 20-track playlist with gradual energy progression",
                "success_criteria": "Energy progression, smooth transitions, exactly ~20 tracks"
            },
            "expected_tools": ["create_smart_playlist", "get_genre_songs"],

            "metadata": {
                "category": "playlist_creation",
                "difficulty": "hard",
                "max_tool_calls": 4,
                "query_type": "progressive_playlist",
                "expected_playlist_size": 20
            }
        },
        {
            "inputs": {"query": "Create a 10 song workout playlist"},
            "outputs": {
                "expected_behavior": "Should create energetic 10-song workout playlist",
                "success_criteria": "High energy songs, workout appropriate, exactly ~10 songs"
            },
            "expected_tools": ["create_smart_playlist", "get_genre_songs"],

            "metadata": {
                "category": "playlist_creation",
                "difficulty": "medium",
                "max_tool_calls": 2,
                "query_type": "workout_sized_playlist",
                "expected_playlist_size": 10
            }
        }
    ])

    # ===================
    # COMPLEX QUERIES - Hard
    # ===================

    dataset.extend([
        {
            "inputs": {"query": "Find me upbeat pop songs similar to Dua Lipa but not too mainstream"},
            "outputs": {
                "expected_behavior": "Should balance multiple constraints: upbeat, pop, Dua Lipa-like, less mainstream",
                "success_criteria": "Meets all constraints, good song selection, appropriate artists"
            },
            "expected_tools": ["get_similar_songs", "search_tracks"],

            "metadata": {
                "category": "complex_query",
                "difficulty": "hard",
                "max_tool_calls": 3,
                "query_type": "multi_constraint"
            }
        },
        {
            "inputs": {"query": "I like Radiohead and Frank Ocean, what else would I enjoy?"},
            "outputs": {
                "expected_behavior": "Should find artists that bridge alternative rock and R&B/hip-hop",
                "success_criteria": "Thoughtful recommendations, explains connections, diverse suggestions"
            },
            "expected_tools": ["get_similar_songs", "search_tracks"],

            "metadata": {
                "category": "complex_query",
                "difficulty": "hard",
                "max_tool_calls": 3,
                "query_type": "taste_analysis"
            }
        },
        {
            "inputs": {"query": "Songs like Bohemian Rhapsody but more modern"},
            "outputs": {
                "expected_behavior": "Should find modern songs with theatrical/progressive elements",
                "success_criteria": "Understands song characteristics, finds modern equivalents"
            },
            "expected_tools": ["get_similar_songs", "search_tracks"],

            "metadata": {
                "category": "complex_query",
                "difficulty": "hard",
                "max_tool_calls": 3,
                "query_type": "song_similarity"
            }
        },
        {
            "inputs": {"query": "Look up the recent Grammy winners and make me a playlist"},
            "outputs": {
                "expected_behavior": "Should search for recent Grammy winners, then create playlist with their songs",
                "success_criteria": "Current Grammy info, playlist created from winners, good variety"
            },
            "expected_tools": ["tavily_search_results_json", "get_artist_top_songs", "create_smart_playlist"],

            "metadata": {
                "category": "complex_query",
                "difficulty": "hard",
                "max_tool_calls": 4,
                "query_type": "search_and_playlist"
            }
        },
        {
            "inputs": {"query": "Make me a playlist of all the NYC bands playing this weekend and tell me where they're performing"},
            "outputs": {
                "expected_behavior": "Should search for NYC events this weekend, create playlist, provide venue info",
                "success_criteria": "Current event data, playlist from performing artists, venue information included"
            },
            "expected_tools": ["tavily_search_results_json", "get_artist_top_songs", "create_smart_playlist"],

            "metadata": {
                "category": "complex_query",
                "difficulty": "hard",
                "max_tool_calls": 4,
                "query_type": "event_playlist_combo"
            }
        }
    ])

    # ===================
    # EVENT SEARCH - Medium
    # ===================

    dataset.extend([
        {
            "inputs": {"query": "Who's performing in New York this weekend?"},
            "outputs": {
                "expected_behavior": "Should search for current concert information in NYC",
                "success_criteria": "Current event data, location-specific, time-relevant"
            },
            "expected_tools": ["tavily_search_results_json"],

            "metadata": {
                "category": "event_search",
                "difficulty": "medium",
                "max_tool_calls": 2,
                "query_type": "local_events"
            }
        },
        {
            "inputs": {"query": "When is Taylor Swift touring next?"},
            "outputs": {
                "expected_behavior": "Should find Taylor Swift tour dates and information",
                "success_criteria": "Current tour info, dates, locations if available"
            },
            "expected_tools": ["tavily_search_results_json"],

            "metadata": {
                "category": "event_search",
                "difficulty": "medium",
                "max_tool_calls": 2,
                "query_type": "artist_tour"
            }
        },
        {
            "inputs": {"query": "What music festivals are happening this summer?"},
            "outputs": {
                "expected_behavior": "Should search for upcoming summer music festivals",
                "success_criteria": "Current festival information, dates, lineups if available"
            },
            "expected_tools": ["tavily_search_results_json"],

            "metadata": {
                "category": "event_search",
                "difficulty": "medium",
                "max_tool_calls": 2,
                "query_type": "festival_search"
            }
        },
        {
            "inputs": {"query": "Find concerts in Los Angeles next month"},
            "outputs": {
                "expected_behavior": "Should search for LA concerts in the next month",
                "success_criteria": "Location-specific, time-specific, current event data"
            },
            "expected_tools": ["tavily_search_results_json"],

            "metadata": {
                "category": "event_search",
                "difficulty": "medium",
                "max_tool_calls": 2,
                "query_type": "location_time_events"
            }
        }
    ])

    # ===================
    # EDGE CASES - Easy to Medium
    # ===================

    dataset.extend([
        {
            "inputs": {"query": "Play some music"},
            "outputs": {
                "expected_behavior": "Should handle vague request gracefully, ask for clarification or provide general recommendations",
                "success_criteria": "Graceful handling, helpful response, no errors"
            },
            "expected_tools": ["get_genre_songs"],

            "metadata": {
                "category": "edge_case",
                "difficulty": "easy",
                "max_tool_calls": 1,
                "query_type": "vague_request"
            }
        },
        {
            "inputs": {"query": "Find songs by XYZNonExistentArtist123"},
            "outputs": {
                "expected_behavior": "Should handle gracefully and suggest alternatives",
                "success_criteria": "No crashes, helpful error handling, alternative suggestions"
            },
            "expected_tools": ["search_tracks"],

            "metadata": {
                "category": "edge_case",
                "difficulty": "medium",
                "max_tool_calls": 2,
                "query_type": "invalid_artist"
            }
        },
        {
            "inputs": {"query": "What's the best song ever made?"},
            "outputs": {
                "expected_behavior": "Should handle subjective question thoughtfully",
                "success_criteria": "Acknowledges subjectivity, provides interesting response"
            },
            "expected_tools": ["get_genre_songs", "search_tracks"],

            "metadata": {
                "category": "edge_case",
                "difficulty": "medium",
                "max_tool_calls": 2,
                "query_type": "subjective_question"
            }
        },
        {
            "inputs": {"query": "🎵🎶🎸"},
            "outputs": {
                "expected_behavior": "Should handle emoji-only query gracefully",
                "success_criteria": "Interprets musical intent, provides music recommendations"
            },
            "expected_tools": ["get_genre_songs"],

            "metadata": {
                "category": "edge_case",
                "difficulty": "medium",
                "max_tool_calls": 1,
                "query_type": "emoji_query"
            }
        }
    ])

    # ===================
    # EFFICIENCY TESTS - Easy
    # ===================

    dataset.extend([
        {
            "inputs": {"query": "Drake"},
            "outputs": {
                "expected_behavior": "Should efficiently return Drake's popular songs",
                "success_criteria": "Single artist, efficient tool usage, good songs"
            },
            "expected_tools": ["get_artist_top_songs"],

            "metadata": {
                "category": "efficiency_test",
                "difficulty": "easy",
                "max_tool_calls": 1,
                "query_type": "minimal_query"
            }
        },
        {
            "inputs": {"query": "Pop music"},
            "outputs": {
                "expected_behavior": "Should return popular pop songs efficiently",
                "success_criteria": "Pop genre, efficient tool usage, good variety"
            },
            "expected_tools": ["get_genre_songs"],

            "metadata": {
                "category": "efficiency_test",
                "difficulty": "easy",
                "max_tool_calls": 1,
                "query_type": "genre_minimal"
            }
        }
    ])

    return dataset


def get_dataset_stats(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate statistics about the dataset"""

    total_cases = len(dataset)

    # Count by category
    categories = {}
    difficulties = {}
    query_types = {}

    for case in dataset:
        metadata = case["metadata"]

        # Category counts
        cat = metadata["category"]
        categories[cat] = categories.get(cat, 0) + 1

        # Difficulty counts
        diff = metadata["difficulty"]
        difficulties[diff] = difficulties.get(diff, 0) + 1

        # Query type counts
        qtype = metadata["query_type"]
        query_types[qtype] = query_types.get(qtype, 0) + 1

    return {
        "total_cases": total_cases,
        "categories": categories,
        "difficulties": difficulties,
        "query_types": query_types
    }


if __name__ == "__main__":
    dataset = get_evaluation_dataset()
    stats = get_dataset_stats(dataset)

    print("Spotify Agent Evaluation Dataset")
    print("=" * 40)
    print(f"Total test cases: {stats['total_cases']}")
    print()

    print("By Category:")
    for cat, count in stats['categories'].items():
        print(f"  {cat}: {count}")
    print()

    print("By Difficulty:")
    for diff, count in stats['difficulties'].items():
        print(f"  {diff}: {count}")
    print()

    print("Query Types:")
    for qtype, count in stats['query_types'].items():
        print(f"  {qtype}: {count}")