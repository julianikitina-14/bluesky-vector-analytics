# Findings

Running notes on what the data showed. Raw material for the README (Phase 7).

## Dataset
- **43,390** English posts in silver, collected from the Bluesky Jetstream over about 24 hours: **2026-10-18 11:06 → 2026-10-19 11:20 UTC**.
- Embeddings: `all-MiniLM-L6-v2` (384 dims), stored in Qdrant (`bluesky_posts`).

## Topics (5.2)

**Final command:** `python -m src.topics --min-cluster-size 100` → **72 topics, 47% noise (-1)**

### Experiments
| Settings | Topics | Noise | Notes |
|---|---|---|---|
| `--min-cluster-size 50` (default), all posts | 115 | 47% | More, smaller topics |
| `--min-cluster-size 100`, all posts | **72** | **47%** | **Chosen** |

`random_state=42` makes runs reproducible: the same settings give the same topics (UMAP then runs on one core, hence the `n_jobs` warning).

### Largest topics
| Topic | Posts | Keywords |
|---|---|---|
| 23 | 1,222 | goal, rfef, league, tercera |
| 22 | 1,183 | tune, music, radio, playing |
| 44 | 828 | ukraine, putin, russia, war |
| 29 | 780 | ai, data, slop, claude, agents |
| 0 | 726 | dlsite, nowplaying, fanza |
| 52 | 714 | coffee, espresso, chocolate, breakfast |
| 9 | 659 | que, la, el, por, para |
| 57 | 644 | vote, early voting, republican, election |
| 65 | 623 | movie, halloween, horror, spooky |
| 56 | 590 | sleep, pain, awake, bed, work |

### Data quality issues the topics exposed
- **Non-English posts tagged as English.** Topic 9 (`que, la, el, por, para`) is Spanish, and topic 23 is mostly Spanish football (Tercera RFEF match scores). Bluesky's `langs` field is *declared by the client*, not detected, so the Phase 2 English filter lets these through. A language-detection step would fix it.
- **Bots and automated accounts form their own topics.** Topic 22 is "Now playing on KB Radio: …" posts. Topic 0 is promotional spam (`dlsite`, `fanza`). Topic 61 (`ting, bong, dring, ding`) looks like clock or bell bots.
- **Bug found and fixed:** the "representative example posts" per topic didn't match their topics. Positions came from the noise-filtered DataFrame but were used to index the full one. The same post even showed up as an example for two topics. Fixed by grouping the full DataFrame and skipping `-1`. Lesson: `.iloc` doesn't complain when positions point at the wrong rows. Always eyeball the output.

## Sentiment (5.3)
- Model: `cardiffnlp/twitter-roberta-base-sentiment-latest`, on a **random sample of 2,000** topic posts (CPU only, ~3.4 posts/s).
- Overall: **49% neutral, 27% negative, 24% positive.**
- 53 topics have at least 15 scored posts (the minimum for a meaningful share).

**Most negative** (share of negative posts)
| Topic | Keywords | Scored | Negative |
|---|---|---|---|
| 58 | trump, traitor, treason, doj | 21 | 95% |
| 33 | cnn, media, fox, news | 15 | 80% |
| 20 | israel, gaza, palestinians | 43 | 74% |
| 35 | trans, gender, women | 21 | 67% |
| 39 | racist, nazi, slavery | 15 | 67% |

**Most positive:** greetings (63 "good morning", 62 "happy monday/sunday") and 67 "thank you" are 100% positive. 50 (travel, Scotland) is 50% positive.

**Caveat:** 15–43 scored posts per topic is a small sample. Treat the percentages as indicative, not precise. More data: `python -m src.sentiment --limit 5000`.

## Trends (5.4)
Fastest-growing topics, comparing the last 3 hours of the data with the rest:

| Topic | Keywords | Growth |
|---|---|---|
| 61 | ting, bong, dring (bell bots) | 1.97× |
| 38 | eu, uk, europe, brexit | 1.91× |
| 42 | iran, saudi, houthi, yemen | 1.80× |
| 25 | telescope, astronomy, stargazing | 1.64× |
| 43 | oil, diesel, climate, prices, energy | 1.63× |

The top "trend" is a bot, so filtering automated accounts would make trends more meaningful.

## Dashboard (Phase 6)
- `streamlit run app.py` → three tabs: semantic search, topics, trends & sentiment.
- `st.bar_chart` sorts bars **alphabetically by label** by default. `sort=False` keeps the order you set in pandas (largest or most negative first).
- The search tab ranks by meaning, so it needs a query. Topic, time and reply filters only narrow the search; they don't browse.

## Ideas for later
- Detect language instead of trusting `langs` (would remove topics 9 and 23).
- Flag bot accounts (many near-identical posts from one author) and exclude them from trends.
- Score more posts for sentiment, so the per-topic shares are more reliable.
