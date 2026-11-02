"""Classify post sentiment (negative/neutral/positive) with a tweet-trained RoBERTa model."""
import argparse
import os

import duckdb
import torch
from transformers import pipeline

from src.config import GOLD_DIR, SENTIMENT_MODEL, SILVER_POSTS

POST_TOPICS = GOLD_DIR / "post_topics.parquet"


def run(limit: int | None = 2000, batch_size: int = 1) -> int:
    # Sample only posts that belong to a topic, so every scored post joins to topics.parquet.
    # The sample is in the outer query: USING SAMPLE inside would sample before the WHERE filter.
    limit_clause = f"USING SAMPLE {limit} ROWS" if limit else ""
    posts = duckdb.sql(f"""
        SELECT uri, clean_text FROM (
            SELECT s.uri, s.clean_text
            FROM read_parquet('{SILVER_POSTS.as_posix()}') s
            JOIN read_parquet('{POST_TOPICS.as_posix()}') t USING (uri)
            WHERE t.topic_id >= 0
        ) {limit_clause}
    """).df()

    
    torch.set_num_threads(os.cpu_count()) # Use all 4 CPU cores
    classifier = pipeline("sentiment-analysis", model=SENTIMENT_MODEL)
    print(f"Scoring sentiment for {len(posts):,} posts ...")

    texts = posts["clean_text"].tolist()
    results = []
    for start in range(0, len(texts), 100):
        results += classifier(texts[start:start + 100], batch_size=batch_size, truncation=True, max_length=128)
        print(f"{len(results):,} / {len(texts):,} scored")

    posts["sentiment"] = [r["label"] for r in results]
    posts["sentiment_score"] = [r["score"] for r in results]
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    posts[["uri", "sentiment", "sentiment_score"]].to_parquet(GOLD_DIR / "sentiment.parquet", index=False)
    print(posts["sentiment"].value_counts(normalize=True).round(3))
    return len(posts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=2000, help="0 = all topic posts (very slow on CPU)")
    args = parser.parse_args()
    run(args.limit or None)