"""Discover topics: UMAP (384 -> 5 dims) + HDBSCAN clustering + TF-IDF keywords per topic."""
import argparse

import duckdb
import numpy as np
import pandas as pd
import umap
from qdrant_client import QdrantClient
from sklearn.cluster import HDBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import COLLECTION, EMBEDDINGS_DIR, GOLD_DIR, QDRANT_URL, SILVER_POSTS
from src.load_qdrant import point_id


def load_posts(sample: int | None):
    sample_clause = f"USING SAMPLE {sample} ROWS" if sample else ""
    return duckdb.sql(f"""
        SELECT s.uri, s.clean_text, e.embedding
        FROM read_parquet('{SILVER_POSTS.as_posix()}') s
        JOIN read_parquet('{EMBEDDINGS_DIR.as_posix()}/*.parquet') e USING (uri)
        {sample_clause}
    """).df()


def topic_keywords(df: pd.DataFrame, top_n: int = 8) -> dict[int, str]:
    docs = df[df.topic_id >= 0].groupby("topic_id")["clean_text"].apply(" ".join)
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),                            # single words and 2-word phrases
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",    # letters only, 2+ chars
        max_df=0.8,                                    # ignore words that appear in >80% of topics
    )
    tfidf = vectorizer.fit_transform(docs.values)
    terms = vectorizer.get_feature_names_out()
    return {
        int(tid): ", ".join(terms[i] for i in row.toarray().ravel().argsort()[::-1][:top_n])
        for tid, row in zip(docs.index, tfidf)
    }


def representative_posts(df: pd.DataFrame, X: np.ndarray, n: int = 3) -> dict[int, list[str]]:
    reps = {}
    for tid, idx in df[df.topic_id >= 0].groupby("topic_id").indices.items():
        centroid = X[idx].mean(axis=0)
        closest = idx[np.argsort(X[idx] @ centroid)[::-1][:n]]
        reps[int(tid)] = df["clean_text"].iloc[closest].tolist()
    return reps


def write_topics_to_qdrant(df: pd.DataFrame, chunk: int = 1000) -> None:
    client = QdrantClient(url=QDRANT_URL)
    for tid, group in df.groupby("topic_id"):
        ids = [point_id(u) for u in group["uri"]]
        for i in range(0, len(ids), chunk):
            client.set_payload(COLLECTION, payload={"topic_id": int(tid)}, points=ids[i:i + chunk])


def run(sample: int | None = None, min_cluster_size: int = 50, n_neighbors: int = 15) -> int:
    df = load_posts(sample)
    X = np.vstack(df["embedding"].to_numpy())
    print(f"Clustering {len(df):,} posts ...")

    reduced = umap.UMAP(
        n_neighbors=n_neighbors, n_components=5, min_dist=0.0, metric="cosine", random_state=42
    ).fit_transform(X)
    df["topic_id"] = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=10).fit_predict(reduced)

    keywords = topic_keywords(df)
    reps = representative_posts(df, X)
    sizes = df[df.topic_id >= 0]["topic_id"].value_counts()
    topics = pd.DataFrame({
        "topic_id": sizes.index.astype(int),
        "size": sizes.values,
        "keywords": [keywords[t] for t in sizes.index],
        "examples": [reps[t] for t in sizes.index],
    })

    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    topics.to_parquet(GOLD_DIR / "topics.parquet", index=False)
    df[["uri", "topic_id"]].to_parquet(GOLD_DIR / "post_topics.parquet", index=False)
    write_topics_to_qdrant(df)

    noise = (df.topic_id == -1).mean()
    print(f"Found {len(topics)} topics; {noise:.0%} of posts are noise (-1)")
    print(topics[["topic_id", "size", "keywords"]].head(20).to_string(index=False))
    return len(topics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--min-cluster-size", type=int, default=50)
    parser.add_argument("--n-neighbors", type=int, default=15)
    args = parser.parse_args()
    run(args.sample, args.min_cluster_size, args.n_neighbors)