"""Embed silver posts that don't have an embedding yet; write one new Parquet part per run."""
import argparse
from datetime import datetime, timezone

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL, EMBEDDINGS_DIR, SILVER_POSTS, VECTOR_SIZE


def find_unembedded_posts(limit: int | None = None):
    sql = f"SELECT uri, clean_text FROM read_parquet('{SILVER_POSTS.as_posix()}')"
    if any(EMBEDDINGS_DIR.glob("*.parquet")):
        sql += f"""
            WHERE uri NOT IN (
                SELECT uri FROM read_parquet('{EMBEDDINGS_DIR.as_posix()}/*.parquet')
            )"""
    if limit:
        sql += f" LIMIT {limit}"
    return duckdb.sql(sql).df()


def run(limit: int | None = None, batch_size: int = 16) -> int:
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    posts = find_unembedded_posts(limit)
    if posts.empty:
        print("Nothing new to embed.")
        return 0

    print(f"Embedding {len(posts):,} posts with {EMBEDDING_MODEL} ...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    vectors = model.encode(
        posts["clean_text"].tolist(),
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    # FixedSizeList stores vectors compactly (exactly VECTOR_SIZE floats per row).
    table = pa.table({
        "uri": pa.array(posts["uri"]),
        "embedding": pa.FixedSizeListArray.from_arrays(pa.array(vectors.ravel()), VECTOR_SIZE),
        "model": pa.array([EMBEDDING_MODEL] * len(posts)),
    })
    out = EMBEDDINGS_DIR / f"part-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}.parquet"
    pq.write_table(table, out)
    print(f"Wrote {len(posts):,} embeddings -> {out}")
    return len(posts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="embed at most N posts (for testing)")
    # 16 was fastest on this 4-core laptop; bigger batches pad more short posts to the longest one.
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    run(args.limit, args.batch_size)