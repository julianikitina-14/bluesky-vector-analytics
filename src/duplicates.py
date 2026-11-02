"""Find copy-paste spam: exact duplicates (SQL) and near-duplicates (vector similarity)."""
import argparse

import duckdb
import numpy as np
from qdrant_client import QdrantClient, models

from src.config import COLLECTION, EMBEDDINGS_DIR, QDRANT_URL, SILVER_POSTS


def exact_duplicates(min_authors: int = 3):
    return duckdb.sql(f"""
        SELECT any_value(clean_text) AS text,
               count(*)            AS posts,
               count(DISTINCT did) AS authors
        FROM read_parquet('{SILVER_POSTS.as_posix()}')
        WHERE length(clean_text) >= 30
        GROUP BY text_hash
        HAVING count(DISTINCT did) >= {min_authors}
        ORDER BY authors DESC
    """).df()


def near_duplicates(sample: int = 1000, threshold: float = 0.95, chunk: int = 100):
    df = duckdb.sql(f"""
        SELECT s.uri, s.did, s.clean_text, e.embedding
        FROM read_parquet('{SILVER_POSTS.as_posix()}') s
        JOIN read_parquet('{EMBEDDINGS_DIR.as_posix()}/*.parquet') e USING (uri)
        USING SAMPLE {sample} ROWS
    """).df()
    client = QdrantClient(url=QDRANT_URL)
    pairs = []
    for start in range(0, len(df), chunk):
        part = df.iloc[start:start + chunk]
        # query_batch_points sends many searches in one request: far fewer network round trips
        requests = [
            models.QueryRequest(query=np.asarray(v).tolist(), limit=6, score_threshold=threshold,
                                with_payload=["did", "text"])
            for v in part["embedding"]
        ]
        for row, response in zip(part.itertuples(), client.query_batch_points(COLLECTION, requests=requests)):
            for p in response.points:
                if p.payload["did"] != row.did:  # same author repeating themselves isn't interesting
                    pairs.append((round(p.score, 3), row.clean_text[:80], p.payload["text"][:80]))
    return pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=1000)
    args = parser.parse_args()
    print("Exact duplicates across >= 3 authors:")
    print(exact_duplicates().head(15).to_string(index=False))
    print("\nNear-duplicates from different authors:")
    for score, a, b in near_duplicates(args.sample)[:20]:
        print(f"{score}  {a!r}\n       {b!r}")