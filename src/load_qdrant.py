"""Load silver posts + their embeddings into Qdrant (idempotent upsert)."""
import uuid

import duckdb
import numpy as np
from qdrant_client import QdrantClient, models

from src.config import COLLECTION, EMBEDDINGS_DIR, QDRANT_URL, SILVER_POSTS, VECTOR_SIZE

PAYLOAD_INDEXES = {
    "created_at": models.PayloadSchemaType.DATETIME,
    "hashtags": models.PayloadSchemaType.KEYWORD,
    "is_reply": models.PayloadSchemaType.BOOL,
    "topic_id": models.PayloadSchemaType.INTEGER,  # filled in later by topics.py
}


def point_id(uri: str) -> str:
    """Deterministic UUID: same post -> same id -> re-loading overwrites instead of duplicating."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, uri))


def ensure_collection(client: QdrantClient) -> None:
    if client.collection_exists(COLLECTION):
        return
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
    )
    for field, schema in PAYLOAD_INDEXES.items():
        client.create_payload_index(COLLECTION, field_name=field, field_schema=schema)
    print(f"Created collection '{COLLECTION}'")


def run() -> int:
    df = duckdb.sql(f"""
        SELECT s.uri, s.did, s.text, s.created_at, s.is_reply, s.hashtags, e.embedding
        FROM read_parquet('{SILVER_POSTS.as_posix()}') s
        JOIN read_parquet('{EMBEDDINGS_DIR.as_posix()}/*.parquet') e USING (uri)
    """).df()

    client = QdrantClient(url=QDRANT_URL)
    ensure_collection(client)

    vectors = np.vstack(df["embedding"].to_numpy())
    payloads = [
        {
            "uri": r.uri,
            "did": r.did,
            "text": r.text,
            "created_at": r.created_at.strftime("%Y-%m-%dT%H:%M:%SZ"),  # RFC 3339, UTC
            "is_reply": bool(r.is_reply),
            "hashtags": list(r.hashtags),
        }
        for r in df.itertuples()
    ]
    client.upload_collection(
        collection_name=COLLECTION,
        vectors=vectors,
        payload=payloads,
        ids=[point_id(u) for u in df["uri"]],
        batch_size=256,  # sent in chunks; one giant request would be slow and fragile
    )
    total = client.count(COLLECTION, exact=True).count
    print(f"Upserted {len(df):,} points; collection now has {total:,}")
    return len(df)


if __name__ == "__main__":
    run()