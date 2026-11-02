"""Semantic search over Bluesky posts: meaning-based, with metadata filters."""
import argparse
from datetime import datetime
from functools import lru_cache

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from src.config import COLLECTION, EMBEDDING_MODEL, QDRANT_URL


@lru_cache  # load the model once per process, not on every search
def get_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


def semantic_search(client: QdrantClient, query: str, limit: int = 10,
                    since: datetime | None = None, topic_id: int | None = None,
                    exclude_replies: bool = False):
    # The query MUST be embedded with the same model (and normalization) as the stored posts.
    vector = get_model().encode(query, normalize_embeddings=True).tolist()

    must = []
    if since:
        must.append(models.FieldCondition(key="created_at", range=models.DatetimeRange(gte=since)))
    if topic_id is not None:
        must.append(models.FieldCondition(key="topic_id", match=models.MatchValue(value=topic_id)))
    if exclude_replies:
        must.append(models.FieldCondition(key="is_reply", match=models.MatchValue(value=False)))

    result = client.query_points(
        collection_name=COLLECTION,
        query=vector,
        query_filter=models.Filter(must=must) if must else None,
        limit=limit,
        with_payload=True,
    )
    return result.points


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    for p in semantic_search(QdrantClient(url=QDRANT_URL), args.query, args.limit):
        print(f"{p.score:.3f}  {p.payload['text'][:120]!r}")