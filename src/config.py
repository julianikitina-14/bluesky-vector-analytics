"""All project settings in one place, so changing a model or path is a one-line edit."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"

SILVER_POSTS = SILVER_DIR / "posts.parquet"
EMBEDDINGS_DIR = GOLD_DIR / "embeddings"

# Jetstream: Bluesky's free JSON stream. wantedCollections filters to posts only (no likes/follows).
JETSTREAM_URL = (
    "wss://jetstream2.us-east.bsky.network/subscribe"
    "?wantedCollections=app.bsky.feed.post"
)

# If you change the model you MUST change VECTOR_SIZE, recreate the collection, and re-embed.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_SIZE = 384

SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

QDRANT_URL = "http://localhost:6333"
COLLECTION = "bluesky_posts"