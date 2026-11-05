"""Streamlit dashboard: semantic search, topics, trends and sentiment over Bluesky posts."""
from datetime import datetime, timedelta, timezone

import duckdb
import pandas as pd
import streamlit as st
from qdrant_client import QdrantClient

from src.config import GOLD_DIR, QDRANT_URL
from src.search import semantic_search
from src.trends import trending_topics

st.set_page_config(page_title="Bluesky Vector Analytics 🦋", layout="wide")
st.title("Bluesky Vector Analytics")


@st.cache_resource
def get_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL)


@st.cache_data
def load_topics() -> pd.DataFrame:
    topics = pd.read_parquet(GOLD_DIR / "topics.parquet")
    topics["label"] = topics["topic_id"].astype(str) + ": " + topics["keywords"].str.slice(0, 40)
    return topics


@st.cache_data
def sentiment_by_topic(min_posts: int = 15) -> pd.DataFrame:
    counts = duckdb.sql(f"""
        SELECT t.topic_id, s.sentiment, count(*) AS posts
        FROM read_parquet('{(GOLD_DIR / "sentiment.parquet").as_posix()}') s
        JOIN read_parquet('{(GOLD_DIR / "post_topics.parquet").as_posix()}') t USING (uri)
        WHERE t.topic_id >= 0
        GROUP BY ALL
    """).df()
    pivot = counts.pivot_table(index="topic_id", columns="sentiment", values="posts", fill_value=0)
    pivot = pivot[pivot.sum(axis=1) >= min_posts]          # too few scored posts = meaningless shares
    shares = pivot.div(pivot.sum(axis=1), axis=0).round(3)
    labels = load_topics().set_index("topic_id")["label"]
    shares.index = shares.index.map(labels)                # show "23: keywords" instead of a bare number
    return shares.sort_values("negative", ascending=False)


topics = load_topics()
tab_search, tab_topics, tab_trends = st.tabs(["Semantic search", "Topics", "Trends & sentiment"])

with tab_search:
    query = st.text_input("Search by meaning", "feeling burned out at work")
    c1, c2, c3 = st.columns(3)
    limit = c1.slider("Results", 5, 50, 10)
    hours = c2.number_input("Only last N hours (0 = all)", min_value=0, value=0)
    topic_label = c3.selectbox("Topic", ["All"] + topics["label"].tolist())
    exclude_replies = st.checkbox("Exclude replies")

    if query:
        since = datetime.now(timezone.utc) - timedelta(hours=hours) if hours else None
        topic_id = None if topic_label == "All" else int(topic_label.split(":")[0])
        try:
            results = semantic_search(get_client(), query, limit, since, topic_id, exclude_replies)
        except Exception as exc:  # most often: Qdrant isn't running
            st.error(f"Search failed. Is Qdrant running (`docker compose ps`)?\n\n{exc}")
            results = []
        for p in results:
            st.markdown(f"**{p.score:.3f}** · {p.payload['created_at']}  \n{p.payload['text']}")
            st.divider()

with tab_topics:
    st.subheader(f"{len(topics)} discovered topics")
    st.bar_chart(topics.head(25).set_index("label")["size"], horizontal=True, sort=False)
    selected = st.selectbox("Inspect a topic", topics["label"].tolist())
    row = topics[topics["label"] == selected].iloc[0]
    st.write(f"**Keywords:** {row['keywords']}  ·  **Posts:** {row['size']:,}")
    for example in row["examples"]:
        st.info(example)

with tab_trends:
    recent_hours = st.slider("Recent window (hours)", 1, 12, 3)
    st.subheader("Fastest-growing topics")
    st.dataframe(trending_topics(recent_hours).head(15), hide_index=True)

    st.subheader("Sentiment by topic")
    st.caption("Share of scored posts per label; only topics with at least 15 scored posts (sample of 2,000).")
    st.bar_chart(sentiment_by_topic(), horizontal=True, sort=False,
                 color=["#d9534f", "#adb5bd", "#5cb85c"])  # negative, neutral, positive