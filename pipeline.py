"""End-to-end pipeline as a Prefect flow: collect -> silver -> embed -> load -> topics."""
from prefect import flow, task

from src import build_silver, embed_posts, ingest_jetstream, load_qdrant, topics


@task(retries=2, retry_delay_seconds=30)  # network can drop: try again
def collect(minutes: float) -> int:
    return ingest_jetstream.run(minutes)


@task
def silver() -> int:
    return build_silver.run()


@task
def embed() -> int:
    return embed_posts.run()  # incremental: only posts without a vector yet


@task(retries=2, retry_delay_seconds=10)  # Qdrant might be briefly unavailable
def load() -> int:
    return load_qdrant.run()


@task
def discover_topics(min_cluster_size: int) -> int:
    return topics.run(sample=None, min_cluster_size=min_cluster_size)


@flow(log_prints=True)
def bluesky_pipeline(collect_minutes: float = 5, min_cluster_size: int = 100, run_topics: bool = True):
    collected = collect(collect_minutes)
    print(f"Collected {collected:,} new messages")
    silver()
    embed()
    load()
    if run_topics:
        discover_topics(min_cluster_size)


if __name__ == "__main__":
    bluesky_pipeline()