"""Trending topics: compare each topic's share of posts in the recent window vs. earlier."""
import argparse

import duckdb

from src.config import GOLD_DIR, SILVER_POSTS


def trending_topics(recent_hours: int = 3, min_posts: int = 20):
    return duckdb.sql(f"""
        WITH tagged AS (
            SELECT t.topic_id,
                   s.created_at > (SELECT max(created_at) FROM read_parquet('{SILVER_POSTS.as_posix()}'))
                                  - INTERVAL {recent_hours} HOUR AS is_recent
            FROM read_parquet('{SILVER_POSTS.as_posix()}') s
            JOIN read_parquet('{(GOLD_DIR / "post_topics.parquet").as_posix()}') t USING (uri)
            WHERE t.topic_id >= 0
        ),
        counts AS (
            SELECT topic_id,
                   count(*) FILTER (WHERE is_recent)     AS recent_posts,
                   count(*) FILTER (WHERE NOT is_recent) AS earlier_posts
            FROM tagged GROUP BY topic_id
        ),
        shares AS (
            SELECT *,
                   recent_posts  / sum(recent_posts)  OVER () AS recent_share,
                   earlier_posts / sum(earlier_posts) OVER () AS earlier_share
            FROM counts
        )
        SELECT sh.topic_id, tp.keywords, sh.recent_posts, sh.earlier_posts,
               round(sh.recent_share * 100, 2)  AS recent_pct,
               round(sh.earlier_share * 100, 2) AS earlier_pct,
               round(sh.recent_share / nullif(sh.earlier_share, 0), 2) AS growth
        FROM shares sh
        JOIN read_parquet('{(GOLD_DIR / "topics.parquet").as_posix()}') tp USING (topic_id)
        WHERE sh.recent_posts + sh.earlier_posts >= {min_posts}
        ORDER BY growth DESC NULLS FIRST
    """).df()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--recent-hours", type=int, default=3)
    args = parser.parse_args()
    print(trending_topics(args.recent_hours).head(15).to_string(index=False))