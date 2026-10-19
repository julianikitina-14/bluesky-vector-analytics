"""Build the silver layer: one clean row per English Bluesky post, deletes applied."""
from pathlib import Path

import duckdb

from src.config import BRONZE_DIR, SILVER_POSTS

MIN_TEXT_LENGTH = 20


def silver_sql(bronze_dir: Path, silver_path: Path) -> str:
    bronze_glob = f"{bronze_dir.as_posix()}/**/*.jsonl"
    # Double braces {{ }} are literal braces inside a Python f-string.
    return rf"""
    COPY (
        WITH events AS (
            SELECT
                did,
                time_us,
                json_extract_string(commit, '$.operation') AS operation,
                json_extract_string(commit, '$.rkey')      AS rkey,
                commit
            FROM read_json(
                '{bronze_glob}',
                format = 'newline_delimited',
                columns = {{did: 'VARCHAR', time_us: 'BIGINT', kind: 'VARCHAR', commit: 'JSON'}}
            )
            WHERE kind = 'commit'
        ),
        deleted AS (
            SELECT DISTINCT did, rkey FROM events WHERE operation = 'delete'
        ),
        created AS (
            SELECT
                did,
                rkey,
                make_timestamp(time_us)                          AS created_at,  -- server time, UTC
                json_extract_string(commit, '$.record.text')     AS text,
                json_extract_string(commit, '$.record.langs[*]') AS langs,       -- list of strings
                json_extract(commit, '$.record.reply') IS NOT NULL AS is_reply,
                -- number duplicates of the same post; we keep rn = 1
                row_number() OVER (PARTITION BY did, rkey ORDER BY time_us) AS rn
            FROM events
            WHERE operation = 'create'
        ),
        cleaned AS (
            SELECT
                c.*,
                trim(regexp_replace(regexp_replace(c.text, 'https?://\S+', '', 'g'), '\s+', ' ', 'g')) AS clean_text
            FROM created c
            LEFT JOIN deleted d USING (did, rkey)
            WHERE c.rn = 1
              AND d.did IS NULL                -- anti-join: drop posts that were deleted
              AND list_contains(c.langs, 'en')
        )
        SELECT
            'at://' || did || '/app.bsky.feed.post/' || rkey AS uri,
            did,
            rkey,
            created_at,
            is_reply,
            text,
            clean_text,
            regexp_extract_all(lower(text), '#(\w+)', 1)     AS hashtags,
            md5(lower(clean_text))                           AS text_hash
        FROM cleaned
        WHERE length(clean_text) >= {MIN_TEXT_LENGTH}
    ) TO '{silver_path.as_posix()}' (FORMAT parquet)
    """


def build_silver(bronze_dir: Path = BRONZE_DIR, silver_path: Path = SILVER_POSTS) -> int:
    silver_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(silver_sql(bronze_dir, silver_path))
    rows = con.execute(f"SELECT count(*) FROM read_parquet('{silver_path.as_posix()}')").fetchone()[0]
    print(f"Silver built: {rows:,} posts -> {silver_path}")
    return rows


def run() -> int:
    return build_silver()


if __name__ == "__main__":
    run()