"""Collect raw Bluesky post events from Jetstream into the bronze layer.

Output: data/bronze/date=YYYY-MM-DD/hour=HH/posts.jsonl, one raw JSON message per line.
Bronze is never modified; every later step can be re-run from these files.
"""
import argparse
import asyncio
import json
import time
from datetime import datetime, timezone

import websockets

from src.config import BRONZE_DIR, JETSTREAM_URL


def partition_path(event_time: datetime):
    """Hive-style partition folders (key=value) like you'd use in S3; DuckDB/Spark/Athena understand them."""
    folder = BRONZE_DIR / f"date={event_time:%Y-%m-%d}" / f"hour={event_time:%H}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / "posts.jsonl"


async def collect(minutes: float, max_messages: int | None = None) -> int:
    deadline = time.monotonic() + minutes * 60
    count = 0
    current_path, out = None, None
    cursor = None  # time_us of the last event we saved; lets us resume after a disconnect

    try:
        while time.monotonic() < deadline:
            # On reconnect, ask Jetstream to replay from where we stopped. This can re-send a few
            # events we already have; that's fine because silver deduplicates on (did, rkey).
            url = JETSTREAM_URL if cursor is None else f"{JETSTREAM_URL}&cursor={cursor}"
            try:
                async with websockets.connect(url, max_size=None, ping_interval=20) as ws:
                    print("Connected to Jetstream")
                    async for raw in ws:
                        msg = json.loads(raw)
                        cursor = msg.get("time_us", cursor)

                        event_time = datetime.fromtimestamp(msg["time_us"] / 1_000_000, tz=timezone.utc)
                        path = partition_path(event_time)
                        if path != current_path:  # crossed into a new hour: switch files
                            if out:
                                out.close()
                            out = open(path, "a", encoding="utf-8")
                            current_path = path

                        out.write(json.dumps(msg, ensure_ascii=False) + "\n")
                        count += 1
                        if count % 1000 == 0:
                            out.flush()
                            print(f"{count:,} messages saved")

                        if time.monotonic() >= deadline or (max_messages and count >= max_messages):
                            return count
            except (websockets.ConnectionClosed, OSError) as e:
                print(f"Connection lost ({e}); reconnecting in 5 s...")
                await asyncio.sleep(5)
    finally:
        if out:
            out.close()
    return count


def run(minutes: float = 10, max_messages: int | None = None) -> int:
    n = asyncio.run(collect(minutes, max_messages))
    print(f"Done. Saved {n:,} messages under {BRONZE_DIR}")
    return n


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect Bluesky posts into bronze")
    parser.add_argument("--minutes", type=float, default=10)
    parser.add_argument("--max-messages", type=int, default=None)
    args = parser.parse_args()
    run(args.minutes, args.max_messages)