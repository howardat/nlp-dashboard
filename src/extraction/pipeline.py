"""
Extraction pipeline: load CSV → parse structure → write SQLite.

Populates conversations and turns tables only.
Clustering runs separately via extraction.clustering.

Usage:
    python -m extraction.pipeline
    python -m extraction.pipeline --sample 100
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import pandas as pd

from extraction.outcome import (
    detect_outcome,
    get_drop_off_turn,
    is_ai_conversation,
    normalize_role,
)

ROOT = Path(__file__).parent.parent.parent
DATA_PATH = ROOT / "data" / "DoroMarine.csv"
DB_PATH = ROOT / "data" / "signals.db"
SCHEMA_PATH = ROOT / "db" / "schema.sql"


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_PATH.read_text())
    conn.commit()


def conversation_id(row: pd.Series) -> str:
    return str(row.get("chat_id") or row.get("user_id") or row.name)


def already_processed(conn: sqlite3.Connection, conv_id: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM conversations WHERE id = ?", (conv_id,)
    ).fetchone() is not None


def process_conversation(
    conn: sqlite3.Connection,
    conv_id: str,
    msgs: list[dict],
    platform: str,
    created_at: str,
) -> None:
    if already_processed(conn, conv_id):
        return

    outcome = detect_outcome(msgs)
    is_ai = is_ai_conversation(msgs)
    drop_off = get_drop_off_turn(msgs, outcome)
    user_turns = [m for m in msgs if isinstance(m, dict) and m.get("role") == "user"]

    conn.execute(
        """INSERT INTO conversations
           (id, outcome, turn_count, user_turn_count, drop_off_turn,
            intervention_flag, is_ai_conversation, platform, created_at)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (
            conv_id, outcome, len(msgs), len(user_turns), drop_off,
            outcome == "human-closed", is_ai, platform, created_at,
        ),
    )

    for i, msg in enumerate(msgs):
        if not isinstance(msg, dict):
            continue
        role = normalize_role(msg)
        raw_text = str(msg.get("content", "")).strip()
        conn.execute(
            "INSERT INTO turns (id, conversation_id, turn_index, role, raw_text) VALUES (?,?,?,?,?)",
            (f"{conv_id}:{i}", conv_id, i, role, raw_text),
        )

    conn.commit()


def run(sample: int | None = None) -> None:
    DB_PATH.unlink(missing_ok=True)
    df = pd.read_csv(DATA_PATH)
    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    processed = skipped = errors = 0

    for _, row in df.iterrows():
        val = row.get("messages")
        if pd.isna(val) or not isinstance(val, str):
            continue
        try:
            msgs = json.loads(val)
        except json.JSONDecodeError:
            continue

        if not is_ai_conversation(msgs):
            skipped += 1
            continue

        conv_id = conversation_id(row)
        try:
            process_conversation(
                conn, conv_id, msgs,
                str(row.get("from_messenger", "")),
                str(row.get("created", "")),
            )
        except Exception as e:
            print(f"\nERROR {conv_id}: {e}")
            errors += 1
            continue

        processed += 1
        print(f"\r{processed} conversations loaded ({skipped} human-only skipped)", end="", flush=True)

        if sample and processed >= sample:
            break

    conn.close()
    print(f"\nDone. conversations={processed}, skipped={skipped}, errors={errors}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None)
    args = parser.parse_args()
    run(sample=args.sample)
