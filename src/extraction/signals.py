"""Batch turn-level labeling into turn_signals via OpenRouter. Resumable + concurrent.

Usage:
    export OPENROUTER_API_KEY=...
    python -m extraction.signals                 # full run
    python -m extraction.signals --sample 60     # smoke test
    python -m extraction.signals --workers 8 --model openai/gpt-4o-mini
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from extraction.labeler import label_agent_turn, label_client_turn

ROOT = Path(__file__).parent.parent.parent
DB_PATH = ROOT / "data" / "signals.db"
MIN_CLIENT_TURNS = 2
MIN_DISCOVERED = 8


def _turns_to_label(conn: sqlite3.Connection):
    """Client+agent turns in conversations with >=MIN_CLIENT_TURNS client turns,
    not yet labeled, with non-empty text."""
    return conn.execute(
        """
        SELECT t.id, t.conversation_id, t.turn_index, t.role, t.raw_text
        FROM turns t
        WHERE t.role IN ('client', 'agent')
          AND t.raw_text IS NOT NULL AND TRIM(t.raw_text) != ''
          AND t.conversation_id IN (
              SELECT conversation_id FROM turns
              WHERE role = 'client' AND TRIM(COALESCE(raw_text, '')) != ''
              GROUP BY conversation_id HAVING COUNT(*) >= ?
          )
          AND t.id NOT IN (SELECT turn_id FROM turn_signals)
        ORDER BY t.conversation_id, t.turn_index
        """,
        (MIN_CLIENT_TURNS,),
    ).fetchall()


def _label_row(row):
    tid, conv_id, idx, role, text = row
    fn = label_client_turn if role == "client" else label_agent_turn
    try:
        d = fn(text)
    except Exception as e:  # network/credit/parse failure -> DON'T write; leave unlabeled & resumable
        print(f"  warn {tid}: {e}", file=sys.stderr)
        d = None
    return (tid, conv_id, idx, role, d)


def run(sample: int | None = None, workers: int = 8, reset: bool = False) -> None:
    conn = sqlite3.connect(DB_PATH)
    if reset:
        conn.execute("DELETE FROM turn_signals")
        conn.commit()
        print("Cleared turn_signals (re-labeling from scratch).")
    rows = _turns_to_label(conn)
    if sample:
        rows = rows[:sample]
    print(f"Labeling {len(rows)} turns with {workers} workers...")

    done = failed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_label_row, r) for r in rows]
        for fut in as_completed(futures):
            tid, conv_id, idx, role, d = fut.result()
            if d is None:  # failed call — skip write so the turn stays unlabeled/resumable
                failed += 1
                continue
            conn.execute(
                """INSERT OR REPLACE INTO turn_signals
                   (turn_id, conversation_id, turn_index, role, intent,
                    intent_confidence, sentiment, sentiment_confidence, discovered)
                   VALUES (?,?,?,?,?,?,?,?,0)""",
                (tid, conv_id, idx, role, d["intent"], d["intent_confidence"],
                 d["sentiment"], d["sentiment_confidence"]),
            )
            done += 1
            if done % 25 == 0:
                conn.commit()
                print(f"\r{done}/{len(rows)}", end="", flush=True)
    conn.commit()
    print(f"\nLabeled {done} turns ({failed} failed, left unlabeled — re-run to retry).")
    _consolidate_discovered(conn)
    conn.close()


def _consolidate_discovered(conn: sqlite3.Connection) -> None:
    """Mark frequent off-taxonomy client intents as discovered; fold rare ones into 'other'."""
    from analysis.taxonomy import CLIENT_INTENTS

    seed = set(CLIENT_INTENTS)
    rows = conn.execute("SELECT intent FROM turn_signals WHERE role='client'").fetchall()
    counts = Counter(r[0] for r in rows if r[0] not in seed)
    keep = {lbl for lbl, n in counts.items() if n >= MIN_DISCOVERED}
    fold = {lbl for lbl in counts if lbl not in keep}
    if keep:
        conn.executemany(
            "UPDATE turn_signals SET discovered=1 WHERE role='client' AND intent=?",
            [(l,) for l in keep],
        )
    if fold:
        conn.executemany(
            "UPDATE turn_signals SET intent='other' WHERE role='client' AND intent=?",
            [(l,) for l in fold],
        )
    conn.commit()
    print(f"Discovered intents kept: {sorted(keep)}; folded {len(fold)} rare labels into 'other'.")


if __name__ == "__main__":
    import os

    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=None)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--model", default=None)
    p.add_argument("--reset", action="store_true", help="Clear turn_signals and re-label all turns")
    args = p.parse_args()
    if args.model:
        os.environ["OPENROUTER_MODEL"] = args.model
    run(sample=args.sample, workers=args.workers, reset=args.reset)
