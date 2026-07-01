"""
Topic modeling pipeline: embed client turns → UMAP → HDBSCAN → LLM cluster labels.

Two modes:
  failure  — client turns from lost/abandoned conversations → surfaces failure patterns
  faq      — all client turns → surfaces what clients ask about most

Usage:
    python -m extraction.clustering --mode failure
    python -m extraction.clustering --mode faq
    python -m extraction.clustering --mode all   # runs both
"""

from __future__ import annotations

import argparse
import sqlite3
import uuid
from pathlib import Path

import hdbscan
import ollama
import umap
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).parent.parent.parent
DB_PATH = ROOT / "data" / "signals.db"

_MODEL: SentenceTransformer | None = None

MIN_TEXT_LEN = 4  # skip phone numbers, "+", single chars


def _get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        print("Loading LaBSE (~1.9GB, first run only)...")
        _MODEL = SentenceTransformer("LaBSE")
    return _MODEL


def _worth_embedding(text: str) -> bool:
    t = text.strip()
    if len(t) < MIN_TEXT_LEN:
        return False
    if t.startswith(("http://", "https://")):
        return False
    if t.replace("+", "").replace(" ", "").replace("-", "").replace("(", "").replace(")", "").isdigit():
        return False
    return True


def _label_cluster(texts: list[str], cluster_type: str) -> str:
    sample = texts[:12]
    if cluster_type == "failure":
        instruction = (
            "These are messages from customers who did NOT convert (lost conversations). "
            "What is the main reason or concern expressed? Summarize in 5 words or fewer. "
            "Reply in Russian or Kazakh, matching the language of the messages."
        )
    else:
        instruction = (
            "These are questions/messages from customers to a sales chatbot. "
            "What topic are they asking about? Summarize in 5 words or fewer. "
            "Reply in Russian or Kazakh, matching the language of the messages."
        )

    response = ollama.chat(
        model="gemma4:latest",
        messages=[{
            "role": "user",
            "content": instruction + "\n\n" + "\n".join(f"- {t}" for t in sample),
        }],
        options={"temperature": 0},
    )
    return response["message"]["content"].strip()


def run_clustering(
    conn: sqlite3.Connection,
    cluster_type: str,
    min_cluster_size: int = 5,
) -> None:
    print(f"\n── {cluster_type.upper()} clustering ──")

    if cluster_type == "failure":
        rows = conn.execute(
            """SELECT t.id, t.raw_text
               FROM turns t
               JOIN conversations c ON t.conversation_id = c.id
               WHERE t.role = 'client'
                 AND c.outcome IN ('lost', 'abandoned')
                 AND c.is_ai_conversation = 1
                 AND t.raw_text IS NOT NULL AND t.raw_text != ''"""
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT t.id, t.raw_text
               FROM turns t
               JOIN conversations c ON t.conversation_id = c.id
               WHERE t.role = 'client'
                 AND c.is_ai_conversation = 1
                 AND t.raw_text IS NOT NULL AND t.raw_text != ''"""
        ).fetchall()

    # Filter noise turns before embedding
    rows = [(tid, text) for tid, text in rows if _worth_embedding(text)]

    if not rows:
        print("No turns to cluster.")
        return

    turn_ids = [r[0] for r in rows]
    texts = [r[1] for r in rows]
    print(f"Embedding {len(texts)} turns...")

    model = _get_model()
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True)

    n_components = min(20, len(texts) - 1)
    print(f"UMAP: {embeddings.shape[1]}d → {n_components}d...")
    reducer = umap.UMAP(n_components=n_components, metric="cosine", random_state=42)
    reduced = reducer.fit_transform(embeddings)

    print(f"HDBSCAN (min_cluster_size={min_cluster_size})...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
    labels = clusterer.fit_predict(reduced)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    print(f"Found {n_clusters} clusters, {n_noise} noise points ({n_noise/len(labels)*100:.0f}%)")

    # Compute 2D coords for visualization (separate from 20D used for clustering)
    print("UMAP 2D for visualization...")
    reducer_2d = umap.UMAP(n_components=2, metric="cosine", random_state=42)
    coords_2d = reducer_2d.fit_transform(embeddings)

    # Free embeddings and model from memory before Ollama labeling calls
    del embeddings, reduced, reducer, reducer_2d
    global _MODEL
    _MODEL = None

    # Clear old clusters and coords of this type
    old_ids = [r[0] for r in conn.execute(
        "SELECT id FROM clusters WHERE cluster_type = ?", (cluster_type,)
    ).fetchall()]
    if old_ids:
        conn.execute(
            f"DELETE FROM cluster_members WHERE cluster_id IN ({','.join('?'*len(old_ids))})",
            old_ids,
        )
        conn.execute("DELETE FROM clusters WHERE cluster_type = ?", (cluster_type,))
    conn.execute("DELETE FROM turn_coords WHERE cluster_type = ?", (cluster_type,))

    # Build label→cluster_id map (assigned after sorting by size)
    cluster_map: dict[int, list[tuple[str, str]]] = {}
    for turn_id, text, label in zip(turn_ids, texts, labels):
        if label == -1:
            continue
        cluster_map.setdefault(label, []).append((turn_id, text))

    # Write 2D coords for every turn (noise points get cluster_id=None)
    label_to_cluster_id: dict[int, str] = {}  # filled below after cluster insertion

    # Sort by size descending so top clusters are easy to find
    for label_idx, (_, members) in enumerate(
        sorted(cluster_map.items(), key=lambda x: -len(x[1]))
    ):
        cluster_id = str(uuid.uuid4())
        sample_texts = [text for _, text in members]
        print(f"  Labeling cluster {label_idx + 1}/{n_clusters} ({len(members)} turns)...", end=" ", flush=True)
        cluster_label = _label_cluster(sample_texts, cluster_type)
        print(cluster_label)

        conn.execute(
            "INSERT INTO clusters (id, label, cluster_type, turn_count) VALUES (?,?,?,?)",
            (cluster_id, cluster_label, cluster_type, len(members)),
        )
        conn.executemany(
            "INSERT OR IGNORE INTO cluster_members (cluster_id, turn_id) VALUES (?,?)",
            [(cluster_id, tid) for tid, _ in members],
        )
        label_to_cluster_id[label_idx] = cluster_id  # note: label_idx not the HDBSCAN label

    # Build turn_id → hdbscan label map for coord writing
    # Re-derive mapping: sorted cluster order matches label_idx above
    sorted_hdbscan_labels = [
        hdb_label for hdb_label, _ in
        sorted(cluster_map.items(), key=lambda x: -len(x[1]))
    ]
    hdbscan_to_cluster_id = {
        hdb_label: label_to_cluster_id[i]
        for i, hdb_label in enumerate(sorted_hdbscan_labels)
    }

    conn.executemany(
        "INSERT OR REPLACE INTO turn_coords (turn_id, cluster_type, x, y, cluster_id) VALUES (?,?,?,?,?)",
        [
            (
                tid,
                cluster_type,
                float(coords_2d[i, 0]),
                float(coords_2d[i, 1]),
                hdbscan_to_cluster_id.get(int(lbl)),  # None for noise (-1)
            )
            for i, (tid, lbl) in enumerate(zip(turn_ids, labels))
        ],
    )

    conn.commit()
    print(f"{cluster_type} clustering done: {n_clusters} clusters written.")


def run(mode: str = "all", min_cluster_size: int = 5) -> None:
    conn = sqlite3.connect(DB_PATH)
    modes = ["failure", "faq"] if mode == "all" else [mode]
    for m in modes:
        run_clustering(conn, m, min_cluster_size)
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["failure", "faq", "all"], default="all")
    parser.add_argument("--min-cluster-size", type=int, default=5)
    args = parser.parse_args()
    run(mode=args.mode, min_cluster_size=args.min_cluster_size)
