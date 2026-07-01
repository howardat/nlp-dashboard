# Conversation Signals Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add sentiment-trajectory, Markov-transition, and intent-pattern analytics to the DoroMarine dashboard, built on a shared OpenRouter per-turn labeling pass.

**Architecture:** A new `extraction/signals.py` batch-labels every turn in the 352 multi-turn conversations via OpenRouter into a new `turn_signals` SQLite table. Pure, testable `analysis/*` modules compute features/transitions/patterns from that table. `app.py` gains three thin rendering tabs.

**Tech Stack:** Python 3.13, SQLite, OpenAI SDK (→ OpenRouter), scikit-learn, scipy, pandas, Streamlit, graphviz DOT.

## Global Constraints

- Analysis corpus = conversations with ≥2 client turns (352: closed 133 + human-closed 110 + lost 109). Abandoned (236) excluded from sequence analysis, kept in outcome totals.
- `converted = outcome in {closed, human-closed}`.
- OpenRouter key from env `OPENROUTER_API_KEY`; model from env `OPENROUTER_MODEL` default `openai/gpt-4o-mini`. Key never written to disk/logs.
- `analysis/*` modules must not import streamlit or open the DB; they take DataFrames/sequences and return DataFrames/structures.
- Labeling must be resumable (skip turns already in `turn_signals`).
- Run commands from `src/` so `extraction`/`analysis`/`utils` import as top-level packages (matches existing pipeline).

---

### Task 1: Dependencies + `turn_signals` schema

**Files:**
- Modify: `pyproject.toml` (add deps)
- Modify: `db/schema.sql` (add table)

**Interfaces:**
- Produces: `turn_signals` table; `openai`, `scikit-learn`, `scipy` importable.

- [ ] **Step 1:** Add to `pyproject.toml` dependencies: `"openai>=1.0.0"`, `"scikit-learn>=1.5.0"`, `"scipy>=1.14.0"`. Run `uv sync`.
- [ ] **Step 2:** Append to `db/schema.sql`:

```sql
CREATE TABLE IF NOT EXISTS turn_signals (
  turn_id TEXT PRIMARY KEY REFERENCES turns(id),
  conversation_id TEXT,
  turn_index INTEGER,
  role TEXT,                 -- 'client' | 'agent'
  intent TEXT,
  intent_confidence REAL,
  sentiment REAL,            -- client turns only; NULL for agent
  sentiment_confidence REAL,
  discovered INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_turn_signals_conv ON turn_signals(conversation_id);
```

- [ ] **Step 3:** Apply to existing DB: `sqlite3 data/signals.db < db/schema.sql` (idempotent — `IF NOT EXISTS`). Verify: `sqlite3 data/signals.db ".tables"` shows `turn_signals`.
- [ ] **Step 4:** Commit (if committing enabled).

---

### Task 2: `analysis/taxonomy.py` — label inventories

**Files:**
- Create: `src/analysis/__init__.py` (empty)
- Create: `src/analysis/taxonomy.py`
- Test: `tests/test_taxonomy.py`

**Interfaces:**
- Produces: `CLIENT_INTENTS: list[str]`, `AGENT_ACTS: list[str]`, `canonical_intent(label, role) -> str`, `terminal_state(outcome) -> str`.

- [ ] **Step 1:** Write `src/analysis/taxonomy.py`:

```python
"""Intent / agent-act taxonomy and state mapping for conversation analysis."""
from __future__ import annotations

CLIENT_INTENTS = [
    "greeting", "price_inquiry", "product_info", "dosage_safety",
    "objection_price", "objection_trust", "objection_timing",
    "ready_to_buy", "shares_contact", "request_human", "complaint", "other",
]

AGENT_ACTS = [
    "greeting", "asked_for_contact", "gave_price", "answered_objection",
    "provided_info", "follow_up", "other",
]

_CLIENT_SET = set(CLIENT_INTENTS)
_AGENT_SET = set(AGENT_ACTS)


def canonical_intent(label: str, role: str) -> str:
    """Map a raw model label to a known label, else 'other'. Used before discovery merge."""
    l = (label or "").strip().lower().replace(" ", "_")
    allowed = _CLIENT_SET if role == "client" else _AGENT_SET
    return l if l in allowed else l or "other"  # keep unknown non-empty for discovery; '' -> 'other'


def terminal_state(outcome: str) -> str:
    """Absorbing Markov state for a conversation outcome."""
    return "CLOSED" if outcome in ("closed", "human-closed") else "LOST"
```

- [ ] **Step 2:** Write `tests/test_taxonomy.py`:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from analysis.taxonomy import canonical_intent, terminal_state

def test_known_client_intent_passthrough():
    assert canonical_intent("price_inquiry", "client") == "price_inquiry"

def test_label_normalized():
    assert canonical_intent("Price Inquiry", "client") == "price_inquiry"

def test_empty_label_is_other():
    assert canonical_intent("", "client") == "other"

def test_unknown_label_kept_for_discovery():
    assert canonical_intent("wants_refund", "client") == "wants_refund"

def test_terminal_state():
    assert terminal_state("closed") == "CLOSED"
    assert terminal_state("human-closed") == "CLOSED"
    assert terminal_state("lost") == "LOST"
    assert terminal_state("abandoned") == "LOST"
```

- [ ] **Step 3:** Run `cd /Users/oljk/Projects/nlp-dashboard && python -m pytest tests/test_taxonomy.py -v` → PASS.
- [ ] **Step 4:** Commit.

---

### Task 3: `extraction/labeler.py` — OpenRouter client + prompts

**Files:**
- Modify: `src/extraction/labeler.py` (replace Ollama path with OpenRouter; keep module name)
- Test: `tests/test_labeler_parse.py`

**Interfaces:**
- Consumes: `analysis.taxonomy` (CLIENT_INTENTS, AGENT_ACTS).
- Produces: `label_client_turn(text) -> dict`, `label_agent_turn(text) -> dict`, `_parse_label(raw, role) -> dict` (pure, testable), `get_client()` (lazy OpenAI client).

- [ ] **Step 1:** Write `tests/test_labeler_parse.py` (pure parser, no network):

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from extraction.labeler import _parse_label

def test_parse_client_full():
    raw = '{"intent":"objection_price","intent_confidence":0.9,"sentiment":-0.4,"sentiment_confidence":0.8}'
    d = _parse_label(raw, "client")
    assert d["intent"] == "objection_price"
    assert d["sentiment"] == -0.4
    assert d["intent_confidence"] == 0.9

def test_parse_agent_act_key():
    raw = '{"act":"asked_for_contact","confidence":0.7}'
    d = _parse_label(raw, "agent")
    assert d["intent"] == "asked_for_contact"
    assert d["intent_confidence"] == 0.7
    assert d["sentiment"] is None

def test_parse_clamps_sentiment():
    d = _parse_label('{"intent":"x","sentiment":5}', "client")
    assert d["sentiment"] == 1.0

def test_parse_bad_json_returns_other():
    d = _parse_label("not json", "client")
    assert d["intent"] == "other"
    assert d["intent_confidence"] == 0.0
```

- [ ] **Step 2:** Write `src/extraction/labeler.py`:

```python
"""Turn-level labeler backed by OpenRouter (OpenAI-compatible API)."""
from __future__ import annotations

import json
import os

from openai import OpenAI

from analysis.taxonomy import AGENT_ACTS, CLIENT_INTENTS

_CLIENT: OpenAI | None = None
MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")

CLIENT_SYSTEM = (
    "You analyze a single CUSTOMER message sent to an AI sales agent for a children's "
    "marine supplement (DoroMarine). Messages are Russian, Kazakh, or mixed, often misspelled.\n"
    "Return ONLY JSON: {\"intent\": <label>, \"intent_confidence\": 0-1, "
    "\"sentiment\": -1..1, \"sentiment_confidence\": 0-1}.\n"
    "Pick intent from this list when one fits: " + ", ".join(CLIENT_INTENTS) + ".\n"
    "If none fit, invent a short snake_case label. sentiment: -1 angry/frustrated, 0 neutral, 1 enthusiastic."
)
AGENT_SYSTEM = (
    "You classify a single AI-SALES-AGENT message (RU/KZ/mixed). "
    "Return ONLY JSON: {\"act\": <label>, \"confidence\": 0-1}.\n"
    "Pick act from: " + ", ".join(AGENT_ACTS) + ". If none fit, use 'other'."
)


def get_client() -> OpenAI:
    global _CLIENT
    if _CLIENT is None:
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY not set in environment.")
        _CLIENT = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
    return _CLIENT


def _clamp(v, lo, hi, default=0.0):
    try:
        return max(lo, min(hi, float(v)))
    except (TypeError, ValueError):
        return default


def _parse_label(raw: str, role: str) -> dict:
    try:
        d = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {"intent": "other", "intent_confidence": 0.0, "sentiment": None, "sentiment_confidence": 0.0}
    if role == "agent":
        return {
            "intent": str(d.get("act", "other")).strip().lower().replace(" ", "_") or "other",
            "intent_confidence": _clamp(d.get("confidence"), 0, 1),
            "sentiment": None,
            "sentiment_confidence": 0.0,
        }
    return {
        "intent": str(d.get("intent", "other")).strip().lower().replace(" ", "_") or "other",
        "intent_confidence": _clamp(d.get("intent_confidence"), 0, 1),
        "sentiment": _clamp(d.get("sentiment"), -1, 1, None) if d.get("sentiment") is not None else None,
        "sentiment_confidence": _clamp(d.get("sentiment_confidence"), 0, 1),
    }


def _label(text: str, role: str) -> dict:
    system = CLIENT_SYSTEM if role == "client" else AGENT_SYSTEM
    resp = get_client().chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": text[:2000]}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return _parse_label(resp.choices[0].message.content or "", role)


def label_client_turn(text: str) -> dict:
    return _label(text, "client")


def label_agent_turn(text: str) -> dict:
    return _label(text, "agent")
```

Note `_clamp(..., None)` for sentiment: if value present but unparseable, returns `None` default. Verify the `default=None` path: call signature `_clamp(d.get("sentiment"), -1, 1, None)`.

- [ ] **Step 3:** Run `cd /Users/oljk/Projects/nlp-dashboard && python -m pytest tests/test_labeler_parse.py -v` → PASS.
- [ ] **Step 4:** Commit.

---

### Task 4: `extraction/signals.py` — batch labeling orchestrator

**Files:**
- Create: `src/extraction/signals.py`

**Interfaces:**
- Consumes: `extraction.labeler` (label_client_turn, label_agent_turn), `analysis.taxonomy`.
- Produces: CLI `python -m extraction.signals [--sample N] [--workers W] [--model M]`; populates `turn_signals`.

- [ ] **Step 1:** Write `src/extraction/signals.py`:

```python
"""Batch turn-level labeling into turn_signals via OpenRouter. Resumable + concurrent."""
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
    """Client+agent turns in conversations with >=2 client turns, not yet in turn_signals, non-empty."""
    return conn.execute(
        """
        SELECT t.id, t.conversation_id, t.turn_index, t.role, t.raw_text
        FROM turns t
        JOIN conversations c ON t.conversation_id = c.id
        WHERE t.role IN ('client','agent')
          AND t.raw_text IS NOT NULL AND TRIM(t.raw_text) != ''
          AND c.conversation_id IN (  -- placeholder; replaced below
              SELECT conversation_id FROM conversations
          )
          AND t.conversation_id IN (
              SELECT conversation_id FROM turns
              WHERE role='client' AND TRIM(COALESCE(raw_text,'')) != ''
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
    except Exception as e:  # network/parse failure -> low-confidence 'other', keep going
        print(f"  warn {tid}: {e}", file=sys.stderr)
        d = {"intent": "other", "intent_confidence": 0.0, "sentiment": None, "sentiment_confidence": 0.0}
    return (tid, conv_id, idx, role, d)


def run(sample: int | None = None, workers: int = 8) -> None:
    conn = sqlite3.connect(DB_PATH)
    rows = _turns_to_label(conn)
    if sample:
        rows = rows[:sample]
    print(f"Labeling {len(rows)} turns with {workers} workers...")

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_label_row, r) for r in rows]
        for fut in as_completed(futures):
            tid, conv_id, idx, role, d = fut.result()
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
    print(f"\nLabeled {done} turns.")
    _consolidate_discovered(conn)
    conn.close()


def _consolidate_discovered(conn: sqlite3.Connection) -> None:
    """Mark frequent off-taxonomy client intents as discovered; fold rare ones into 'other'."""
    from analysis.taxonomy import CLIENT_INTENTS
    seed = set(CLIENT_INTENTS)
    rows = conn.execute(
        "SELECT intent FROM turn_signals WHERE role='client'"
    ).fetchall()
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
    args = p.parse_args()
    if args.model:
        os.environ["OPENROUTER_MODEL"] = args.model
    run(sample=args.sample, workers=args.workers)
```

Fix the SQL: remove the placeholder sub-select block (it was illustrative). Final WHERE uses only the `t.conversation_id IN (... HAVING COUNT>=?)` and `t.id NOT IN (turn_signals)` clauses. The implementer must delete the `c.conversation_id IN (SELECT conversation_id FROM conversations)` placeholder lines.

- [ ] **Step 2:** Syntax check: `cd src && python -c "import ast,pathlib; ast.parse(pathlib.Path('extraction/signals.py').read_text())"` → no error.
- [ ] **Step 3:** Dry-run query only (no key needed): `cd src && python -c "import sqlite3,extraction.signals as s; c=sqlite3.connect(s.DB_PATH); print(len(s._turns_to_label(c)))"` → prints a count (~3000–4000).
- [ ] **Step 4:** Commit.
- [ ] **Step 5 (run, needs key — execute at integration, Task 10):** `export OPENROUTER_API_KEY=...; cd src && python -m extraction.signals`.

---

### Task 5: `analysis/sentiment.py` — trajectory features + correlation

**Files:**
- Create: `src/analysis/sentiment.py`
- Test: `tests/test_sentiment.py`

**Interfaces:**
- Consumes: a DataFrame `signals` with columns `conversation_id, turn_index, role, sentiment` and a DataFrame `convs` with `conversation_id, converted` (bool).
- Produces: `trajectory_features(signals) -> DataFrame` (one row/conv, columns: conversation_id, mean, final, min, max_drop, slope, neg_run_len, start_to_end, n_client_turns); `correlations(features, convs) -> DataFrame` (feature, r, p_value, p_adj, direction); `avg_trajectory(signals, convs, bins) -> DataFrame`.

- [ ] **Step 1:** Write `tests/test_sentiment.py`:

```python
import sys, os
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from analysis.sentiment import trajectory_features, correlations

def _sig(conv, vals):
    return pd.DataFrame([
        {"conversation_id": conv, "turn_index": i, "role": "client", "sentiment": v}
        for i, v in enumerate(vals)
    ])

def test_features_basic():
    df = _sig("a", [0.5, 0.0, -0.5])
    f = trajectory_features(df).set_index("conversation_id").loc["a"]
    assert f["final"] == -0.5
    assert f["min"] == -0.5
    assert f["max_drop"] == 1.0          # 0.5 -> -0.5
    assert f["n_client_turns"] == 3
    assert f["slope"] < 0

def test_neg_run_len():
    df = _sig("a", [0.2, -0.1, -0.2, -0.3, 0.1])
    f = trajectory_features(df).set_index("conversation_id").loc["a"]
    assert f["neg_run_len"] == 3

def test_correlation_direction():
    feats = pd.DataFrame({
        "conversation_id": ["a","b","c","d"],
        "final": [0.9, 0.8, -0.8, -0.9],
    })
    convs = pd.DataFrame({"conversation_id": ["a","b","c","d"], "converted": [True,True,False,False]})
    c = correlations(feats, convs)
    row = c[c["feature"] == "final"].iloc[0]
    assert row["r"] > 0  # higher final sentiment -> converted
```

- [ ] **Step 2:** Write `src/analysis/sentiment.py`:

```python
"""Sentiment trajectory features and their correlation with conversion."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

FEATURE_COLS = ["mean", "final", "min", "max_drop", "slope", "neg_run_len", "start_to_end", "n_client_turns"]


def _max_drop(vals: np.ndarray) -> float:
    """Largest peak-to-trough decline along the series (>=0)."""
    peak = vals[0]
    drop = 0.0
    for v in vals:
        peak = max(peak, v)
        drop = max(drop, peak - v)
    return float(drop)


def _neg_run(vals: np.ndarray) -> int:
    best = cur = 0
    for v in vals:
        cur = cur + 1 if v < 0 else 0
        best = max(best, cur)
    return best


def trajectory_features(signals: pd.DataFrame) -> pd.DataFrame:
    cl = signals[(signals["role"] == "client") & signals["sentiment"].notna()]
    out = []
    for conv, g in cl.sort_values("turn_index").groupby("conversation_id"):
        v = g["sentiment"].to_numpy(dtype=float)
        if len(v) == 0:
            continue
        slope = float(np.polyfit(np.arange(len(v)), v, 1)[0]) if len(v) > 1 else 0.0
        out.append({
            "conversation_id": conv,
            "mean": float(v.mean()),
            "final": float(v[-1]),
            "min": float(v.min()),
            "max_drop": _max_drop(v),
            "slope": slope,
            "neg_run_len": _neg_run(v),
            "start_to_end": float(v[-1] - v[0]),
            "n_client_turns": int(len(v)),
        })
    return pd.DataFrame(out, columns=["conversation_id"] + FEATURE_COLS)


def correlations(features: pd.DataFrame, convs: pd.DataFrame) -> pd.DataFrame:
    df = features.merge(convs[["conversation_id", "converted"]], on="conversation_id")
    y = df["converted"].astype(int).to_numpy()
    rows = []
    feats = [c for c in FEATURE_COLS if c in df.columns]
    for f in feats:
        x = df[f].to_numpy(dtype=float)
        if np.std(x) == 0:
            r, p = 0.0, 1.0
        else:
            r, p = stats.pointbiserialr(y, x)
        rows.append({"feature": f, "r": float(r), "p_value": float(p)})
    res = pd.DataFrame(rows)
    # Benjamini-Hochberg
    m = len(res)
    res = res.sort_values("p_value").reset_index(drop=True)
    res["p_adj"] = (res["p_value"] * m / (res.index + 1)).clip(upper=1.0)
    res["p_adj"] = res["p_adj"][::-1].cummin()[::-1]
    res["direction"] = np.where(res["r"] >= 0, "higher → converts", "higher → loses")
    return res.sort_values("r", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)


def avg_trajectory(signals: pd.DataFrame, convs: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Mean sentiment per normalized-progress bin, split by converted. Columns: bin, converted, mean, n."""
    cl = signals[(signals["role"] == "client") & signals["sentiment"].notna()].merge(
        convs[["conversation_id", "converted"]], on="conversation_id"
    )
    rows = []
    for conv, g in cl.sort_values("turn_index").groupby("conversation_id"):
        n = len(g)
        conv_flag = bool(g["converted"].iloc[0])
        for i, (_, r) in enumerate(g.iterrows()):
            b = int(i / n * bins) if n > 1 else 0
            rows.append({"bin": min(b, bins - 1), "converted": conv_flag, "sentiment": r["sentiment"]})
    d = pd.DataFrame(rows)
    if d.empty:
        return d
    return (d.groupby(["bin", "converted"])["sentiment"]
            .agg(["mean", "count"]).reset_index().rename(columns={"count": "n"}))
```

- [ ] **Step 3:** Run `cd /Users/oljk/Projects/nlp-dashboard && python -m pytest tests/test_sentiment.py -v` → PASS.
- [ ] **Step 4:** Commit.

---

### Task 6: `analysis/markov.py` — transitions + worst edges

**Files:**
- Create: `src/analysis/markov.py`
- Test: `tests/test_markov.py`

**Interfaces:**
- Consumes: `sequences: dict[conv_id -> list[str]]` (state labels in order, NOT yet including terminal), `convs` DataFrame with `conversation_id, converted, outcome`.
- Produces: `build_sequences(signals, convs) -> dict`; `transition_table(sequences, convs, min_support=5) -> DataFrame` (from_state, to_state, count, prob, conv_rate, p_lost); `worst_transitions(table, n) -> DataFrame`; `to_dot(table, ...) -> str`.

- [ ] **Step 1:** Write `tests/test_markov.py`:

```python
import sys, os
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from analysis.markov import transition_table, worst_transitions

def test_transition_counts_and_terminal():
    seqs = {"a": ["greeting", "price_inquiry"], "b": ["greeting", "objection_price"]}
    convs = pd.DataFrame({
        "conversation_id": ["a", "b"],
        "converted": [True, False],
        "outcome": ["closed", "lost"],
    })
    t = transition_table(seqs, convs, min_support=1)
    # greeting->price_inquiry leads to CLOSED ; greeting->objection_price leads to LOST
    gp = t[(t.from_state=="greeting") & (t.to_state=="price_inquiry")].iloc[0]
    assert gp["count"] == 1
    op = t[(t.from_state=="objection_price") & (t.to_state=="LOST")].iloc[0]
    assert op["conv_rate"] == 0.0

def test_worst_orders_by_low_conversion():
    seqs = {f"c{i}": ["greeting", "objection_price"] for i in range(5)}
    seqs.update({f"d{i}": ["greeting", "ready_to_buy"] for i in range(5)})
    convs = pd.DataFrame({
        "conversation_id": list(seqs),
        "converted": [False]*5 + [True]*5,
        "outcome": ["lost"]*5 + ["closed"]*5,
    })
    t = transition_table(seqs, convs, min_support=2)
    w = worst_transitions(t, n=1).iloc[0]
    assert w["to_state"] in ("objection_price", "LOST")
    assert w["conv_rate"] == 0.0
```

- [ ] **Step 2:** Write `src/analysis/markov.py`:

```python
"""First-order Markov transition model over conversation states with terminal CLOSED/LOST."""
from __future__ import annotations

import pandas as pd

from analysis.taxonomy import terminal_state


def build_sequences(signals: pd.DataFrame, convs: pd.DataFrame) -> dict[str, list[str]]:
    """Interleave client intents + agent acts by turn_index per conversation."""
    seqs: dict[str, list[str]] = {}
    for conv, g in signals.sort_values("turn_index").groupby("conversation_id"):
        seqs[conv] = g["intent"].tolist()
    return seqs


def transition_table(sequences, convs, min_support: int = 5) -> pd.DataFrame:
    out = convs.set_index("conversation_id")[["converted", "outcome"]].to_dict("index")
    # accumulate per-edge: count, converted_count, lost_count
    agg: dict[tuple[str, str], list[int]] = {}
    for conv, states in sequences.items():
        meta = out.get(conv)
        if meta is None:
            continue
        full = list(states) + [terminal_state(meta["outcome"])]
        conv_ok = 1 if meta["converted"] else 0
        for a, b in zip(full, full[1:]):
            rec = agg.setdefault((a, b), [0, 0, 0])
            rec[0] += 1
            rec[1] += conv_ok
            rec[2] += 0 if conv_ok else 1
    rows = []
    for (a, b), (cnt, convc, lostc) in agg.items():
        rows.append({
            "from_state": a, "to_state": b, "count": cnt,
            "conv_rate": convc / cnt, "p_lost": lostc / cnt,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    totals = df.groupby("from_state")["count"].transform("sum")
    df["prob"] = df["count"] / totals
    df = df[df["count"] >= min_support].reset_index(drop=True)
    return df.sort_values("count", ascending=False).reset_index(drop=True)


def worst_transitions(table: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    if table.empty:
        return table
    t = table.copy()
    t["badness"] = (1 - t["conv_rate"]) * t["count"]
    return t.sort_values(["conv_rate", "count"], ascending=[True, False]).head(n).reset_index(drop=True)


def to_dot(table: pd.DataFrame, max_edges: int = 40) -> str:
    """Graphviz DOT: nodes shaded by avg conv_rate of incoming, edges labeled with prob."""
    if table.empty:
        return "digraph G { label=\"no data\"; }"
    t = table.sort_values("count", ascending=False).head(max_edges)
    # node color by mean conv_rate of edges touching it (as source)
    node_rate = t.groupby("from_state")["conv_rate"].mean().to_dict()
    lines = ["digraph G {", "rankdir=LR;", 'node [style=filled, fontname="Helvetica"];']
    states = set(t["from_state"]) | set(t["to_state"])
    for s in states:
        r = node_rate.get(s, 0.5)
        # red (low) -> green (high)
        red = int(255 * (1 - r)); grn = int(255 * r)
        color = f"#{red:02x}{grn:02x}88"
        shape = "doublecircle" if s in ("CLOSED", "LOST") else "box"
        lines.append(f'"{s}" [fillcolor="{color}", shape={shape}];')
    for _, e in t.iterrows():
        pen = 1 + 4 * e["prob"]
        lines.append(f'"{e.from_state}" -> "{e.to_state}" [penwidth={pen:.1f}, label="{e.prob:.0%}"];')
    lines.append("}")
    return "\n".join(lines)
```

- [ ] **Step 3:** Run `cd /Users/oljk/Projects/nlp-dashboard && python -m pytest tests/test_markov.py -v` → PASS.
- [ ] **Step 4:** Commit.

---

### Task 7: `analysis/patterns.py` — archetypes + lost patterns

**Files:**
- Create: `src/analysis/patterns.py`
- Test: `tests/test_patterns.py`

**Interfaces:**
- Consumes: `sequences: dict[conv_id -> list[str]]` (client-intent sequences), `convs` DataFrame (`conversation_id, converted`).
- Produces: `archetypes(sequences, convs, k_range=range(3,9)) -> DataFrame` (conversation_id, cluster); `archetype_summary(assignments, convs) -> DataFrame` (cluster, size, conv_rate, top_intents); `lost_patterns(sequences, convs, n_range=(2,4), top=15) -> DataFrame` (pattern, length, count, conv_rate, score).

- [ ] **Step 1:** Write `tests/test_patterns.py`:

```python
import sys, os
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from analysis.patterns import lost_patterns

def test_lost_pattern_surfaces_frequent_loss_ngram():
    seqs = {}
    for i in range(6):
        seqs[f"l{i}"] = ["greeting", "objection_price", "request_human"]  # all lost
    for i in range(6):
        seqs[f"w{i}"] = ["greeting", "ready_to_buy", "shares_contact"]    # all won
    convs = pd.DataFrame({
        "conversation_id": list(seqs),
        "converted": [False]*6 + [True]*6,
    })
    lp = lost_patterns(seqs, convs, top=10)
    pats = [tuple(p) for p in lp["pattern"]]
    assert ("objection_price", "request_human") in pats
    row = lp[lp["pattern"].apply(lambda p: tuple(p) == ("objection_price","request_human"))].iloc[0]
    assert row["conv_rate"] == 0.0
    assert row["count"] == 6
```

- [ ] **Step 2:** Write `src/analysis/patterns.py`:

```python
"""Conversation archetype clustering + frequent lost-conversation intent patterns."""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score


def _docs(sequences: dict[str, list[str]]):
    ids = list(sequences)
    docs = [" ".join(sequences[i]) for i in ids]
    return ids, docs


def archetypes(sequences, convs, k_range=range(3, 9)) -> pd.DataFrame:
    ids, docs = _docs(sequences)
    if len(ids) < max(k_range):
        k_range = range(2, max(3, len(ids) // 2 + 1))
    vec = TfidfVectorizer(ngram_range=(1, 2), token_pattern=r"[^\s]+")
    X = vec.fit_transform(docs)
    best_k, best_score, best_labels = None, -1.0, None
    for k in k_range:
        if k >= len(ids):
            continue
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
        try:
            s = silhouette_score(X, km.labels_)
        except ValueError:
            continue
        if s > best_score:
            best_k, best_score, best_labels = k, s, km.labels_
    if best_labels is None:
        best_labels = np.zeros(len(ids), dtype=int)
    return pd.DataFrame({"conversation_id": ids, "cluster": best_labels})


def archetype_summary(assignments: pd.DataFrame, convs: pd.DataFrame,
                      sequences: dict[str, list[str]]) -> pd.DataFrame:
    df = assignments.merge(convs[["conversation_id", "converted"]], on="conversation_id")
    rows = []
    for cl, g in df.groupby("cluster"):
        intents = Counter()
        for cid in g["conversation_id"]:
            intents.update(sequences.get(cid, []))
        top = ", ".join(lbl for lbl, _ in intents.most_common(4))
        rows.append({
            "cluster": int(cl), "size": len(g),
            "conv_rate": float(g["converted"].mean()),
            "top_intents": top,
        })
    return pd.DataFrame(rows).sort_values("conv_rate").reset_index(drop=True)


def lost_patterns(sequences, convs, n_range=(2, 4), top: int = 15) -> pd.DataFrame:
    conv_map = convs.set_index("conversation_id")["converted"].to_dict()
    pat_count: Counter = Counter()
    pat_conv: defaultdict = defaultdict(int)
    for cid, states in sequences.items():
        if cid not in conv_map:
            continue
        seen = set()
        for n in range(n_range[0], n_range[1] + 1):
            for i in range(len(states) - n + 1):
                pat = tuple(states[i:i + n])
                if pat in seen:
                    continue  # count each pattern once per conversation
                seen.add(pat)
                pat_count[pat] += 1
                if conv_map[cid]:
                    pat_conv[pat] += 1
    rows = []
    for pat, cnt in pat_count.items():
        if cnt < 3:
            continue
        conv_rate = pat_conv[pat] / cnt
        rows.append({
            "pattern": list(pat), "length": len(pat), "count": cnt,
            "conv_rate": conv_rate, "score": cnt * (1 - conv_rate),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("score", ascending=False).head(top).reset_index(drop=True)
```

- [ ] **Step 3:** Run `cd /Users/oljk/Projects/nlp-dashboard && python -m pytest tests/test_patterns.py -v` → PASS.
- [ ] **Step 4:** Commit.

---

### Task 8: Extract `render_messages` helper

**Files:**
- Modify: `src/utils/chat.py` (add `render_messages`)
- Modify: `src/app.py` (Chat Viewer tab calls helper)

**Interfaces:**
- Produces: `utils.chat.render_messages(messages: list[dict]) -> None` (Streamlit rendering of the message loop currently inlined in app.py tab 4).

- [ ] **Step 1:** Move the message-rendering `for msg in messages:` loop (app.py ~lines 683–735, the role-based `st.chat_message` rendering + Raw JSON expander) into `utils/chat.py::render_messages(messages)`. Import `streamlit as st` and `json` inside `utils/chat.py`.
- [ ] **Step 2:** Replace the inlined loop in app.py Chat Viewer with `from utils.chat import render_messages` (add to existing import) and `render_messages(messages)`.
- [ ] **Step 3:** Syntax check: `cd src && python -c "import ast,pathlib; ast.parse(pathlib.Path('app.py').read_text()); ast.parse(pathlib.Path('utils/chat.py').read_text())"`.
- [ ] **Step 4:** Commit.

---

### Task 9: Three dashboard tabs

**Files:**
- Modify: `src/app.py` (add 3 tabs; add a shared `load_turn_signals()` reader)

**Interfaces:**
- Consumes: `analysis.sentiment`, `analysis.markov`, `analysis.patterns`, `utils.chat.render_messages`.

- [ ] **Step 1:** Extend the `st.tabs([...])` list with `"Sentiment Trajectory"`, `"Transitions"`, `"Intent Patterns"` and unpack three more tab variables.
- [ ] **Step 2:** Add a cached loader near `load_data`:

```python
@st.cache_data
def load_signals_frames():
    conn = load_signals_db()
    if conn is None:
        return None, None
    sig = pd.read_sql("SELECT conversation_id, turn_index, role, intent, intent_confidence, sentiment, sentiment_confidence FROM turn_signals", conn)
    convs = pd.read_sql("SELECT id as conversation_id, outcome FROM conversations WHERE is_ai_conversation=1", conn)
    conn.close()
    if sig.empty:
        return None, None
    convs["converted"] = convs["outcome"].isin(["closed", "human-closed"])
    convs = convs[convs["conversation_id"].isin(sig["conversation_id"].unique())]
    return sig, convs
```

- [ ] **Step 3:** Sentiment tab — compute `trajectory_features`, `correlations`, `avg_trajectory`; render a Plotly line (two series converted/lost over normalized bins), the correlation table, and a one-line readout. Show empty-state info with the `python -m extraction.signals` command if `load_signals_frames()` returns None.
- [ ] **Step 4:** Transitions tab — `build_sequences` (client+agent), `transition_table`, `to_dot` → `st.graphviz_chart(dot)`, plus `worst_transitions` table. Min-support slider (default 5).
- [ ] **Step 5:** Intent Patterns tab — client-only sequences: `archetypes` + `archetype_summary` table/bar; `lost_patterns` rendered as `st.expander` cards. For examples, look up conversation rows from the main `df` by `_conv_id` and call `render_messages(parse_messages(row["messages"]))` for 2 examples per pattern.
- [ ] **Step 6:** Run the app: `cd src && streamlit run app.py` — manually verify all three tabs render the empty-state before labeling, and (after Task 10) the real charts. Capture no exceptions in console.
- [ ] **Step 7:** Commit.

---

### Task 10: Integration run + verification

**Files:** none (operational)

- [ ] **Step 1:** With key set: `export OPENROUTER_API_KEY=...; cd src && python -m extraction.signals --sample 60 --workers 6` — verify rows appear: `sqlite3 ../data/signals.db "SELECT role, COUNT(*) FROM turn_signals GROUP BY role;"` and inspect a few labels for sanity (RU/KZ intents reasonable).
- [ ] **Step 2:** Full run: `python -m extraction.signals --workers 8`. Confirm discovered-intents print line and final count (~3000–4000).
- [ ] **Step 3:** Launch `streamlit run app.py`; verify each tab: sentiment trajectory diverges won vs lost; transition graph renders with CLOSED/LOST nodes; archetypes show varying conv rates; lost patterns expand to real transcripts.
- [ ] **Step 4:** Sanity-check counts: analysis corpus conversation count ≈ 352; correlation table lists 8 features; no tab throws.
- [ ] **Step 5:** Commit any fixups.

---

## Self-Review

**Spec coverage:** turn_signals (T1) ✓; hybrid taxonomy + discovery (T2,T4 `_consolidate_discovered`) ✓; OpenRouter labeling client+agent (T3,T4) ✓; resumable/concurrent/sample (T4) ✓; sentiment trajectory + correlation + BH (T5) ✓; avg trajectory viz (T5,T9) ✓; Markov client+agent states + terminal + worst + graph (T6,T9) ✓; archetypes + lost patterns + expandable examples (T7,T9) ✓; render_messages reuse (T8) ✓; deps (T1) ✓; empty-state guards (T9) ✓; honesty captions (T9, fold into render) ✓; tests for analysis/* (T5–T7) ✓.

**Placeholder scan:** Task 4 contains an intentional illustrative SQL placeholder with an explicit instruction to delete it — flagged, not silent. No other TBD/TODO.

**Type consistency:** `sequences` is `dict[str, list[str]]` across markov/patterns; `convs` always has `conversation_id, converted` (+`outcome` for markov); `trajectory_features`→`correlations` share `FEATURE_COLS`; `archetype_summary` takes `sequences` (added to signature in T7 — consumers in T9 pass it). `build_sequences` returns the dict consumed by `transition_table`/`archetypes`.

**Note:** `archetype_summary` signature is `(assignments, convs, sequences)` — Task 9 Step 5 must pass all three.
