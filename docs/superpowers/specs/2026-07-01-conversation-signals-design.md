# Design: Sentiment Trajectory, Markov Transitions & Intent Patterns

**Date:** 2026-07-01
**Status:** Approved design (pending spec review)

## Goal

Add three analytics features to the DoroMarine dashboard, built on a shared per-turn labeling pass:

1. **Sentiment trajectory** + correlation of trajectory features with conversation success/failure.
2. **Markov transition model** over conversation states, surfacing the worst-converting transitions.
3. **Intent labeling + conversation-archetype clustering** with per-cluster success rate, plus the top recurring intent patterns in *lost* conversations with expandable real examples.

## Decisions (locked)

- **Intent taxonomy:** hybrid — fixed seed taxonomy + LLM-discovered extras consolidated by frequency.
- **Markov states:** client intents **+** key agent acts (interleaved by turn index) + absorbing terminal states.
- **Modeling depth:** LLM "silver" labels only (no distilled classifier).
- **Conversation scope for sequence analysis:** the **352 multi-turn conversations** (≥2 client turns = closed 133 + human-closed 110 + lost 109). The 236 single-message "abandoned" chats are excluded from trajectory/Markov/pattern analysis but remain in outcome totals.
- **Success definition:** `converted = closed OR human-closed` (consistent with the existing Overview tab). human-closed = conversion via human takeover; noted as such.
- **Labeling backend:** OpenRouter (OpenAI-compatible API). Key from `OPENROUTER_API_KEY`. Model from `OPENROUTER_MODEL`, default `openai/gpt-4o-mini`.

## Architecture & new files

```
src/
  extraction/
    signals.py        # NEW orchestrator: batch turn-level labeling → turn_signals; hybrid discovery; resumable; concurrent
    labeler.py        # extend: OpenRouter client + intent/act prompts (replaces Ollama path for signals)
  analysis/           # NEW package — pure functions, no Streamlit, unit-testable
    __init__.py
    taxonomy.py       # seed intent + agent-act inventories, terminal-state mapping
    sentiment.py      # trajectory features + correlation stats
    markov.py         # transition matrix, worst transitions, DOT graph builder
    patterns.py       # archetype clustering + lost-pattern n-gram mining
  app.py              # 3 new tabs: Sentiment Trajectory · Transitions · Intent Patterns
db/schema.sql         # + turn_signals table
tests/                # NEW: unit tests for analysis/* on synthetic sequences
```

`analysis/*` modules take a DataFrame / list-of-sequences and return DataFrames or plain structures. The dashboard is a thin rendering layer over them. This keeps the math testable without Streamlit or a live DB.

## Data model

```sql
CREATE TABLE IF NOT EXISTS turn_signals (
  turn_id TEXT PRIMARY KEY REFERENCES turns(id),
  conversation_id TEXT,
  turn_index INTEGER,
  role TEXT,                 -- 'client' | 'agent'
  intent TEXT,               -- client intent OR agent act (canonical label)
  intent_confidence REAL,
  sentiment REAL,            -- client turns only: -1.0..1.0 ; NULL for agent turns
  sentiment_confidence REAL,
  discovered INTEGER DEFAULT 0  -- 1 if intent came from the discovery pass
);
CREATE INDEX IF NOT EXISTS idx_turn_signals_conv ON turn_signals(conversation_id);
```

## Labeling pipeline (`extraction/signals.py` + `labeler.py`)

- **Client turn → JSON:** `{ "intent": <seed-label|new short label>, "intent_confidence": 0-1, "sentiment": -1..1, "sentiment_confidence": 0-1 }`.
- **Agent turn → JSON:** `{ "act": <agent-act label>, "confidence": 0-1 }`.
- **Seed client intents:** `greeting, price_inquiry, product_info, dosage_safety, objection_price, objection_trust, objection_timing, ready_to_buy, shares_contact, request_human, complaint, other`.
- **Agent acts:** `greeting, asked_for_contact, gave_price, answered_objection, provided_info, follow_up, other`.
- **Prompts:** system prompt states the taxonomy + definitions, instructs RU/KZ/code-switch/misspelling tolerance, allows a short free-text intent only when nothing fits (→ candidate for discovery). `response_format={"type":"json_object"}`; reasoning discouraged (return JSON only). temperature 0.
- **Hybrid discovery pass:** after first pass, collect free-text intents; any appearing ≥ `MIN_DISCOVERED=8` times becomes a canonical intent (`discovered=1`); rarer ones map to `other`. (No re-labeling round for v1 — the kept labels are already assigned.)
- **Execution:** `ThreadPoolExecutor` (default 8 workers) with a simple token-bucket rate limit; retry with backoff on 429/5xx; **resumable** (skip turns already in `turn_signals`); `--sample N`, `--workers`, `--model` flags; progress to stderr.
- **Key handling:** read `OPENROUTER_API_KEY` from env; fail fast with a clear message if missing. Never logged or persisted.
- **Scope:** labels turns belonging to conversations with ≥2 client turns (the 352-conversation analysis set).

## Feature 1 — Sentiment trajectory (`analysis/sentiment.py`)

- **Per-conversation features** from the ordered client-turn sentiment series: `mean, final, min, max_drop` (largest peak→trough decline), `slope` (OLS over turn index), `neg_run_len` (longest consecutive-negative run), `start_to_end_delta`, `n_client_turns`.
- **Correlation:** point-biserial (Pearson) of each feature vs `converted`; **logistic regression** over all features → standardized coefficients + ROC-AUC (5-fold CV); per-feature p-values with **Benjamini–Hochberg** correction. Flag only BH-significant features as real separators.
- **Dashboard tab "Sentiment Trajectory":**
  - Average sentiment-by-turn line, converted vs lost, with 95% bands — shown two ways: absolute turn index (first ~12) and normalized progress (10 bins).
  - Correlation table (feature, r, p_adj, direction) + logistic AUC headline.
  - Short plain-language readout (e.g. "lost conversations show a sharper late-conversation sentiment drop").

## Feature 2 — Markov transitions (`analysis/markov.py`)

- **State sequence per conversation:** interleave client intents and agent acts in `turn_index` order; append terminal **CLOSED** (closed/human-closed) or **LOST**.
- **Matrix:** first-order transition counts → row-normalized probabilities.
- **Worst transitions:** for each edge X→Y with support ≥ `MIN_SUPPORT` (default 5): conversion rate of conversations containing it, frequency, and `P(reach LOST | traversed X→Y)`. Ranked by low conversion rate × support.
- **Dashboard tab "Transitions":**
  - Transition **graph** via `st.graphviz_chart` (DOT string): nodes shaded by avg conversion (red→green), edges weighted/labeled by probability; light edges (below support floor) pruned.
  - "Top worst transitions" table: from, to, support, conversion %, P(LOST).

## Feature 3 — Intent patterns (`analysis/patterns.py`)

1. **Conversation archetypes:** vectorize each conversation as TF-IDF over its intent uni-/bi-grams → KMeans (k chosen by silhouette over a small range, e.g. 3–8). Per cluster: size, conversion rate, top characteristic intents. Rendered as a table + bar chart, sortable, worst conversion highlighted.
2. **Top lost patterns:** frequent **contiguous intent n-grams (n=2..4)** among lost conversations, ranked by `frequency × (1 − conversion_rate_of_convos_containing_it)` (i.e. frequent *and* loss-associated). Each pattern → expandable card: the intent sequence, count, conversion rate, and 2–3 example conversations rendered with the existing Chat Viewer message renderer (extracted into a shared helper).
- **Dashboard tab "Intent Patterns":** archetypes section + lost-patterns section.

## Dashboard integration

- Extract the message-rendering loop from the Chat Viewer tab into `utils/chat.py::render_messages(messages)` so Feature 3 can reuse it.
- All three tabs show an info message with the exact command to run if `turn_signals` is empty (mirrors the existing Failure Patterns tab pattern).
- `st.cache_data` on the analysis computations keyed by row counts.

## Dependencies

Add to `pyproject.toml`: `openai` (OpenRouter client), `scikit-learn`, `scipy` (KMeans/silhouette/logistic/stats; already transitively present via umap-learn — pinned explicitly). No graph library needed (DOT string). PrefixSpan avoided (contiguous n-grams instead).

## Validation & honesty

- Surface label **confidence distributions** in each tab's caption; LLM labels framed as "silver."
- Correlation claims gated on BH-significance.
- Markov/archetype tables show **support counts**; low-support rows flagged.
- human-closed conversions labeled as human-takeover, not pure-AI.

## Out of scope (v1)

- Distilled/trained classifier; gold-set + Cohen's κ validation harness; PrefixSpan gapped subsequences; graph community detection on the transition graph; migrating the existing `clustering.py` cluster-labeler off Ollama (can follow later).

## Testing

`tests/` unit tests for `analysis/sentiment.py` (feature math on synthetic series), `analysis/markov.py` (transition counts/worst-edge on hand-built sequences), `analysis/patterns.py` (n-gram mining + archetype shape). No network/LLM in tests.
