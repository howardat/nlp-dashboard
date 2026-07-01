"""First-order Markov transition model over conversation states with terminal CLOSED/LOST."""
from __future__ import annotations

from collections import defaultdict

import pandas as pd

from analysis.taxonomy import terminal_state


def build_sequences(signals: pd.DataFrame, convs: pd.DataFrame | None = None) -> dict[str, list[str]]:
    """Interleave client intents + agent acts by turn_index per conversation."""
    seqs: dict[str, list[str]] = {}
    for conv, g in signals.sort_values("turn_index").groupby("conversation_id"):
        seqs[conv] = g["intent"].tolist()
    return seqs


def transition_table(sequences, convs, min_support: int = 5) -> pd.DataFrame:
    meta = convs.set_index("conversation_id")[["converted", "outcome"]].to_dict("index")
    agg: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0, 0])  # count, converted, lost
    for conv, states in sequences.items():
        m = meta.get(conv)
        if m is None:
            continue
        full = list(states) + [terminal_state(m["outcome"])]
        conv_ok = 1 if m["converted"] else 0
        for a, b in zip(full, full[1:]):
            rec = agg[(a, b)]
            rec[0] += 1
            rec[1] += conv_ok
            rec[2] += 0 if conv_ok else 1
    rows = [
        {"from_state": a, "to_state": b, "count": cnt, "conv_rate": convc / cnt, "p_lost": lostc / cnt}
        for (a, b), (cnt, convc, lostc) in agg.items()
    ]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    totals = df.groupby("from_state")["count"].transform("sum")
    df["prob"] = df["count"] / totals
    df = df[df["count"] >= min_support].reset_index(drop=True)
    return df.sort_values("count", ascending=False).reset_index(drop=True)


TERMINAL_STATES = ("CLOSED", "LOST")


def worst_transitions(table: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    """Lowest-converting transitions between real (non-terminal) states.

    Edges *into* CLOSED/LOST are excluded: their conversion rate is 0 or 1 by
    definition of the terminal state, so they carry no behavioral signal."""
    if table.empty:
        return table
    t = table[~table["to_state"].isin(TERMINAL_STATES)].copy()
    if t.empty:
        return t
    t["badness"] = (1 - t["conv_rate"]) * t["count"]
    return t.sort_values(["conv_rate", "count"], ascending=[True, False]).head(n).reset_index(drop=True)


def to_dot(table: pd.DataFrame, max_edges: int = 40) -> str:
    """Graphviz DOT: nodes shaded by avg conv_rate (red low -> green high), edges labeled with prob."""
    if table.empty:
        return 'digraph G { label="no data"; }'
    t = table.sort_values("count", ascending=False).head(max_edges)
    node_rate = t.groupby("from_state")["conv_rate"].mean().to_dict()
    lines = ["digraph G {", "rankdir=LR;", 'node [style=filled, fontname="Helvetica"];']
    states = set(t["from_state"]) | set(t["to_state"])
    for s in states:
        r = node_rate.get(s, 0.5)
        red = int(255 * (1 - r))
        grn = int(255 * r)
        color = f"#{red:02x}{grn:02x}88"
        shape = "doublecircle" if s in ("CLOSED", "LOST") else "box"
        lines.append(f'"{s}" [fillcolor="{color}", shape={shape}];')
    for _, e in t.iterrows():
        pen = 1 + 4 * e["prob"]
        lines.append(f'"{e.from_state}" -> "{e.to_state}" [penwidth={pen:.1f}, label="{e.prob:.0%}"];')
    lines.append("}")
    return "\n".join(lines)
