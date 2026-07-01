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
    docs = [" ".join(sequences[i]) if sequences[i] else "empty" for i in ids]
    return ids, docs


def archetypes(sequences, convs=None, k_range=range(3, 9)) -> pd.DataFrame:
    ids, docs = _docs(sequences)
    k_range = [k for k in k_range if k < len(ids)]
    if not k_range:
        return pd.DataFrame({"conversation_id": ids, "cluster": [0] * len(ids)})
    vec = TfidfVectorizer(ngram_range=(1, 2), token_pattern=r"[^\s]+")
    X = vec.fit_transform(docs)
    best_score, best_labels = -1.0, None
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
        if len(set(km.labels_)) < 2:
            continue
        try:
            s = silhouette_score(X, km.labels_)
        except ValueError:
            continue
        if s > best_score:
            best_score, best_labels = s, km.labels_
    if best_labels is None:
        best_labels = np.zeros(len(ids), dtype=int)
    return pd.DataFrame({"conversation_id": ids, "cluster": best_labels})


def archetype_summary(assignments: pd.DataFrame, convs: pd.DataFrame,
                      sequences: dict[str, list[str]]) -> pd.DataFrame:
    df = assignments.merge(convs[["conversation_id", "converted"]], on="conversation_id")
    rows = []
    for cl, g in df.groupby("cluster"):
        intents: Counter = Counter()
        for cid in g["conversation_id"]:
            intents.update(sequences.get(cid, []))
        top = ", ".join(lbl for lbl, _ in intents.most_common(4))
        rows.append({
            "cluster": int(cl),
            "size": len(g),
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
