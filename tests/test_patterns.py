import sys, os
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from analysis.patterns import lost_patterns, archetypes, archetype_summary


def test_lost_pattern_surfaces_frequent_loss_ngram():
    seqs = {}
    for i in range(6):
        seqs[f"l{i}"] = ["greeting", "objection_price", "request_human"]  # all lost
    for i in range(6):
        seqs[f"w{i}"] = ["greeting", "ready_to_buy", "shares_contact"]    # all won
    convs = pd.DataFrame({
        "conversation_id": list(seqs),
        "converted": [False] * 6 + [True] * 6,
    })
    lp = lost_patterns(seqs, convs, top=10)
    pats = [tuple(p) for p in lp["pattern"]]
    assert ("objection_price", "request_human") in pats
    row = lp[lp["pattern"].apply(lambda p: tuple(p) == ("objection_price", "request_human"))].iloc[0]
    assert row["conv_rate"] == 0.0
    assert row["count"] == 6


def test_archetypes_and_summary_shape():
    seqs = {}
    for i in range(10):
        seqs[f"l{i}"] = ["greeting", "objection_price", "request_human"]
    for i in range(10):
        seqs[f"w{i}"] = ["greeting", "ready_to_buy", "shares_contact"]
    convs = pd.DataFrame({
        "conversation_id": list(seqs),
        "converted": [False] * 10 + [True] * 10,
    })
    asg = archetypes(seqs, convs)
    assert set(asg.columns) == {"conversation_id", "cluster"}
    summ = archetype_summary(asg, convs, seqs)
    assert {"cluster", "size", "conv_rate", "top_intents"}.issubset(summ.columns)
    assert summ["size"].sum() == 20
