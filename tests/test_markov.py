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
    gp = t[(t.from_state == "greeting") & (t.to_state == "price_inquiry")].iloc[0]
    assert gp["count"] == 1
    op = t[(t.from_state == "objection_price") & (t.to_state == "LOST")].iloc[0]
    assert op["conv_rate"] == 0.0


def test_worst_orders_by_low_conversion():
    seqs = {f"c{i}": ["greeting", "objection_price"] for i in range(5)}
    seqs.update({f"d{i}": ["greeting", "ready_to_buy"] for i in range(5)})
    convs = pd.DataFrame({
        "conversation_id": list(seqs),
        "converted": [False] * 5 + [True] * 5,
        "outcome": ["lost"] * 5 + ["closed"] * 5,
    })
    t = transition_table(seqs, convs, min_support=2)
    w = worst_transitions(t, n=5)
    # terminal destinations are excluded (tautological 0/1 conversion)
    assert not w["to_state"].isin(["CLOSED", "LOST"]).any()
    top = w.iloc[0]
    assert top["to_state"] == "objection_price"
    assert top["conv_rate"] == 0.0
