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
        "conversation_id": ["a", "b", "c", "d"],
        "mean": [0.9, 0.8, -0.8, -0.9],
        "final": [0.9, 0.8, -0.8, -0.9],
        "min": [0.9, 0.8, -0.8, -0.9],
        "max_drop": [0.0, 0.0, 0.0, 0.0],
        "slope": [0.0, 0.0, 0.0, 0.0],
        "neg_run_len": [0, 0, 1, 1],
        "start_to_end": [0.0, 0.0, 0.0, 0.0],
        "n_client_turns": [3, 3, 3, 3],
    })
    convs = pd.DataFrame({"conversation_id": ["a", "b", "c", "d"], "converted": [True, True, False, False]})
    c = correlations(feats, convs)
    row = c[c["feature"] == "final"].iloc[0]
    assert row["r"] > 0  # higher final sentiment -> converted
