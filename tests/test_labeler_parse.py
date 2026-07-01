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
