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
