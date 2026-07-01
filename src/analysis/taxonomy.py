"""Intent / agent-act taxonomy and state mapping for conversation analysis."""
from __future__ import annotations

CLIENT_INTENTS = [
    "greeting", "price_inquiry", "product_info", "dosage_safety",
    "child_condition", "order_details", "barter_partnership",
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
    """Normalize a raw model label. Known labels pass through; unknown non-empty labels
    are kept (candidates for the discovery merge); empty maps to 'other'."""
    l = (label or "").strip().lower().replace(" ", "_")
    if not l:
        return "other"
    allowed = _CLIENT_SET if role == "client" else _AGENT_SET
    return l if l in allowed else l


def terminal_state(outcome: str) -> str:
    """Absorbing Markov state for a conversation outcome."""
    return "CLOSED" if outcome in ("closed", "human-closed") else "LOST"
