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
    'Return ONLY JSON: {"intent": <label>, "intent_confidence": 0-1, '
    '"sentiment": -1..1, "sentiment_confidence": 0-1}.\n'
    "Choose the SINGLE best intent from this list: " + ", ".join(CLIENT_INTENTS) + ".\n"
    "Guidance:\n"
    "- child_condition: describes the child's age, diagnosis, or health (e.g. 'сыну 5 лет аутизм', 'тәбеті жоқ').\n"
    "- order_details: names products/quantities to order (e.g. 'Доромарин 5 банка, сироп 1').\n"
    "- barter_partnership: asks about cooperation, barter, or reselling.\n"
    "- shares_contact: gives a phone number OR their own name.\n"
    "- ready_to_buy: says they want to buy / place an order without specifics.\n"
    "Use 'other' ONLY when nothing above fits at all.\n"
    "sentiment: -1 angry/frustrated, 0 neutral, 1 enthusiastic."
)
AGENT_SYSTEM = (
    "You classify a single AI-SALES-AGENT message (RU/KZ/mixed) to a customer.\n"
    'Return ONLY JSON: {"act": <label>, "confidence": 0-1}.\n'
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
    if not isinstance(d, dict):
        return {"intent": "other", "intent_confidence": 0.0, "sentiment": None, "sentiment_confidence": 0.0}
    if role == "agent":
        return {
            "intent": str(d.get("act", "other")).strip().lower().replace(" ", "_") or "other",
            "intent_confidence": _clamp(d.get("confidence"), 0, 1),
            "sentiment": None,
            "sentiment_confidence": 0.0,
        }
    s = d.get("sentiment")
    return {
        "intent": str(d.get("intent", "other")).strip().lower().replace(" ", "_") or "other",
        "intent_confidence": _clamp(d.get("intent_confidence"), 0, 1),
        "sentiment": _clamp(s, -1, 1, None) if s is not None else None,
        "sentiment_confidence": _clamp(d.get("sentiment_confidence"), 0, 1),
    }


def _label(text: str, role: str) -> dict:
    system = CLIENT_SYSTEM if role == "client" else AGENT_SYSTEM
    resp = get_client().chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": text[:2000]},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return _parse_label(resp.choices[0].message.content or "", role)


def label_client_turn(text: str) -> dict:
    return _label(text, "client")


def label_agent_turn(text: str) -> dict:
    return _label(text, "agent")
