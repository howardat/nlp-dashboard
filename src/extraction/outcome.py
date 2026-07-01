"""Conversation-level outcome detection from raw message lists."""

from __future__ import annotations


def detect_outcome(msgs: list[dict]) -> str:
    """
    Returns 'closed' | 'human-closed' | 'lost' | 'abandoned'.

    Priority: human-closed > closed > abandoned > lost
    """
    has_intervention = False
    has_crm_success = False

    user_msgs = [m for m in msgs if isinstance(m, dict) and m.get("role") == "user"]

    for turn in msgs:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role", "")
        content = str(turn.get("content", ""))

        if role == "info" and "PAUSE_DIALOG_OPERATOR_INTERVENTION" in content:
            has_intervention = True

        if role == "tool":
            result = str(turn.get("tool_result", ""))
            if "Успешно" in result or '"result":"success"' in result or '"result": "success"' in result:
                has_crm_success = True

    if has_intervention:
        return "human-closed"
    if has_crm_success:
        return "closed"
    if len(user_msgs) < 2:
        return "abandoned"
    return "lost"


def is_ai_conversation(msgs: list[dict]) -> bool:
    """True if the AI agent was involved (not a pure human-manager conversation)."""
    return any(
        isinstance(m, dict) and m.get("role") == "assistant" and m.get("type") != "manager"
        for m in msgs
    )


def get_drop_off_turn(msgs: list[dict], outcome: str) -> int | None:
    """Index of the last user turn for non-closed conversations."""
    if outcome in ("closed", "human-closed"):
        return None
    last_user_idx = None
    for i, m in enumerate(msgs):
        if isinstance(m, dict) and m.get("role") == "user":
            last_user_idx = i
    return last_user_idx


def normalize_role(msg: dict) -> str:
    """Map raw roles to canonical turn roles."""
    role = msg.get("role", "")
    if role == "user":
        return "client"
    if role == "assistant":
        return "human" if msg.get("type") == "manager" else "agent"
    if role in ("tool_calls", "tool"):
        return "tool"
    return role  # system, info, error
