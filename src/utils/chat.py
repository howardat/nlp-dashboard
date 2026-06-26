import json

import pandas as pd


def parse_messages(val: str) -> list[dict]:
    if pd.isna(val) or not isinstance(val, str):
        return []
    try:
        parsed = json.loads(val)
        return parsed if isinstance(parsed, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def has_operator_intervention(val: str) -> bool:
    """True when any message content equals PAUSE_DIALOG_OPERATOR_INTERVENTION."""
    return any(
        isinstance(m, dict) and m.get("content") == "PAUSE_DIALOG_OPERATOR_INTERVENTION"
        for m in parse_messages(val)
    )


def is_low_quality(val: str) -> bool:
    """True when the chat has exactly one user message shorter than 2 characters."""
    messages = parse_messages(val)
    user_messages = [m for m in messages if isinstance(m, dict) and m.get("role") == "user"]
    return len(user_messages) == 1 and len(str(user_messages[0].get("content", "")).strip()) < 2


def build_chat_labels(df: pd.DataFrame) -> list[str]:
    labels = []
    for _, row in df.iterrows():
        marker = "✓" if row["converted"] else "✗"
        date_str = row["created"].strftime("%Y-%m-%d %H:%M")
        user = str(row.get("user_name", ""))
        if not user or user == "nan":
            user = str(row.get("user_id", "?"))
        user = user[:22]
        platform = str(row.get("from_messenger", "?"))
        labels.append(f"[{marker}] {date_str}  {platform}  {user}")
    return labels
