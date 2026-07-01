import json

import pandas as pd
import streamlit as st


def render_messages(messages: list[dict], highlight: set[int] | None = None) -> None:
    """Render a parsed message list as Streamlit chat bubbles. Shared by the Chat
    Viewer tab, the Intent Patterns cards, and the transition inspector.

    `highlight` is a set of message indices (== turn_index) to flag — used to mark
    the two turns of a Markov transition inside the full transcript."""
    if not messages:
        st.info("No messages to display.")
        return

    highlight = highlight or set()
    for idx, msg in enumerate(messages):
        if idx in highlight:
            st.markdown(":red-background[**● transition step**]")
        role = msg.get("role", "")
        content = str(msg.get("content", "")).strip()
        timestamp = msg.get("date", "")
        is_human_manager = role == "assistant" and msg.get("type") == "manager"

        if role == "user":
            with st.chat_message("user"):
                if content:
                    st.write(content)
                if timestamp:
                    st.caption(timestamp)

        elif role == "assistant":
            avatar = "👤" if is_human_manager else "🤖"
            label = "Human manager" if is_human_manager else "AI agent"
            with st.chat_message("assistant", avatar=avatar):
                if content:
                    st.write(content)
                st.caption(f"{label} · {timestamp}" if timestamp else label)

        elif role == "info":
            if "PAUSE_DIALOG_OPERATOR_INTERVENTION" in content:
                st.warning("⚡ Human manager took over the conversation", icon="⚡")

        elif role == "error":
            st.error(f"Delivery error: {content}")

        elif role == "tool_calls":
            with st.chat_message("assistant", avatar="🔧"):
                for tc in msg.get("tool_calls", []):
                    fn = tc.get("function", {})
                    name = fn.get("name", "unknown")
                    args = fn.get("arguments", {})
                    st.markdown(f"**Tool:** `{name}`")
                    if args:
                        st.code(json.dumps(args, ensure_ascii=False, indent=2), language="json")
                if timestamp:
                    st.caption(timestamp)

        elif role == "tool":
            with st.expander("Tool result"):
                st.code(str(msg.get("tool_result", "")))

        elif role == "system":
            with st.expander("System instruction"):
                st.caption(content)

    with st.expander("Raw JSON"):
        st.code(json.dumps(messages, ensure_ascii=False, indent=2), language="json")


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
