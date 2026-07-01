import json
import sqlite3
from pathlib import Path

import pandas as pd

_DATA_PATH = Path(__file__).parent.parent.parent / "data" / "DoroMarine.csv"
_DB_PATH = Path(__file__).parent.parent.parent / "data" / "signals.db"
DB_PATH = _DB_PATH  # public export for callers that need to open their own connection

_FREQ_MAP = {"Daily": "D", "Weekly": "W-MON", "Monthly": "ME"}
_LABEL_FMT = {"Daily": "%b %d", "Weekly": "Week of %b %d", "Monthly": "%b %Y"}


def _parse_msgs(val: str) -> list[dict]:
    if pd.isna(val) or not isinstance(val, str):
        return []
    try:
        parsed = json.loads(val)
        return parsed if isinstance(parsed, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def _detect_outcome(msgs: list[dict]) -> str:
    has_intervention = any(
        isinstance(m, dict)
        and m.get("role") == "info"
        and "PAUSE_DIALOG_OPERATOR_INTERVENTION" in str(m.get("content", ""))
        for m in msgs
    )
    if has_intervention:
        return "human-closed"
    has_crm = any(
        isinstance(m, dict)
        and m.get("role") == "tool"
        and ("Успешно" in str(m.get("tool_result", "")) or '"result":"success"' in str(m.get("tool_result", "")))
        for m in msgs
    )
    if has_crm:
        return "closed"
    user_count = sum(1 for m in msgs if isinstance(m, dict) and m.get("role") == "user")
    if user_count < 2:
        return "abandoned"
    return "lost"


def _is_ai_conversation(msgs: list[dict]) -> bool:
    return any(
        isinstance(m, dict) and m.get("role") == "assistant" and m.get("type") != "manager"
        for m in msgs
    )


def is_converted(val: str) -> bool:
    msgs = _parse_msgs(val)
    return _detect_outcome(msgs) in ("closed", "human-closed")


def load_raw_data() -> pd.DataFrame:
    df = pd.read_csv(_DATA_PATH)
    df["created"] = pd.to_datetime(df["created"], format="mixed", utc=True).dt.tz_convert(None)

    msgs_series = df["messages"].apply(_parse_msgs)
    df["outcome"] = msgs_series.apply(_detect_outcome)
    df["converted"] = df["outcome"].isin(["closed", "human-closed"])
    df["is_ai"] = msgs_series.apply(_is_ai_conversation)
    df["intervention"] = df["outcome"] == "human-closed"
    df["ai_closed"] = df["outcome"] == "closed"
    return df


def aggregate_by_period(df: pd.DataFrame, granularity: str) -> pd.DataFrame:
    freq = _FREQ_MAP[granularity]
    agg = (
        df.set_index("created")
        .resample(freq)
        .agg(total=("converted", "count"), converted=("converted", "sum"))
        .reset_index()
    )
    agg["conversion_rate"] = agg.apply(
        lambda r: r["converted"] / r["total"] * 100 if r["total"] > 0 else 0,
        axis=1,
    )
    agg["period_label"] = agg["created"].dt.strftime(_LABEL_FMT[granularity])
    return agg


def platform_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby("from_messenger")
        .agg(total=("converted", "count"), converted=("converted", "sum"))
        .reset_index()
    )
    agg["conversion_rate"] = (agg["converted"] / agg["total"] * 100).round(1)
    agg.columns = ["Platform", "Total chats", "Converted", "Conversion rate (%)"]
    return agg


def outcome_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["outcome"].value_counts().reset_index()
    counts.columns = ["Outcome", "Count"]
    total = counts["Count"].sum()
    counts["Share (%)"] = (counts["Count"] / total * 100).round(1)
    return counts


def load_signals_db() -> sqlite3.Connection | None:
    if not _DB_PATH.exists():
        return None
    return sqlite3.connect(_DB_PATH, check_same_thread=False)
