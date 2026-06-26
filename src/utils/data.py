import json
from pathlib import Path

import pandas as pd

_DATA_PATH = Path(__file__).parent.parent.parent / "data" / "DoroMarine.csv"

_FREQ_MAP = {"Daily": "D", "Weekly": "W-MON", "Monthly": "ME"}
_LABEL_FMT = {"Daily": "%b %d", "Weekly": "Week of %b %d", "Monthly": "%b %Y"}


def is_converted(val: str) -> bool:
    if pd.isna(val) or not isinstance(val, str):
        return False
    try:
        messages = json.loads(val)
        return any(isinstance(m, dict) and m.get("role") == "tool_calls" for m in messages)
    except (json.JSONDecodeError, TypeError):
        return False


def load_raw_data() -> pd.DataFrame:
    df = pd.read_csv(_DATA_PATH)
    df["created"] = pd.to_datetime(df["created"], format="mixed", utc=True).dt.tz_convert(None)
    df["converted"] = df["messages"].apply(is_converted)
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
