"""
Extracts all user-role messages from DoroMarine.csv into a flat JSON file.

Output: data/user_messages.json
Each entry contains the message content alongside its parent chat metadata.
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
OUTPUT = ROOT / "data" / "user_messages.json"


def extract(row: pd.Series) -> list[dict]:
    val = row["messages"]
    if pd.isna(val) or not isinstance(val, str):
        return []
    try:
        messages = json.loads(val)
    except (json.JSONDecodeError, TypeError):
        return []

    results = []
    pending: list[str] = []

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "user":
            pending.append(msg.get("content", ""))
        else:
            if pending:
                results.append({"content": " ".join(pending)})
                pending = []

    if pending:
        results.append({"content": " ".join(pending)})

    return results


def main():
    df = pd.read_csv(ROOT / "data" / "DoroMarine.csv")

    all_messages = []
    for _, row in df.iterrows():
        all_messages.extend(extract(row))

    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(json.dumps(all_messages, ensure_ascii=False, indent=2))

    print(f"Extracted {len(all_messages)} user messages → {OUTPUT}")


if __name__ == "__main__":
    main()
