import streamlit as st

from utils.chat import build_chat_labels, has_operator_intervention, is_low_quality, parse_messages
from utils.data import aggregate_by_period, load_raw_data, platform_breakdown

st.set_page_config(page_title="DoroMarine Conversion Dashboard", layout="wide")


@st.cache_data
def load_data():
    return load_raw_data()


df = load_data()

# ── Sidebar filters ──────────────────────────────────────────────────────────

st.sidebar.header("Filters")

platforms = ["All"] + sorted(df["from_messenger"].dropna().unique().tolist())
selected_platform = st.sidebar.selectbox("Platform", platforms)

granularity = st.sidebar.radio("Time granularity", ["Daily", "Weekly", "Monthly"])

min_date = df["created"].min().date()
max_date = df["created"].max().date()
date_range = st.sidebar.date_input(
    "Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

# ── Apply filters ────────────────────────────────────────────────────────────

filtered = df.copy()

if selected_platform != "All":
    filtered = filtered[filtered["from_messenger"] == selected_platform]

if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start, end = date_range
    filtered = filtered[
        (filtered["created"].dt.date >= start) & (filtered["created"].dt.date <= end)
    ]

# ── Tabs ─────────────────────────────────────────────────────────────────────

tab_dashboard, tab_chats = st.tabs(["Dashboard", "Chat Viewer"])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════

with tab_dashboard:
    st.title("DoroMarine — Conversion Rate Dashboard")

    total_chats = len(filtered)
    total_converted = int(filtered["converted"].sum())
    overall_rate = total_converted / total_chats * 100 if total_chats else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Total chats", f"{total_chats:,}")
    c2.metric("Converted chats", f"{total_converted:,}")
    c3.metric("Overall conversion rate", f"{overall_rate:.1f}%")

    st.divider()

    agg = aggregate_by_period(filtered, granularity)

    st.subheader(f"Conversion Rate — {granularity}")
    if agg.empty or agg["total"].sum() == 0:
        st.warning("No data for the selected filters.")
    else:
        st.bar_chart(
            agg.set_index("period_label")[["conversion_rate"]],
            y_label="Conversion rate (%)",
            use_container_width=True,
        )

    st.subheader(f"Chat Volume — {granularity}")
    if not agg.empty:
        volume = agg.set_index("period_label")[["total", "converted"]].rename(
            columns={"total": "Total chats", "converted": "Converted"}
        )
        st.bar_chart(volume, use_container_width=True)

    with st.expander("Period breakdown table"):
        display = agg[["period_label", "total", "converted", "conversion_rate"]].copy()
        display.columns = ["Period", "Total chats", "Converted", "Conversion rate (%)"]
        display["Conversion rate (%)"] = display["Conversion rate (%)"].round(1)
        st.dataframe(display, use_container_width=True, hide_index=True)

    st.subheader("Conversion Rate by Platform")
    st.dataframe(platform_breakdown(df), use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — CHAT VIEWER
# ═══════════════════════════════════════════════════════════════════════════

with tab_chats:
    st.title("Chat Viewer")

    filter_col, toggle_col = st.columns([3, 2])
    with filter_col:
        conv_filter = st.radio(
            "Show", ["All", "Converted only", "Not converted"], horizontal=True
        )
    with toggle_col:
        hide_low_quality = st.toggle(
            "Hide single-symbol replies",
            value=True,
            help="Remove chats where the user sent only one message shorter than 2 characters",
        )

    view_df = filtered.copy()
    if conv_filter == "Converted only":
        view_df = view_df[view_df["converted"]]
    elif conv_filter == "Not converted":
        view_df = view_df[~view_df["converted"]]

    if hide_low_quality:
        view_df = view_df[~view_df["messages"].apply(is_low_quality)]

    if st.toggle("Operator intervention only", help='Show only chats containing "PAUSE_DIALOG_OPERATOR_INTERVENTION"'):
        view_df = view_df[view_df["messages"].apply(has_operator_intervention)]

    view_df = view_df.sort_values("created", ascending=False).reset_index(drop=True)

    if view_df.empty:
        st.warning("No chats match the current filters.")
        st.stop()

    labels = build_chat_labels(view_df)
    selected_idx = st.selectbox(
        f"Select chat ({len(view_df):,} chats)",
        range(len(labels)),
        format_func=lambda i: labels[i],
    )

    row = view_df.iloc[selected_idx]

    st.divider()

    # ── Chat metadata ────────────────────────────────────────────────────────

    user = str(row.get("user_name", ""))
    if not user or user == "nan":
        user = str(row.get("user_id", "?"))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("User", user[:30])
    c2.metric("Platform", str(row.get("from_messenger", "?")))
    c3.metric("Date", row["created"].strftime("%Y-%m-%d %H:%M"))
    c4.metric("Converted", "Yes ✓" if row["converted"] else "No ✗")

    st.divider()

    # ── Messages ─────────────────────────────────────────────────────────────

    messages = parse_messages(row["messages"])

    if not messages:
        st.info("No messages to display.")
    else:
        for msg in messages:
            role = msg.get("role", "")
            content = str(msg.get("content", "")).strip()
            timestamp = msg.get("date", "")

            if role == "user":
                with st.chat_message("user"):
                    if content:
                        st.write(content)
                    if timestamp:
                        st.caption(timestamp)

            elif role == "assistant":
                with st.chat_message("assistant"):
                    if content:
                        st.write(content)
                    if timestamp:
                        st.caption(timestamp)

            elif role == "tool_calls":
                with st.chat_message("assistant", avatar="🔧"):
                    tool_calls = msg.get("tool_calls", [])
                    if tool_calls:
                        for tc in tool_calls:
                            fn = tc.get("function", {})
                            name = fn.get("name", "unknown")
                            args = fn.get("arguments", "")
                            st.markdown(f"**Tool call:** `{name}`")
                            if args:
                                st.code(args, language="json")
                    else:
                        st.caption("Phone number captured — conversion recorded")
                    if timestamp:
                        st.caption(timestamp)
