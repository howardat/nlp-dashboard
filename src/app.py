import os
import sys

import pandas as pd
import streamlit as st

# Allow imports from src/
sys.path.insert(0, os.path.dirname(__file__))

from utils.chat import (
    build_chat_labels,
    has_operator_intervention,
    is_low_quality,
    parse_messages,
    render_messages,
)
from utils.data import (
    DB_PATH,
    aggregate_by_period,
    load_raw_data,
    load_signals_db,
    outcome_breakdown,
    platform_breakdown,
)

st.set_page_config(page_title="DoroMarine — Sales Agent Analytics", layout="wide")


@st.cache_data
def load_data() -> pd.DataFrame:
    return load_raw_data()


@st.cache_data
def load_signals_frames():
    """Load per-turn signals + conversation outcomes for the analysis tabs.

    Only fully-labeled conversations are returned (labeled turns == expected
    labelable turns), so a partial/interrupted labeling run can't corrupt the
    sequence analyses. Returns (signals_df, convs_df) or (None, None)."""
    conn = load_signals_db()
    if conn is None:
        return None, None
    sig = pd.read_sql(
        "SELECT conversation_id, turn_index, role, intent, intent_confidence, "
        "sentiment, sentiment_confidence FROM turn_signals",
        conn,
    )
    convs = pd.read_sql(
        "SELECT id as conversation_id, outcome FROM conversations WHERE is_ai_conversation=1",
        conn,
    )
    expected = pd.read_sql(
        "SELECT conversation_id, COUNT(*) AS n_expected FROM turns "
        "WHERE role IN ('client','agent') AND TRIM(COALESCE(raw_text,'')) != '' "
        "GROUP BY conversation_id",
        conn,
    )
    conn.close()
    if sig.empty:
        return None, None

    # keep only conversations whose every labelable turn is present
    labeled = sig.groupby("conversation_id").size().rename("n_labeled").reset_index()
    complete = labeled.merge(expected, on="conversation_id")
    full_ids = set(complete.loc[complete["n_labeled"] >= complete["n_expected"], "conversation_id"])
    sig = sig[sig["conversation_id"].isin(full_ids)].reset_index(drop=True)
    if sig.empty:
        return None, None

    convs["converted"] = convs["outcome"].isin(["closed", "human-closed"])
    convs = convs[convs["conversation_id"].isin(sig["conversation_id"].unique())].reset_index(drop=True)
    return sig, convs


_SIGNALS_EMPTY_MSG = (
    "Run the turn-labeling pass first to populate this tab:\n\n"
    "```bash\nexport OPENROUTER_API_KEY=...\ncd src && python -m extraction.signals\n```"
)


@st.cache_data
def conv_messages_map() -> dict:
    """conversation_id -> raw messages JSON string, matching the pipeline's id rule."""
    d = load_data()

    def _cid(r):
        return str(r.get("chat_id") or r.get("user_id") or r.name)

    return {_cid(r): r["messages"] for _, r in d.iterrows()}


df = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────

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

show_ai_only = st.sidebar.toggle(
    "AI conversations only",
    value=True,
    help="Exclude conversations where a human manager started and handled the chat with no AI involvement",
)

# ── Apply filters ─────────────────────────────────────────────────────────────

filtered = df.copy()

if selected_platform != "All":
    filtered = filtered[filtered["from_messenger"] == selected_platform]

if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start, end = date_range
    filtered = filtered[
        (filtered["created"].dt.date >= start) & (filtered["created"].dt.date <= end)
    ]

if show_ai_only:
    filtered = filtered[filtered["is_ai"]]

# ── Tabs ──────────────────────────────────────────────────────────────────────

(
    tab_dashboard, tab_failures, tab_clusters, tab_chats, tab_findings,
    tab_sentiment, tab_transitions, tab_patterns,
) = st.tabs([
    "Overview", "Failure Patterns", "FAQ Clusters", "Chat Viewer", "Findings",
    "Sentiment Trajectory", "Transitions", "Intent Patterns",
])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════

with tab_dashboard:
    st.title("DoroMarine — Conversion Dashboard")

    total = len(filtered)
    ai_closed = int(filtered["ai_closed"].sum())
    human_closed = int(filtered["intervention"].sum())
    lost = int((filtered["outcome"] == "lost").sum())
    abandoned = int((filtered["outcome"] == "abandoned").sum())
    converted = ai_closed + human_closed

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total chats", f"{total:,}")
    c2.metric("AI closed", f"{ai_closed:,}", help="Lead captured by AI agent via CRM tool call")
    c3.metric("Human closed", f"{human_closed:,}", help="Human manager intervened to close")
    c4.metric("Lost", f"{lost:,}")
    c5.metric("Overall conversion", f"{converted / total * 100:.1f}%" if total else "—")

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Outcome breakdown")
        ob = outcome_breakdown(filtered)
        st.bar_chart(ob.set_index("Outcome")["Count"])

    with col_right:
        st.subheader("Intervention rate")
        st.metric(
            "Conversations requiring human takeover",
            f"{human_closed / total * 100:.1f}%" if total else "—",
            help="How often the AI agent couldn't close without a human stepping in",
        )
        if total:
            ai_conv = filtered[filtered["is_ai"]]
            ai_total = len(ai_conv)
            if ai_total:
                st.metric(
                    "AI close rate (no human needed)",
                    f"{ai_closed / ai_total * 100:.1f}%",
                )

    st.divider()

    agg = aggregate_by_period(filtered, granularity)
    st.subheader(f"Conversion Rate — {granularity}")
    if agg.empty or agg["total"].sum() == 0:
        st.info("No data for the selected filters.")
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

    st.subheader("By Platform")
    st.dataframe(platform_breakdown(filtered), use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — FAILURE PATTERNS
# ═══════════════════════════════════════════════════════════════════════════

with tab_failures:
    st.title("Failure Patterns")

    conn = load_signals_db()

    if conn is None:
        st.info(
            "Run the extraction pipeline first to see failure patterns.\n\n"
            "```bash\ncd src && python -m extraction.pipeline --sample 100 --no-label\n```"
        )
    else:
        ai_convs = filtered[filtered["is_ai"]]
        lost_convs = ai_convs[ai_convs["outcome"].isin(["lost", "abandoned"])]
        total_ai = len(ai_convs)
        total_lost = len(lost_convs)

        st.subheader("Where do conversations fail?")

        c1, c2, c3 = st.columns(3)
        c1.metric("AI conversations analyzed", f"{total_ai:,}")
        c2.metric("Lost / abandoned", f"{total_lost:,}")
        c3.metric("Loss rate", f"{total_lost / total_ai * 100:.1f}%" if total_ai else "—")

        st.divider()

        # Drop-off turn distribution from SQLite
        try:
            drop_off_df = pd.read_sql(
                """
                SELECT drop_off_turn, COUNT(*) as count
                FROM conversations
                WHERE is_ai_conversation = 1
                  AND outcome IN ('lost', 'abandoned')
                  AND drop_off_turn IS NOT NULL
                GROUP BY drop_off_turn
                ORDER BY drop_off_turn
                """,
                conn,
            )

            if not drop_off_df.empty:
                st.subheader("Drop-off turn distribution")
                st.caption("Which turn index clients stop responding (turn 0 = first message)")
                st.bar_chart(
                    drop_off_df.set_index("drop_off_turn")["count"],
                    x_label="Turn index",
                    y_label="Conversations dropped",
                    use_container_width=True,
                )
            else:
                st.info("No drop-off data yet — run the pipeline to populate.")
        except Exception:
            st.info("Drop-off data not yet available.")

        # Failure pattern clusters from topic modeling
        try:
            fail_df = pd.read_sql(
                """
                SELECT label, turn_count
                FROM clusters
                WHERE cluster_type = 'failure'
                ORDER BY turn_count DESC
                LIMIT 10
                """,
                conn,
            )

            if not fail_df.empty:
                st.subheader("Top failure patterns (from topic modeling)")
                st.caption("Discovered from client messages in lost conversations — no predefined categories")
                total_failure_turns = fail_df["turn_count"].sum()
                fail_df["share_%"] = (fail_df["turn_count"] / total_failure_turns * 100).round(1)
                st.dataframe(
                    fail_df.rename(columns={"label": "Pattern", "turn_count": "Occurrences", "share_%": "Share (%)"}),
                    use_container_width=True,
                    hide_index=True,
                )
                st.bar_chart(
                    fail_df.set_index("label")["turn_count"],
                    x_label="Pattern",
                    y_label="Client messages",
                    use_container_width=True,
                    horizontal=True,
                )
            else:
                st.info(
                    "Run failure clustering to discover patterns:\n"
                    "```bash\ncd src && python -m extraction.clustering --mode failure\n```"
                )
        except Exception:
            st.info(
                "Run failure clustering:\n"
                "```bash\ncd src && python -m extraction.clustering --mode failure\n```"
            )

        conn.close()

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — FAQ CLUSTERS
# ═══════════════════════════════════════════════════════════════════════════

with tab_clusters:
    import plotly.express as px

    st.title("FAQ & Topic Clusters")
    st.caption("What clients ask about most — discovered by embedding + clustering, no predefined categories")

    cluster_mode = st.radio("View", ["FAQ clusters", "Failure clusters"], horizontal=True)
    db_cluster_type = "faq" if cluster_mode == "FAQ clusters" else "failure"

    conn = load_signals_db()

    if conn is None:
        st.info("Run the pipeline and clustering first.")
    else:
        clusters_df = pd.read_sql(
            "SELECT id, label, turn_count FROM clusters WHERE cluster_type = ? ORDER BY turn_count DESC",
            conn, params=(db_cluster_type,),
        )

        if clusters_df.empty:
            st.info(f"No {db_cluster_type} clusters yet. Run: `python -m extraction.clustering --mode {db_cluster_type}`")
        else:
            # ── 2D scatter ────────────────────────────────────────────────
            coords_df = pd.read_sql(
                """SELECT tc.turn_id, tc.x, tc.y, tc.cluster_id, t.raw_text,
                          COALESCE(cl.label, 'Noise') as label
                   FROM turn_coords tc
                   JOIN turns t ON tc.turn_id = t.id
                   LEFT JOIN clusters cl ON tc.cluster_id = cl.id
                   WHERE tc.cluster_type = ?""",
                conn, params=(db_cluster_type,),
            )

            if not coords_df.empty:
                st.subheader("Cluster map")
                coords_df["text_short"] = coords_df["raw_text"].str[:80]
                fig = px.scatter(
                    coords_df,
                    x="x", y="y",
                    color="label",
                    hover_data={"text_short": True, "x": False, "y": False, "label": False},
                    labels={"text_short": "Message", "label": "Cluster"},
                    height=500,
                )
                fig.update_traces(marker=dict(size=5, opacity=0.7))
                fig.update_layout(
                    legend=dict(title="Cluster", font=dict(size=10)),
                    margin=dict(l=0, r=0, t=20, b=0),
                    showlegend=True,
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Re-run clustering to generate 2D coordinates for the scatter plot.")

            st.divider()

            # ── Bar chart + drill-down ────────────────────────────────────
            st.subheader(f"Clusters by size — {cluster_mode}")
            top_clusters = clusters_df.head(20)
            st.bar_chart(
                top_clusters.set_index("label")["turn_count"],
                horizontal=True,
                use_container_width=True,
            )

            st.divider()

            # ── Drill-down ────────────────────────────────────────────────
            st.subheader("Inspect a cluster")
            cluster_options = clusters_df["label"].tolist()
            selected_label = st.selectbox(
                "Select cluster",
                cluster_options,
                format_func=lambda l: f"{l}  ({clusters_df[clusters_df['label']==l]['turn_count'].values[0]} messages)",
            )

            selected_id = clusters_df[clusters_df["label"] == selected_label]["id"].values[0]

            messages_df = pd.read_sql(
                """SELECT t.raw_text, c.outcome, c.platform
                   FROM cluster_members cm
                   JOIN turns t ON cm.turn_id = t.id
                   JOIN conversations c ON t.conversation_id = c.id
                   WHERE cm.cluster_id = ?
                   ORDER BY c.outcome, t.raw_text""",
                conn, params=(selected_id,),
            )

            if not messages_df.empty:
                outcome_counts = messages_df["outcome"].value_counts()
                cols = st.columns(len(outcome_counts))
                for col, (outcome, count) in zip(cols, outcome_counts.items()):
                    col.metric(outcome, count)

                st.dataframe(
                    messages_df.rename(columns={
                        "raw_text": "Message",
                        "outcome": "Conversation outcome",
                        "platform": "Platform",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )

        conn.close()

# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — CHAT VIEWER
# ═══════════════════════════════════════════════════════════════════════════

import sqlite3 as _sqlite3
import uuid as _uuid
from datetime import datetime as _dt, timezone as _tz


def _get_tags() -> list[str]:
    if not DB_PATH.exists():
        return []
    with _sqlite3.connect(DB_PATH) as c:
        return [r[0] for r in c.execute("SELECT name FROM tags ORDER BY name").fetchall()]


def _add_tag(name: str) -> None:
    name = name.strip()
    if not name:
        return
    with _sqlite3.connect(DB_PATH) as c:
        c.execute(
            "INSERT OR IGNORE INTO tags (id, name, created_at) VALUES (?,?,?)",
            (str(_uuid.uuid4()), name, _dt.now(_tz.utc).isoformat()),
        )


def _remove_tag(name: str) -> None:
    with _sqlite3.connect(DB_PATH) as c:
        c.execute("DELETE FROM tags WHERE name = ?", (name,))


def _merge_tags(sources: list[str], target: str) -> int:
    """Replace every occurrence of any source tag with target across all annotations."""
    import json as _json
    if not sources or not target:
        return 0
    with _sqlite3.connect(DB_PATH) as c:
        rows = c.execute("SELECT rowid, failure_type FROM annotations").fetchall()
        updated = 0
        for rowid, ft in rows:
            try:
                tags = _json.loads(ft) if ft else []
                if not isinstance(tags, list):
                    tags = [tags] if tags else []
            except (ValueError, TypeError):
                tags = [ft] if ft else []
            new_tags = [target if t in sources else t for t in tags]
            new_tags = list(dict.fromkeys(new_tags))  # deduplicate, preserve order
            if new_tags != tags:
                c.execute("UPDATE annotations SET failure_type=? WHERE rowid=?",
                          (_json.dumps(new_tags, ensure_ascii=False), rowid))
                updated += 1
        # Remove source tags from the tag library (keep target)
        for s in sources:
            if s != target:
                c.execute("DELETE FROM tags WHERE name=?", (s,))
        return updated


def _save_annotation(conv_id: str, tags: list[str], note: str) -> None:
    import json as _json
    with _sqlite3.connect(DB_PATH) as c:
        c.execute(
            "INSERT OR REPLACE INTO annotations (id, conversation_id, failure_type, note, annotated_at) VALUES (?,?,?,?,?)",
            (str(_uuid.uuid4()), conv_id, _json.dumps(tags, ensure_ascii=False), note, _dt.now(_tz.utc).isoformat()),
        )


def _delete_annotation(conv_id: str) -> None:
    with _sqlite3.connect(DB_PATH) as c:
        c.execute("DELETE FROM annotations WHERE conversation_id = ?", (conv_id,))


def _get_annotation(conv_id: str):
    if not DB_PATH.exists():
        return None
    import json as _json
    with _sqlite3.connect(DB_PATH) as c:
        row = c.execute(
            "SELECT failure_type, note FROM annotations WHERE conversation_id = ? ORDER BY annotated_at DESC LIMIT 1",
            (conv_id,),
        ).fetchone()
    if row is None:
        return None
    raw = row[0]
    try:
        parsed = _json.loads(raw) if raw else []
        tags = parsed if isinstance(parsed, list) else ([parsed] if parsed else [])
    except (ValueError, TypeError):
        tags = [raw] if raw else []
    return (tags, row[1])


with tab_chats:
    st.title("Chat Viewer")

    filter_col, toggle_col = st.columns([3, 2])
    with filter_col:
        conv_filter = st.radio(
            "Show", ["All", "Converted only", "Not converted", "Not converted — unannotated"], horizontal=True
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
    elif conv_filter in ("Not converted", "Not converted — unannotated"):
        view_df = view_df[~view_df["converted"]]

    if hide_low_quality:
        view_df = view_df[~view_df["messages"].apply(is_low_quality)]

    if st.toggle("Operator intervention only", help='Show only chats containing human takeover'):
        view_df = view_df[view_df["messages"].apply(has_operator_intervention)]

    view_df = view_df.sort_values("created", ascending=False).reset_index(drop=True)

    if view_df.empty:
        st.warning("No chats match the current filters.")
        st.stop()

    # Compute stable conv_id for each row (mirrors pipeline.py)
    def _row_conv_id(r):
        return str(r.get("chat_id") or r.get("user_id") or r.name)

    view_df = view_df.copy()
    view_df["_conv_id"] = view_df.apply(_row_conv_id, axis=1)

    # Filter to unannotated if requested
    if conv_filter == "Not converted — unannotated" and DB_PATH.exists():
        with _sqlite3.connect(DB_PATH) as _ac:
            annotated_ids = {r[0] for r in _ac.execute("SELECT DISTINCT conversation_id FROM annotations").fetchall()}
        view_df = view_df[~view_df["_conv_id"].isin(annotated_ids)].reset_index(drop=True)
        if view_df.empty:
            st.success("All filtered conversations are annotated.")
            st.stop()

    labels = build_chat_labels(view_df)
    n_chats = len(view_df)

    # Clamp BEFORE the widget renders (writing before instantiation is allowed)
    if "chat_selectbox" not in st.session_state:
        st.session_state["chat_selectbox"] = 0
    elif st.session_state["chat_selectbox"] >= n_chats:
        st.session_state["chat_selectbox"] = max(0, n_chats - 1)

    # on_click callbacks run before widgets on the next rerun — safe to write chat_selectbox there
    def _nav_prev():
        st.session_state["chat_selectbox"] = max(0, st.session_state["chat_selectbox"] - 1)

    def _nav_next():
        st.session_state["chat_selectbox"] = min(n_chats - 1, st.session_state["chat_selectbox"] + 1)

    selected_idx = st.selectbox(
        f"Select chat ({n_chats:,} chats)",
        range(n_chats),
        format_func=lambda i: labels[i],
        key="chat_selectbox",
    )

    row = view_df.iloc[selected_idx]
    conv_id = row["_conv_id"]

    st.divider()

    user = str(row.get("user_name", ""))
    if not user or user == "nan":
        user = str(row.get("user_id", "?"))

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("User", user[:30])
    c2.metric("Platform", str(row.get("from_messenger", "?")))
    c3.metric("Date", row["created"].strftime("%Y-%m-%d %H:%M"))
    c4.metric("Outcome", row.get("outcome", "?").upper())
    c5.metric("AI involved", "Yes" if row.get("is_ai") else "No (human only)")

    st.divider()

    # ── Annotation panel ──────────────────────────────────────────────────
    if DB_PATH.exists():
        existing = _get_annotation(conv_id)
        tags = _get_tags()

        # If existing annotation uses tags that were deleted, keep them visible
        existing_tag_list = existing[0] if existing else []
        for _t in existing_tag_list:
            if _t not in tags:
                tags.insert(0, _t)

        with st.expander(
            ("✅ Annotated — click to edit" if existing else "Tag this conversation"),
            expanded=not existing,
        ):
            if tags:
                st.multiselect(
                    "Tags",
                    tags,
                    default=existing_tag_list,
                    key=f"ann_type_{conv_id}",
                )
            else:
                st.info("No tags yet — create one below.")

            st.text_area(
                "Note (optional — quote the specific moment, agent mistake, etc.)",
                value=existing[1] if existing else "",
                height=80,
                key=f"ann_note_{conv_id}",
            )

            def _do_save(cid):
                _save_annotation(
                    cid,
                    st.session_state.get(f"ann_type_{cid}", []),
                    st.session_state.get(f"ann_note_{cid}", ""),
                )

            def _do_save_next(cid):
                _do_save(cid)
                _nav_next()

            if tags:
                save_col, save_next_col, clear_col = st.columns([2, 2, 1])
                with save_col:
                    st.button("Save", type="primary", key=f"ann_save_{conv_id}",
                              on_click=_do_save, args=(conv_id,))
                with save_next_col:
                    st.button("Save & Next →", type="primary",
                              disabled=selected_idx >= n_chats - 1,
                              key=f"ann_save_next_{conv_id}",
                              help="Save tag and jump to the next conversation",
                              on_click=_do_save_next, args=(conv_id,))
                with clear_col:
                    if existing:
                        st.button("Remove", key=f"ann_del_{conv_id}",
                                  on_click=_delete_annotation, args=(conv_id,))

            # ── Tag management ────────────────────────────────────────────
            st.divider()
            st.caption("Manage tags")
            t_input, t_add = st.columns([5, 1])
            with t_input:
                st.text_input("New tag", key="new_tag_input", placeholder="Type a name and press Add")
            with t_add:
                st.write("")  # vertical align
                def _add_tag_click():
                    _add_tag(st.session_state.get("new_tag_input", ""))
                    st.session_state["new_tag_input"] = ""
                st.button("Add", key="add_tag_btn", on_click=_add_tag_click, use_container_width=True)

            if tags:
                del_tag = st.selectbox(
                    "Delete tag",
                    ["— pick to delete —"] + tags,
                    key="del_tag_select",
                )
                if del_tag != "— pick to delete —":
                    def _del_tag_click(name):
                        _remove_tag(name)
                        st.session_state["del_tag_select"] = "— pick to delete —"
                    st.button(
                        f'Delete "{del_tag}"',
                        key="del_tag_btn",
                        on_click=_del_tag_click,
                        args=(del_tag,),
                    )

                st.divider()
                st.caption("Merge tags")
                merge_sources = st.multiselect(
                    "Tags to merge (will be replaced)",
                    tags,
                    key="merge_sources",
                )
                merge_target = st.selectbox(
                    "Merge into",
                    ["— pick target —"] + tags,
                    key="merge_target",
                )
                if merge_sources and merge_target != "— pick target —":
                    def _do_merge():
                        n = _merge_tags(st.session_state.get("merge_sources", []),
                                        st.session_state.get("merge_target", ""))
                        st.session_state["_merge_result"] = n
                        st.session_state["merge_sources"] = []
                        st.session_state["merge_target"] = "— pick target —"
                    st.button(
                        f"Merge {len(merge_sources)} tag(s) → \"{merge_target}\"",
                        key="merge_btn",
                        type="primary",
                        on_click=_do_merge,
                    )
                if st.session_state.get("_merge_result") is not None:
                    st.success(f"Merged — {st.session_state.pop('_merge_result')} annotation(s) updated.")

    # ── Navigation (below annotation so you can tag-then-advance) ─────────
    nav_prev, nav_next, nav_info = st.columns([1, 1, 5])
    with nav_prev:
        st.button("← Prev", disabled=selected_idx == 0, use_container_width=True,
                  key="nav_prev", on_click=_nav_prev)
    with nav_next:
        st.button("Next →", disabled=selected_idx >= n_chats - 1, use_container_width=True,
                  key="nav_next", on_click=_nav_next)
    with nav_info:
        _quick = _get_annotation(conv_id)
        ann_label = f" · ✅ {', '.join(_quick[0])}" if (_quick and _quick[0]) else " · untagged"
        st.caption(f"**{selected_idx + 1} / {n_chats}**{ann_label}")

    st.divider()

    render_messages(parse_messages(row["messages"]))

# ═══════════════════════════════════════════════════════════════════════════
# TAB 5 — FINDINGS
# ═══════════════════════════════════════════════════════════════════════════

with tab_findings:
    st.title("Findings")
    st.caption("Your manual annotations — what the AI agent actually fails at")

    if not DB_PATH.exists():
        st.info("Run the pipeline first to enable annotations.")
    else:
        with _sqlite3.connect(DB_PATH) as _fc:
            total_ann = _fc.execute("SELECT COUNT(*) FROM annotations").fetchone()[0]

            if total_ann == 0:
                st.info(
                    "No annotations yet. Go to **Chat Viewer**, filter by 'Not converted', "
                    "and tag each lost conversation with a failure type."
                )
            else:
                total_lost_ai = _fc.execute(
                    "SELECT COUNT(*) FROM conversations WHERE outcome IN ('lost','abandoned') AND is_ai_conversation=1"
                ).fetchone()[0]
                annotated_convs = _fc.execute(
                    "SELECT COUNT(DISTINCT conversation_id) FROM annotations"
                ).fetchone()[0]

                c1, c2, c3 = st.columns(3)
                c1.metric("Conversations annotated", f"{annotated_convs:,}")
                c2.metric("Lost AI conversations", f"{total_lost_ai:,}")
                c3.metric("Coverage", f"{annotated_convs / total_lost_ai * 100:.0f}%" if total_lost_ai else "—")

                st.divider()

                breakdown = pd.read_sql(
                    """SELECT j.value as failure_type, COUNT(*) as count
                       FROM annotations, json_each(
                           CASE WHEN json_valid(failure_type) THEN failure_type
                                ELSE json_array(failure_type)
                           END
                       ) j
                       WHERE failure_type IS NOT NULL AND failure_type != ''
                       GROUP BY j.value
                       ORDER BY count DESC""",
                    _fc,
                )
                breakdown["share_%"] = (breakdown["count"] / breakdown["count"].sum() * 100).round(1)

                st.subheader("Tag distribution")
                st.bar_chart(breakdown.set_index("failure_type")["count"], horizontal=True, use_container_width=True)
                st.dataframe(
                    breakdown.rename(columns={"failure_type": "Tag", "count": "Count", "share_%": "Share (%)"}),
                    use_container_width=True,
                    hide_index=True,
                )

                st.divider()

                st.subheader("Inspect examples")
                selected_type = st.selectbox(
                    "Tag",
                    breakdown["failure_type"].tolist(),
                    format_func=lambda t: f"{t}  ({breakdown[breakdown['failure_type']==t]['count'].values[0]})",
                )
                examples = pd.read_sql(
                    """SELECT a.conversation_id, a.note, a.annotated_at, c.outcome, c.platform, c.created_at
                       FROM annotations a
                       JOIN conversations c ON a.conversation_id = c.id
                       WHERE EXISTS (
                           SELECT 1 FROM json_each(
                               CASE WHEN json_valid(a.failure_type) THEN a.failure_type
                                    ELSE json_array(a.failure_type)
                               END
                           ) WHERE value = ?
                       )
                       ORDER BY a.annotated_at DESC""",
                    _fc,
                    params=(selected_type,),
                )
                for _, ex in examples.iterrows():
                    with st.expander(f"{ex['outcome'].upper()} · {ex['platform']} · {ex['created_at'][:10]}"):
                        if ex["note"]:
                            st.write(ex["note"])
                        else:
                            st.caption("No note written.")
                        st.caption(f"Conv ID: {ex['conversation_id']}")

# ═══════════════════════════════════════════════════════════════════════════
# TAB 6 — SENTIMENT TRAJECTORY
# ═══════════════════════════════════════════════════════════════════════════

with tab_sentiment:
    import plotly.express as px

    from analysis.sentiment import avg_trajectory, correlations, logistic_auc, trajectory_features

    st.title("Sentiment Trajectory")
    st.caption("How client sentiment moves through a conversation, and which trajectory "
               "features separate converted from lost chats. Labels are LLM-generated ('silver').")

    sig, convs = load_signals_frames()
    if sig is None:
        st.info(_SIGNALS_EMPTY_MSG)
    else:
        st.warning(
            f"⚠️ Limited sample: {len(convs)} fully-labeled conversations "
            "(labeling was interrupted by an API credit limit). Re-run "
            "`python -m extraction.signals` when credits allow to cover the full corpus.",
            icon="⚠️",
        )
        feats = trajectory_features(sig)
        if feats.empty:
            st.info("No client-turn sentiment found. Re-run the labeling pass.")
        else:
            corr = correlations(feats, convs)
            auc = logistic_auc(feats, convs)

            merged = feats.merge(convs[["conversation_id", "converted"]], on="conversation_id")
            n_conv = int(merged["converted"].sum())
            n_lost = int((~merged["converted"]).sum())

            c1, c2, c3 = st.columns(3)
            c1.metric("Conversations analyzed", f"{len(merged):,}")
            c2.metric("Converted / lost", f"{n_conv} / {n_lost}")
            c3.metric("Logistic AUC", f"{auc:.2f}" if auc == auc else "—",
                      help="5-fold CV ROC-AUC predicting conversion from trajectory features. 0.5 = no signal.")

            st.divider()
            st.subheader("Average sentiment over conversation progress")
            st.caption("Client turns binned into 10 progress steps (0% = first message, 100% = last).")
            at = avg_trajectory(sig, convs, bins=10)
            if not at.empty:
                at = at.copy()
                at["Outcome"] = at["converted"].map({True: "Converted", False: "Lost"})
                at["Progress %"] = (at["bin"] * 10).astype(int)
                fig = px.line(at, x="Progress %", y="mean", color="Outcome",
                              markers=True, labels={"mean": "Mean sentiment"},
                              color_discrete_map={"Converted": "#2ca02c", "Lost": "#d62728"})
                fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0), yaxis_range=[-1, 1])
                fig.add_hline(y=0, line_dash="dot", line_color="gray")
                st.plotly_chart(fig, use_container_width=True)

            st.divider()
            st.subheader("Which trajectory features correlate with conversion?")
            st.caption("Point-biserial correlation with `converted`; p_adj = Benjamini–Hochberg corrected. "
                       "★ marks features that stay significant after correction (p_adj < 0.05).")
            show = corr.copy()
            show["significant"] = show["p_adj"].apply(lambda p: "★" if p < 0.05 else "")
            show["r"] = show["r"].round(3)
            show["p_adj"] = show["p_adj"].round(4)
            st.dataframe(
                show[["feature", "r", "p_adj", "direction", "significant"]].rename(columns={
                    "feature": "Feature", "r": "Correlation (r)", "p_adj": "p (adj)",
                    "direction": "Direction", "significant": "Sig.",
                }),
                use_container_width=True, hide_index=True,
            )
            top_sig = show[show["significant"] == "★"]
            if not top_sig.empty:
                lead = top_sig.iloc[0]
                st.info(f"Strongest signal: **{lead['feature']}** (r={lead['r']}), {lead['direction']}.")

# ═══════════════════════════════════════════════════════════════════════════
# TAB 7 — TRANSITIONS (Markov)
# ═══════════════════════════════════════════════════════════════════════════

with tab_transitions:
    from analysis.markov import build_sequences, to_dot, transition_table, worst_transitions

    st.title("Conversation Transitions")
    st.caption("A Markov model over client intents + key agent acts, ending in CLOSED or LOST. "
               "Nodes are shaded red→green by the conversion rate of chats leaving them.")

    sig, convs = load_signals_frames()
    if sig is None:
        st.info(_SIGNALS_EMPTY_MSG)
    else:
        st.warning(
            f"⚠️ Limited sample: {len(convs)} fully-labeled conversations "
            "(labeling interrupted by an API credit limit).",
            icon="⚠️",
        )
        min_support = st.slider("Minimum transition support (occurrences)", 2, 30, 3)
        seqs = build_sequences(sig, convs)
        table = transition_table(seqs, convs, min_support=min_support)

        if table.empty:
            st.info("No transitions meet the support threshold. Lower the slider.")
        else:
            st.subheader("Transition graph")
            st.graphviz_chart(to_dot(table, max_edges=45), use_container_width=True)

            st.divider()
            st.subheader("Worst-converting transitions")
            st.caption("Edges (with enough support) whose conversations convert least often.")
            worst = worst_transitions(table, n=15)
            disp = worst.copy()
            disp["conv_rate"] = (disp["conv_rate"] * 100).round(1)
            disp["p_lost"] = (disp["p_lost"] * 100).round(1)
            disp["prob"] = (disp["prob"] * 100).round(1)
            st.dataframe(
                disp[["from_state", "to_state", "count", "prob", "conv_rate", "p_lost"]].rename(columns={
                    "from_state": "From", "to_state": "To", "count": "Occurrences",
                    "prob": "P(transition) %", "conv_rate": "Conversion %", "p_lost": "P(lost) %",
                }),
                use_container_width=True, hide_index=True,
            )

            # ── Inspect a transition in chat mode ────────────────────────────
            st.divider()
            st.subheader("Inspect a transition in chat")
            st.caption("Pick a worst-converting transition to see real conversations where it "
                       "occurs, with the two turns highlighted in context.")

            options = list(zip(worst["from_state"], worst["to_state"]))
            if options:
                pick = st.selectbox(
                    "Transition",
                    options,
                    format_func=lambda ab: f"{ab[0]} → {ab[1]}  "
                    f"({worst[(worst.from_state==ab[0]) & (worst.to_state==ab[1])]['conv_rate'].values[0]*100:.0f}% convert)",
                )
                a, b = pick

                # per-conversation ordered (turn_index, intent) to locate the adjacent pair
                conv_states = {
                    cid: list(zip(g["turn_index"], g["intent"]))
                    for cid, g in sig.sort_values("turn_index").groupby("conversation_id")
                }
                examples = []  # (conv_id, from_turn_index, to_turn_index)
                for cid, states in conv_states.items():
                    for (ti, si), (tj, sj) in zip(states, states[1:]):
                        if si == a and sj == b:
                            examples.append((cid, int(ti), int(tj)))
                            break

                conv_outcome = convs.set_index("conversation_id")["outcome"].to_dict()
                msg_map = conv_messages_map()
                st.caption(f"{len(examples)} conversation(s) contain this transition. Showing up to 3.")
                for cid, ti, tj in examples[:3]:
                    oc = conv_outcome.get(cid, "?")
                    with st.expander(f"{oc.upper()} · conversation `{cid}`  (turns {ti}→{tj})"):
                        raw = msg_map.get(cid)
                        if raw is not None:
                            render_messages(parse_messages(raw), highlight={ti, tj})
                        else:
                            st.caption("Transcript not found in current dataset.")

# ═══════════════════════════════════════════════════════════════════════════
# TAB 8 — INTENT PATTERNS
# ═══════════════════════════════════════════════════════════════════════════

with tab_patterns:
    from analysis.markov import build_sequences as _build_seqs
    from analysis.patterns import archetype_summary, archetypes, lost_patterns

    st.title("Intent Patterns")
    st.caption("Client-intent sequences: conversation archetypes and the intent patterns "
               "most associated with lost conversations.")

    sig, convs = load_signals_frames()
    if sig is None:
        st.info(_SIGNALS_EMPTY_MSG)
    else:
        st.warning(
            f"⚠️ Limited sample: {len(convs)} fully-labeled conversations "
            "(labeling interrupted by an API credit limit).",
            icon="⚠️",
        )
        client_sig = sig[sig["role"] == "client"]
        client_seqs = _build_seqs(client_sig, convs)

        st.subheader("Conversation archetypes")
        st.caption("Conversations clustered by their intent sequence (TF-IDF of intent n-grams + KMeans). "
                   "Sorted by conversion rate — worst first.")
        asg = archetypes(client_seqs, convs)
        summ = archetype_summary(asg, convs, client_seqs)
        disp = summ.copy()
        disp["conv_rate"] = (disp["conv_rate"] * 100).round(1)
        st.dataframe(
            disp.rename(columns={
                "cluster": "Archetype", "size": "Conversations",
                "conv_rate": "Conversion %", "top_intents": "Characteristic intents",
            }),
            use_container_width=True, hide_index=True,
        )
        st.bar_chart(summ.set_index("cluster")["conv_rate"], horizontal=True, use_container_width=True)

        # ── Drill into an archetype's conversations ──────────────────────────
        st.markdown("**Browse conversations in an archetype**")
        conv_outcome = convs.set_index("conversation_id")["outcome"].to_dict()
        msg_map = conv_messages_map()
        cluster_opts = summ["cluster"].tolist()
        sel_cluster = st.selectbox(
            "Archetype",
            cluster_opts,
            format_func=lambda c: f"Archetype {c}  "
            f"({int(summ[summ.cluster==c]['size'].values[0])} chats, "
            f"{summ[summ.cluster==c]['conv_rate'].values[0]*100:.0f}% convert)",
        )
        members = asg[asg["cluster"] == sel_cluster]["conversation_id"].tolist()
        st.caption(f"{len(members)} conversations in this archetype.")
        for cid in members:
            oc = conv_outcome.get(cid, "?")
            mark = "✓" if oc in ("closed", "human-closed") else "✗"
            with st.expander(f"[{mark}] {oc.upper()} · conversation `{cid}`"):
                raw = msg_map.get(cid)
                if raw is not None:
                    render_messages(parse_messages(raw))
                else:
                    st.caption("Transcript not found in current dataset.")

        st.divider()
        st.subheader("Top recurring patterns in lost conversations")
        st.caption("Frequent contiguous intent sequences, ranked by frequency × loss rate.")

        lp = lost_patterns(client_seqs, convs, top=12)
        if lp.empty:
            st.info("Not enough repeated patterns yet.")
        else:
            id_to_msgs = conv_messages_map()
            conv_lost = convs.set_index("conversation_id")["converted"].to_dict()

            def _contains(seq, pat):
                n = len(pat)
                return any(tuple(seq[i:i + n]) == tuple(pat) for i in range(len(seq) - n + 1))

            for _, p in lp.iterrows():
                pattern = p["pattern"]
                title = " → ".join(pattern)
                header = (f"{title}  ·  {int(p['count'])} chats  ·  "
                          f"{p['conv_rate']*100:.0f}% convert")
                with st.expander(header):
                    # find lost example conversations containing this pattern
                    examples = [
                        cid for cid, seq in client_seqs.items()
                        if conv_lost.get(cid) is False and _contains(seq, pattern)  # lost only
                    ][:2]
                    if not examples:
                        st.caption("No lost example available.")
                    for cid in examples:
                        st.markdown(f"**Conversation `{cid}`**")
                        msgs = id_to_msgs.get(cid)
                        if msgs is not None:
                            render_messages(parse_messages(msgs))
                        else:
                            st.caption("Transcript not found in current dataset.")
