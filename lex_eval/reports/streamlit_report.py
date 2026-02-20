"""
Streamlit evaluation dashboard for LexChat eval runs.

Launch:
    streamlit run lex_eval/reports/streamlit_report.py

Or via run_evals.py:
    python lex_eval/run_evals.py --report streamlit
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
_REPORTS_DIR = _HERE
_DATA_DIR = _HERE.parent / "data"

EVAL_JSON = _REPORTS_DIR / "eval_report.json"
RESPONSES_JSONL = _DATA_DIR / "responses.jsonl"

# ---------------------------------------------------------------------------
# Data loading (cached so navigation re-renders are instant)
# ---------------------------------------------------------------------------


@st.cache_data
def load_eval_results() -> list[dict]:
    """Load raw eval results from eval_report.json."""
    with open(EVAL_JSON) as f:
        return json.load(f)


@st.cache_data
def load_responses() -> dict[tuple[str, int], list[dict]]:
    """
    Load responses.jsonl and index by (llm_name, question_id).
    Each key maps to a list of response records (usually 2 runs).
    """
    idx: dict[tuple[str, int], list[dict]] = defaultdict(list)
    with open(RESPONSES_JSONL) as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                key = (rec["llm_name"], int(rec["question_id"]))
                idx[key].append(rec)
    return dict(idx)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _aggregate_metrics(results: list[dict]) -> list[dict]:
    """
    Return one aggregated result per metric type, deduplicating within each
    test_name and computing mean/min/max across distinct test names.
    """
    by_metric: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_metric[r["metric_name"]].append(r)

    aggregated: list[dict] = []
    for metric_name, metric_results in by_metric.items():
        # Step 1: deduplicate within each test_name (keep first occurrence)
        seen: dict[str, dict] = {}
        for r in metric_results:
            if r["test_name"] not in seen:
                seen[r["test_name"]] = r
        deduped = list(seen.values())

        if metric_name == "Consistency":
            aggregated.append(deduped[0])
        else:
            scores = [r["score"] for r in deduped]
            mean_score = sum(scores) / len(scores)
            aggregated.append(
                {
                    **deduped[0],
                    "score": mean_score,
                    "min_score": min(scores),
                    "max_score": max(scores),
                    "test_names": [r["test_name"] for r in deduped],
                    "raw_results": deduped,
                    "passed": mean_score >= deduped[0]["threshold"],
                }
            )
    return aggregated


def _build_hierarchy(
    raw: list[dict],
) -> dict[str, dict[int, list[dict]]]:
    """
    Group raw results into:
        llm_name → question_id → [aggregated metric results]
    """
    grouped: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in raw:
        grouped[r["llm_name"]][int(r["question_id"])].append(r)
    # Aggregate per metric
    hierarchy: dict[str, dict[int, list[dict]]] = {}
    for llm, questions in grouped.items():
        hierarchy[llm] = {}
        for qid, results in questions.items():
            hierarchy[llm][qid] = _aggregate_metrics(results)
    return hierarchy


# ---------------------------------------------------------------------------
# Helpers for score badge rendering
# ---------------------------------------------------------------------------

_SCORE_THRESHOLDS = [(0.95, "excellent"), (0.80, "good"), (0.60, "warning")]

_BADGE_COLOURS = {
    "excellent": ("#1a4d2e", "#3fb950"),
    "good": ("#2d3a1f", "#7ee787"),
    "warning": ("#4a3a1f", "#f0ad4e"),
    "poor": ("#4c1f1f", "#f85149"),
    "passed": ("#1a4d2e", "#3fb950"),
    "failed": ("#4c1f1f", "#f85149"),
}


def _score_level(score: float) -> str:
    for threshold, level in _SCORE_THRESHOLDS:
        if score >= threshold:
            return level
    return "poor"


def _score_badge(score: float | str, level: str | None = None) -> str:
    """Return an inline-HTML coloured score badge."""
    if isinstance(score, float):
        text = f"{score:.3f}"
        lvl = level or _score_level(score)
    else:
        text = str(score)
        lvl = level or "poor"
    bg, fg = _BADGE_COLOURS.get(lvl, ("#30363d", "#c9d1d9"))
    return (
        f'<span style="background:{bg};color:{fg};padding:2px 8px;'
        f'border-radius:4px;font-family:monospace;font-size:0.85em;'
        f'font-weight:600;">{text}</span>'
    )


def _status_icon(passed: bool) -> str:
    return "✅" if passed else "❌"


# ---------------------------------------------------------------------------
# Page sections
# ---------------------------------------------------------------------------


def _render_top_summary(hierarchy: dict) -> None:
    """Per-LLM summary row shown at the top of the page."""
    rows_html = ""
    for llm in sorted(hierarchy.keys()):
        q_data = hierarchy[llm]
        all_m = [r for results in q_data.values() for r in results]
        total = len(all_m)
        passed = sum(1 for r in all_m if r["passed"])
        failed = total - passed
        pct = passed / total * 100 if total else 0.0
        pct_colour = "#3fb950" if pct >= 80 else "#f0ad4e" if pct >= 50 else "#f85149"
        rows_html += (
            f"<tr>"
            f'<td style="padding:8px 14px;color:#c9d1d9;font-weight:600;">{llm}</td>'
            f'<td style="padding:8px 14px;color:#3fb950;font-weight:600;">{passed}</td>'
            f'<td style="padding:8px 14px;color:#f85149;font-weight:600;">{failed}</td>'
            f'<td style="padding:8px 14px;font-family:monospace;color:#8b949e;">{total}</td>'
            f'<td style="padding:8px 14px;font-weight:600;color:{pct_colour};">{pct:.1f}%</td>'
            f"</tr>"
        )
    st.markdown(
        f"""
        <table style="border-collapse:collapse;width:100%;
                      background:#161b22;border-radius:6px;overflow:hidden;margin-bottom:0.5rem;">
          <thead>
            <tr style="background:#21262d;color:#8b949e;font-size:0.8em;text-transform:uppercase;">
              <th style="padding:8px 14px;text-align:left;">LLM</th>
              <th style="padding:8px 14px;text-align:left;">Passed</th>
              <th style="padding:8px 14px;text-align:left;">Failed</th>
              <th style="padding:8px 14px;text-align:left;">Total</th>
              <th style="padding:8px 14px;text-align:left;">Pass Rate</th>
            </tr>
          </thead>
          <tbody>{rows_html}</tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )


def _render_llm_summary_bar(llm: str, q_data: dict[int, list[dict]]) -> None:
    """One-line header stats for an LLM."""
    all_m = [r for results in q_data.values() for r in results]
    total = len(all_m)
    passed = sum(1 for r in all_m if r["passed"])
    pct = passed / total * 100 if total else 0.0
    colour = "#3fb950" if pct >= 80 else "#f0ad4e" if pct >= 50 else "#f85149"
    st.markdown(
        f"**{passed}/{total}** metrics passed &nbsp;"
        f'<span style="color:{colour};font-weight:600;">{pct:.1f}%</span>',
        unsafe_allow_html=True,
    )


def _render_metric_summary_table(metrics: list[dict]) -> None:
    """
    Compact summary row per metric showing:
    - metric name, score badge (+ min/max for aggregated), threshold, status
    """
    rows_html = ""
    for m in metrics:
        name = m["metric_name"]
        score = m["score"]
        threshold = m["threshold"]
        passed = m["passed"]
        has_range = "min_score" in m and "max_score" in m

        badge = _score_badge(score)
        status = _status_icon(passed)

        if has_range:
            min_s = m["min_score"]
            max_s = m["max_score"]
            score_cell = (
                f"{badge}"
                f'&nbsp;<span style="font-size:0.78em;color:#8b949e;">'
                f"min&nbsp;<code>{min_s:.3f}</code>&nbsp;"
                f"max&nbsp;<code>{max_s:.3f}</code></span>"
            )
        else:
            score_cell = badge

        rows_html += (
            f"<tr>"
            f'<td style="padding:6px 12px;color:#c9d1d9;">{name}</td>'
            f'<td style="padding:6px 12px;">{score_cell}</td>'
            f'<td style="padding:6px 12px;font-family:monospace;color:#8b949e;">{threshold:.3f}</td>'
            f'<td style="padding:6px 12px;font-size:1.1em;">{status}</td>'
            f"</tr>"
        )

    table_html = f"""
    <table style="border-collapse:collapse;width:100%;
                  background:#161b22;border-radius:6px;overflow:hidden;">
      <thead>
        <tr style="background:#21262d;color:#8b949e;font-size:0.8em;text-transform:uppercase;">
          <th style="padding:8px 12px;text-align:left;">Metric</th>
          <th style="padding:8px 12px;text-align:left;">Score</th>
          <th style="padding:8px 12px;text-align:left;">Threshold</th>
          <th style="padding:8px 12px;text-align:left;">Status</th>
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>
    """
    st.markdown(table_html, unsafe_allow_html=True)


def _render_metric_detail(metrics: list[dict]) -> None:
    """
    Drill-down: show individual raw eval results for each metric.
    For aggregated metrics also shows per-test breakdown.
    """
    for m in metrics:
        name = m["metric_name"]
        has_range = "min_score" in m

        with st.expander(
            f"**{name}** — {_status_icon(m['passed'])} score: {m['score']:.3f}",
            expanded=False,
        ):
            if has_range:
                st.markdown(
                    f"**Mean:** `{m['score']:.3f}` &nbsp;|&nbsp; "
                    f"**Min:** `{m['min_score']:.3f}` &nbsp;|&nbsp; "
                    f"**Max:** `{m['max_score']:.3f}` &nbsp;|&nbsp; "
                    f"**Threshold:** `{m['threshold']:.3f}`"
                )
                st.markdown("**Individual test results:**")
                for raw in m.get("raw_results", []):
                    _render_single_eval_result(raw)
            else:
                # Consistency — single result
                _render_single_eval_result(m)


def _render_single_eval_result(r: dict) -> None:
    """Render one raw eval result entry."""
    passed = r["passed"]
    colour = "#3fb950" if passed else "#f85149"
    label = "✓ Passed" if passed else "✗ Failed"

    st.markdown(
        f'<div style="background:#0d1117;border-left:3px solid {colour};'
        f'padding:10px 14px;border-radius:4px;margin:6px 0;">'
        f'<span style="color:#8b949e;font-size:0.8em;">'
        f'{r.get("test_name","")}</span>&nbsp;&nbsp;'
        f'<span style="color:{colour};font-size:0.85em;font-weight:600;">{label}</span>'
        f"&nbsp;&nbsp;score: <code>{r['score']:.3f}</code>"
        f'<div style="color:#8b949e;font-size:0.85em;margin-top:6px;">'
        f'{r.get("reason","")}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )

    tools = r.get("tools_used")
    if tools:
        st.caption(f"Tools used: {', '.join(tools)}")

    if r.get("error"):
        st.error(r["error"])


def _render_chat_interaction(records: list[dict]) -> None:
    """
    Render the raw chat interaction(s) for an LLM/question pair.
    Each record in responses.jsonl is one run.
    """
    if not records:
        st.info("No response records found in responses.jsonl for this combination.")
        return

    run_tabs = st.tabs([f"Run {i + 1}  ({r['timestamp'][:19]})" for i, r in enumerate(records)])

    for tab, rec in zip(run_tabs, records):
        with tab:
            tc = rec.get("test_case", {})

            # ---- LLM Answer ----
            st.markdown("#### LLM Answer")
            actual = tc.get("actual_output", "")
            if actual:
                st.markdown(actual)
            else:
                st.caption("_(no output captured)_")

            st.divider()

            # ---- Tools Called ----
            tools_called: list[dict] = tc.get("tools_called") or []
            if tools_called:
                st.markdown(f"#### Tools Called ({len(tools_called)})")
                for i, tool in enumerate(tools_called):
                    tool_name = tool.get("name", f"tool_{i}")
                    with st.expander(f"🔧 {tool_name}", expanded=False):
                        output_raw = tool.get("output", "")
                        # Try to parse as JSON for nice rendering
                        if isinstance(output_raw, str):
                            try:
                                parsed = json.loads(output_raw)
                                st.json(parsed, expanded=False)
                            except (json.JSONDecodeError, ValueError):
                                st.code(output_raw, language="text")
                        elif isinstance(output_raw, (dict, list)):
                            st.json(output_raw, expanded=False)
                        else:
                            st.text(str(output_raw))
            else:
                st.caption("No tools_called data captured.")

            st.divider()

            # ---- Retrieval Context ----
            contexts: list[str] = tc.get("retrieval_context") or []
            if contexts:
                st.markdown(f"#### Retrieved Context ({len(contexts)} items)")
                for i, ctx in enumerate(contexts):
                    with st.expander(f"📄 Context {i + 1}", expanded=False):
                        # Legislation text — render as plain text, not markdown,
                        # to avoid spurious formatting
                        st.text(ctx)
            else:
                st.caption("No retrieval context captured.")

            # ---- Metadata ----
            with st.expander("ℹ️ Record metadata", expanded=False):
                st.json(
                    {
                        "timestamp": rec.get("timestamp"),
                        "deep_research": rec.get("deep_research"),
                        "llm_name": rec.get("llm_name"),
                        "question_id": rec.get("question_id"),
                    }
                )


def _render_question_block(
    qid: int,
    question_text: str,
    metrics: list[dict],
    response_records: list[dict],
) -> None:
    """Full block for one question within an LLM section."""
    all_pass = all(m["passed"] for m in metrics)
    n_pass = sum(1 for m in metrics if m["passed"])
    n_total = len(metrics)
    icon = "✅" if all_pass else ("⚠️" if n_pass > 0 else "❌")

    with st.expander(
        f"{icon}  **Q{qid}** — {question_text[:120]}{'…' if len(question_text) > 120 else ''}  "
        f"*({n_pass}/{n_total} metrics passed)*",
        expanded=False,
    ):
        # Metric summary table always visible at the top
        _render_metric_summary_table(metrics)

        st.markdown("")  # spacer

        # Drill-down via tabs
        detail_tab, chat_tab = st.tabs(["📊 Metric Detail", "💬 Chat Interaction"])

        with detail_tab:
            _render_metric_detail(metrics)

        with chat_tab:
            _render_chat_interaction(response_records)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="LexChat Eval Dashboard",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.title("⚖️ LexChat Evaluation Dashboard")

    # Load data
    if not EVAL_JSON.exists():
        st.error(f"eval_report.json not found at {EVAL_JSON}. Run the eval suite first.")
        st.stop()

    raw_results = load_eval_results()
    hierarchy = _build_hierarchy(raw_results)
    responses = load_responses() if RESPONSES_JSONL.exists() else {}

    if not responses:
        st.warning(f"responses.jsonl not found at {RESPONSES_JSONL} — chat interaction tab will be empty.")

    # Overall summary — one row per LLM
    _render_top_summary(hierarchy)
    st.divider()

    # One tab per LLM
    llm_names = sorted(hierarchy.keys())
    llm_tabs = st.tabs(llm_names)

    for tab, llm in zip(llm_tabs, llm_names):
        with tab:
            q_data = hierarchy[llm]
            _render_llm_summary_bar(llm, q_data)
            st.markdown("")  # spacer

            for qid in sorted(q_data.keys()):
                metrics = q_data[qid]
                question_text = metrics[0].get("question", "")
                response_records = responses.get((llm, qid), [])
                _render_question_block(qid, question_text, metrics, response_records)


if __name__ == "__main__":
    main()
