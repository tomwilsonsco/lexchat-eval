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


_HERE = Path(__file__).parent
_REPORTS_DIR = _HERE
_DATA_DIR = _HERE.parent / "data"

SUITE_RESULTS = {
    "groundedness": _REPORTS_DIR / "groundedness_results.json",
    "consistency": _REPORTS_DIR / "consistency_results.json",
    "consistency_llm": _REPORTS_DIR / "consistency_llm_results.json",
    "tool_usage": _REPORTS_DIR / "tool_usage_results.json",
    "structure": _REPORTS_DIR / "structure_results.json",
}
RESPONSES_JSONL = _DATA_DIR / "responses.jsonl"


def _result_file_mtimes() -> tuple[float, ...]:
    """Return modification times of all result files for cache-busting."""
    return tuple(
        path.stat().st_mtime if path.exists() else 0.0
        for path in SUITE_RESULTS.values()
    )


@st.cache_data
def load_eval_results(_mtimes: tuple[float, ...] = ()) -> list[dict]:
    """Load and merge raw eval results from all per-suite JSON files."""
    all_results: list[dict] = []
    for suite, path in SUITE_RESULTS.items():
        if path.exists():
            try:
                with open(path) as f:
                    results = json.load(f)
                all_results.extend(results)
            except (json.JSONDecodeError, ValueError):
                pass
    return all_results


@st.cache_data
def load_responses(_mtime: float = 0.0) -> dict[tuple[str, int], list[dict]]:
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



_AGGREGATE_ONLY_METRICS = {"Consistency (Simple)", "Consistency (AI Judge)"}

# ---------------------------------------------------------------------------
# Metric display order
# Edit this list to control the order metrics appear in per-question tables
# and detail panels.  Any metric not listed here will appear at the end.
# ---------------------------------------------------------------------------
METRIC_DISPLAY_ORDER: list[str] = [
    "Tool Usage",
    "Research Output Structure",
    "Reference Links",
    "Consistency (Simple)",
    "Consistency (AI Judge)",
    "Answer Relevancy",
    "Faithfulness",
]


def _metric_sort_key(metric: dict) -> int:
    name = metric["metric_name"]
    try:
        return METRIC_DISPLAY_ORDER.index(name)
    except ValueError:
        return len(METRIC_DISPLAY_ORDER)


def _aggregate_metrics(results: list[dict]) -> list[dict]:
    """
    Return one aggregated result per metric type.

    - **Consistency**: single aggregated entry (no per-run breakdown).
    - **Everything else** (Faithfulness, Answer Relevancy, Tool Usage, …):
      keeps every individual run, computes mean/min/max over all of them,
      and stores the full list in ``raw_results`` for the detail expander.
    """
    by_metric: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_metric[r["metric_name"]].append(r)

    aggregated: list[dict] = []
    for metric_name, metric_results in by_metric.items():
        if metric_name in _AGGREGATE_ONLY_METRICS:
            # Consistency — show a single summary, no per-run drill-down.
            aggregated.append(metric_results[0])
        else:
            # Faithfulness / Answer Relevancy / Tool Usage —
            # keep *all* runs so each one appears in the detail expander.
            scores = [r["score"] for r in metric_results]
            mean_score = sum(scores) / len(scores)
            aggregated.append(
                {
                    **metric_results[0],
                    "score": mean_score,
                    "min_score": min(scores),
                    "max_score": max(scores),
                    "n_runs": len(metric_results),
                    "test_names": [r["test_name"] for r in metric_results],
                    "raw_results": metric_results,
                    "passed": mean_score >= metric_results[0]["threshold"],
                }
            )
    return sorted(aggregated, key=_metric_sort_key)


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
    # aggregate per metric
    hierarchy: dict[str, dict[int, list[dict]]] = {}
    for llm, questions in grouped.items():
        hierarchy[llm] = {}
        for qid, results in questions.items():
            hierarchy[llm][qid] = _aggregate_metrics(results)
    return hierarchy


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


def _render_top_summary(hierarchy: dict) -> None:
    """Per-LLM summary rows at the top of the page, each expandable to show
    mean score per metric across all questions."""
    for llm in sorted(hierarchy.keys()):
        q_data = hierarchy[llm]
        all_m = [r for results in q_data.values() for r in results]
        total = len(all_m)
        passed = sum(1 for r in all_m if r["passed"])
        failed = total - passed
        pct = passed / total * 100 if total else 0.0
        pct_colour = "#3fb950" if pct >= 80 else "#f0ad4e" if pct >= 50 else "#f85149"

        label = (
            f"**{llm}** &nbsp;|&nbsp; "
            f"Passed: **{passed}** &nbsp; Failed: **{failed}** &nbsp; "
            f"Total: **{total}** &nbsp; Pass Rate: **{pct:.1f}%**"
        )

        with st.expander(label, expanded=False):
            # Compute mean score per metric (in display order)
            by_metric: dict[str, list[float]] = defaultdict(list)
            thresholds: dict[str, float] = {}
            for m in all_m:
                name = m["metric_name"]
                by_metric[name].append(m["score"])
                thresholds[name] = m["threshold"]

            # Sort by METRIC_DISPLAY_ORDER
            metric_names = sorted(
                by_metric.keys(),
                key=lambda n: METRIC_DISPLAY_ORDER.index(n)
                if n in METRIC_DISPLAY_ORDER
                else len(METRIC_DISPLAY_ORDER),
            )

            rows_html = ""
            for name in metric_names:
                scores = by_metric[name]
                mean = sum(scores) / len(scores)
                threshold = thresholds[name]
                badge = _score_badge(mean)
                pass_count = sum(1 for s in scores if s >= threshold)
                rows_html += (
                    f"<tr>"
                    f'<td style="padding:6px 14px;color:#c9d1d9;">{name}</td>'
                    f'<td style="padding:6px 14px;">{badge}</td>'
                    f'<td style="padding:6px 14px;font-family:monospace;color:#8b949e;">{threshold:.2f}</td>'
                    f'<td style="padding:6px 14px;font-family:monospace;color:#8b949e;">{pass_count}/{len(scores)}</td>'
                    f"</tr>"
                )

            st.markdown(
                f"""
                <table style="border-collapse:collapse;width:100%;
                              background:#161b22;border-radius:6px;overflow:hidden;">
                  <thead>
                    <tr style="background:#21262d;color:#8b949e;font-size:0.8em;text-transform:uppercase;">
                      <th style="padding:8px 14px;text-align:left;">Metric</th>
                      <th style="padding:8px 14px;text-align:left;">Mean Score</th>
                      <th style="padding:8px 14px;text-align:left;">Threshold</th>
                      <th style="padding:8px 14px;text-align:left;">Questions Passed</th>
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
    For aggregated metrics also shows per-run breakdown.
    Uses containers (not expanders) to avoid nested-expander violations.
    """
    for m in metrics:
        name = m["metric_name"]
        has_range = "min_score" in m
        icon = _status_icon(m["passed"])

        st.markdown(
            f"**{name}** {icon} — score: `{m['score']:.3f}`",
        )
        with st.container():
            if has_range:
                n = m.get("n_runs", len(m.get("raw_results", [])))
                st.markdown(
                    f"**Mean:** `{m['score']:.3f}` &nbsp;|&nbsp; "
                    f"**Min:** `{m['min_score']:.3f}` &nbsp;|&nbsp; "
                    f"**Max:** `{m['max_score']:.3f}` &nbsp;|&nbsp; "
                    f"**Threshold:** `{m['threshold']:.3f}` &nbsp;|&nbsp; "
                    f"**Runs:** `{n}`"
                )
                for idx, raw in enumerate(m.get("raw_results", []), 1):
                    _render_single_eval_result(raw, run_label=f"Run {idx}")
            else:
                # Consistency — single aggregated result, no per-run breakdown
                _render_single_eval_result(m)
        st.divider()


def _render_single_eval_result(r: dict, run_label: str | None = None) -> None:
    """Render one raw eval result entry."""
    passed = r["passed"]
    colour = "#3fb950" if passed else "#f85149"
    label = "✓ Passed" if passed else "✗ Failed"
    prefix = f"{run_label} — " if run_label else ""

    st.markdown(
        f'<div style="background:#0d1117;border-left:3px solid {colour};'
        f'padding:10px 14px;border-radius:4px;margin:6px 0;">'
        f'<span style="color:#8b949e;font-size:0.8em;">'
        f'{prefix}{r.get("test_name","")}</span>&nbsp;&nbsp;'
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

            st.markdown("#### LLM Answer")
            actual = tc.get("actual_output", "")
            if actual:
                st.markdown(actual)
            else:
                st.caption("_(no output captured)_")

            st.divider()

            tools_called: list[dict] = tc.get("tools_called") or []
            if tools_called:
                st.markdown(f"#### Tools Called ({len(tools_called)})")
                for i, tool in enumerate(tools_called):
                    tool_name = tool.get("name", f"tool_{i}")
                    st.markdown(f"🔧 **{tool_name}**")
                    output_raw = tool.get("output", "")
                    with st.container():
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

            contexts: list[str] = tc.get("retrieval_context") or []
            if contexts:
                st.markdown(f"#### Retrieved Context ({len(contexts)} items)")
                for i, ctx in enumerate(contexts):
                    st.markdown(f"📄 **Context {i + 1}**")
                    with st.container():
                        st.text(ctx)
            else:
                st.caption("No retrieval context captured.")

            st.markdown("ℹ️ **Record metadata**")
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
        # summary table always visible at the top
        _render_metric_summary_table(metrics)

        st.markdown("")  # spacer

        # drill-down via tabs
        detail_tab, chat_tab = st.tabs(["📊 Metric Detail", "💬 Chat Interaction"])

        with detail_tab:
            _render_metric_detail(metrics)

        with chat_tab:
            _render_chat_interaction(response_records)


def main() -> None:
    st.set_page_config(
        page_title="LexChat Eval",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.title("LexChat Evaluation")

    # load data
    available = [name for name, path in SUITE_RESULTS.items() if path.exists()]
    if not available:
        st.error(
            "No results files found. Run the eval suite first:\n\n"
            "```\npython lex_eval/run_evals.py\n```"
        )
        st.stop()

    st.caption(f"Loaded results from: {', '.join(available)}")

    raw_results = load_eval_results(_mtimes=_result_file_mtimes())
    hierarchy = _build_hierarchy(raw_results)
    _resp_mtime = RESPONSES_JSONL.stat().st_mtime if RESPONSES_JSONL.exists() else 0.0
    responses = load_responses(_mtime=_resp_mtime) if RESPONSES_JSONL.exists() else {}

    if not responses:
        st.warning(f"responses.jsonl not found at {RESPONSES_JSONL} — chat interaction tab will be empty.")

    _render_top_summary(hierarchy)
    st.divider()

    # tab per LLM
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
