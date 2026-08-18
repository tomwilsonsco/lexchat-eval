from __future__ import annotations

import html
import json
import sys
from collections import defaultdict
from pathlib import Path

import streamlit as st

# Ensure the repo root is importable when Streamlit launches this file directly.
_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from lex_eval.utils.db import (
    DEFAULT_DB,
    load_eval_results as db_load_eval_results,
    load_records as db_load_records,
)

script_dir = Path(__file__).parent
data_dir = script_dir.parent / "data"

RESPONSES_DB = DEFAULT_DB


DEFAULT_CHAT_MODE = "research"


@st.cache_data
def load_eval_results(_db_mtime: float = 0.0) -> list[dict]:
    """Load every metric's eval_<metric> table and tag each row with the
    test_name/metric_name implied by which table it came from (the table
    itself doesn't store them, since the table name already is the metric).

    Also tags each row with the chat_mode of the response it scored. The
    eval_<metric> tables have no chat_mode column, so it is looked up through
    response_id. Everything downstream groups on it, so a deep research run is
    never averaged together with a single-shot one."""
    modes = _response_chat_modes()
    results: list[dict] = []
    for key, display_name, _tooltip in METRICS:
        for row in db_load_eval_results(RESPONSES_DB, metric=key):
            results.append(
                {
                    **row,
                    "test_name": key,
                    "metric_name": display_name,
                    # "unknown" rather than a silent default: an eval row whose
                    # response_id matches no response is a broken FK, and it
                    # should surface as its own group rather than quietly
                    # inflating the single-shot numbers.
                    "chat_mode": modes.get(row.get("response_id"), "unknown"),
                }
            )
    return results


def _response_chat_modes() -> dict[int, str]:
    """response_id -> chat_mode, for tagging eval rows."""
    return {
        int(rec["response_id"]): (rec.get("chat_mode") or DEFAULT_CHAT_MODE)
        for rec in db_load_records(RESPONSES_DB)
        if rec.get("response_id") is not None
    }


@st.cache_data
def load_responses(_mtime: float = 0.0) -> dict[tuple[str, str, int], list[dict]]:
    """
    Load responses from DuckDB and index by (llm_name, chat_mode, question_id).
    Each key maps to a list of response records (could be 2+ runs).

    chat_mode is part of the key so the Chat Interaction tab shows the same runs
    the metrics above it were scored on, rather than every run of the question.
    """
    idx: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for rec in db_load_records(RESPONSES_DB):
        key = (
            rec["llm_name"],
            rec.get("chat_mode") or DEFAULT_CHAT_MODE,
            int(rec["question_id"]),
        )
        idx[key].append(rec)
    return dict(idx)


# Single source of truth for every metric this dashboard displays: its
# eval_<key> table, its display name, and its tooltip, in display order.
# Keys here must match run_evals.py::METRIC_FILES.
METRICS: list[tuple[str, str, str]] = [
    (
        "tool_usage",
        "Tool Usage",
        "Are all of delegate research, search legislation and search legislation sections used, in the correct order (search legislation - search legislation sections - get legislation text if needed), and does the Worker stick to that order rather than looping back to an earlier step later in the same run?",
    ),
    (
        "mandatory_structure",
        "Research Output Structure",
        "Does the worker agent return the findings to the manager with the requested headers.",
    ),
    (
        "citation_passthrough",
        "Reference Links",
        "Are all reference links found by the researcher included in the final answer given to the user.",
    ),
    (
        "citation_grounding",
        "Citation Grounding",
        "Does every Act cited in the researcher's report correspond to legislation the run's own tool calls actually retrieved, rather than one invented by the model.",
    ),
    (
        "citation_domain",
        "Citation Domain",
        "Does every citation link in the researcher's report point to legislation.gov.uk, the only domain the Worker is permitted to cite.",
    ),
    (
        "genuine_gap",
        "Genuine Gap",
        "When retrieval found no usable legislation text, does the researcher's report say so plainly instead of answering with unsupported confidence.",
    ),
    (
        "consistency",
        "Consistency (Cosine)",
        "Compare the answers provided when the same question is asked multiple times in the same chat mode, using TF cosine similarity. Research and deep research answers are never compared against each other, and a mode with only one stored run is not scored. Any legislation section cited in one answer but not the other is listed in the detail, but does not decide pass or fail.",
    ),
    (
        "citation_agreement",
        "Citation Agreement",
        "Of the legislation provisions the hand written reference answer cites, how many does the response cite too. No AI judge, it compares the two lists of legislation.gov.uk links.",
    ),
    (
        "reference_answer_agreement",
        "Reference Answer Agreement",
        "AI as a judge metric: How many of the question's key statements the response also makes, at most 5 of them. The statements are written once alongside the hand written reference answer and stored with it, so the judge labels a fixed list rather than picking the points afresh on every run. A second judge call looks for contradictions and nothing else, which is what catches a long answer that makes a point correctly in one section and then undoes it in another. A statement the response contradicts fails the metric outright, since a confidently wrong statement of law is worse than a missing one. The reference answers are unverified drafts, so read a flagged contradiction as a prompt to compare the two texts.",
    ),
    (
        "plan_coverage",
        "Plan Coverage",
        "Deep research only, AI as a judge metric: Does the approved research plan set out to cover the question's key statements, before any research happens. Reuses the same fixed statement list as Reference Answer Agreement rather than a separately authored golden plan.",
    ),
    (
        "response_groundedness",
        "Response Groundedness",
        "AI as a judge metric: Is the final answer to the user grounded in the research worker's summary. A near-unmodified copy is accepted automatically with no AI judge involved. Anything reworded enough to matter goes to the judge, which fails it on any unsupported claim or meaningful misrepresentation and passes only trivial wording differences. There is no partial credit, so the average is a pass rate.",
    ),
    (
        "claim_support",
        "Claim Support",
        "AI as a judge metric: What share of the report's verifiable legal claims are backed by text the researcher actually read? Claims whose truth depends on the absence of a provision are reported separately because absence generally cannot be established from retrieved excerpts or summaries.",
    ),
]

# order shown in streamlit
METRIC_DISPLAY_ORDER: list[str] = [name for _key, name, _tooltip in METRICS]

# hover over tips on app summary tables
METRIC_TOOLTIPS: dict[str, str] = {name: tip for _key, name, tip in METRICS}

# do not keep the individual response results of these metrics as only make sense
# comparing multiple
_AGGREGATE_ONLY_METRICS = {"Consistency (Cosine)"}

# Reason prefixes written by judge exceptions and harness capture gates (see
# metrics/*.py except blocks, tests/eval/test_groundedness.py gate functions,
# and structure.py's delegate_research precondition). Rows carrying one of
# these are not a genuine quality verdict, excluded from the mean, reported
# separately instead of averaged in as 0.0.
_NON_SCORED_PREFIXES = (
    "Judge error:",
    "Output too short",
    "No retrieval context captured",
    "No research output captured",
    "No reference outputs provided.",
    "No 'delegate_research' tool call found;",
    "No reference answer for this question;",
    "No reference statements for this question;",
    "No reference answer citations to compare against;",
    "No research plan for this record;",
)


def _is_scored(result: dict) -> bool:
    reason = result.get("reason") or ""
    return not reason.startswith(_NON_SCORED_PREFIXES)


def _metric_sort_key(metric: dict) -> int:
    name = metric["metric_name"]
    try:
        return METRIC_DISPLAY_ORDER.index(name)
    except ValueError:
        return len(METRIC_DISPLAY_ORDER)


def _aggregate_metrics(results: list[dict]) -> list[dict]:
    """
    return aggregated result per metric type.

    Consistency: single aggregated entry (no per-run breakdown).
    All other metrics:
      keep every individual run, computes mean/min/max over all of them,
      stores the full list in raw_results for the detail expander.
    """
    by_metric: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_metric[r["metric_name"]].append(r)

    aggregated: list[dict] = []
    for metric_name, metric_results in by_metric.items():
        if metric_name in _AGGREGATE_ONLY_METRICS:
            aggregated.append(
                {**metric_results[0], "scored": _is_scored(metric_results[0])}
            )
        else:
            scored = [r for r in metric_results if _is_scored(r)]
            not_scored = [r for r in metric_results if not _is_scored(r)]
            scores = [r["score"] for r in scored]
            mean_score = sum(scores) / len(scores) if scores else 0.0
            aggregated.append(
                {
                    **metric_results[0],
                    "score": mean_score,
                    "min_score": min(scores) if scores else 0.0,
                    "max_score": max(scores) if scores else 0.0,
                    "n_runs": len(metric_results),
                    "test_names": [r["test_name"] for r in metric_results],
                    "raw_results": metric_results,
                    "passed": bool(scores)
                    and mean_score >= metric_results[0]["threshold"],
                    "scored": bool(scores),
                    "not_scored_count": len(not_scored),
                    "not_scored_reasons": [r["reason"] for r in not_scored],
                }
            )
    return sorted(aggregated, key=_metric_sort_key)


def _build_hierarchy(
    raw: list[dict],
) -> dict[tuple[str, str], dict[int, list[dict]]]:
    """
    group raw results
    (llm_name, chat_mode) + question_id + [aggregated metric results]

    chat_mode is part of the key because a deep research run and a single-shot
    run of the same question are different products of the same model, and
    averaging them into one score hides the difference between them.
    """
    grouped: dict[tuple[str, str], dict[int, list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in raw:
        key = (r["llm_name"], r.get("chat_mode") or DEFAULT_CHAT_MODE)
        grouped[key][int(r["question_id"])].append(r)
    hierarchy: dict[tuple[str, str], dict[int, list[dict]]] = {}
    for key, questions in grouped.items():
        hierarchy[key] = {}
        for qid, results in questions.items():
            hierarchy[key][qid] = _aggregate_metrics(results)
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
    """inline-HTML coloured score badge."""
    if isinstance(score, float):
        text = f"{score:.3f}"
        lvl = level or _score_level(score)
    else:
        text = str(score)
        lvl = level or "poor"
    bg, fg = _BADGE_COLOURS.get(lvl, ("#30363d", "#c9d1d9"))
    return (
        f'<span style="background:{bg};color:{fg};padding:2px 8px;'
        f"border-radius:4px;font-family:monospace;font-size:0.85em;"
        f'font-weight:600;">{text}</span>'
    )


def _get_group_pass_rate(key: tuple[str, str], hierarchy: dict) -> float:
    """Calculate the overall pass rate for one (llm, chat_mode) group.

    Metric groups with nothing scored (every run a judge error or capture
    gate) are excluded entirely, not counted as failed.
    """
    q_data = hierarchy.get(key, {})
    all_m = [r for results in q_data.values() for r in results if r.get("scored", True)]
    total = len(all_m)
    return (sum(1 for r in all_m if r["passed"]) / total) if total else 0.0


def _group_label(key: tuple[str, str], show_mode: bool) -> str:
    """Display name for an (llm, chat_mode) group."""
    llm, mode = key
    return f"{llm}  ·  {_chat_mode_badge(mode)}" if show_mode else llm


def _sorted_group_keys(hierarchy: dict) -> list[tuple[str, str]]:
    """Groups worst pass rate first, so the ones needing attention lead."""
    return sorted(
        hierarchy.keys(), key=lambda k: (_get_group_pass_rate(k, hierarchy), k)
    )


def _render_top_summary(hierarchy: dict, show_mode: bool) -> None:
    """summary rows at the top of the page for each (llm, chat_mode) group.
    Expand to show mean score per metric across all questions."""
    for key in _sorted_group_keys(hierarchy):
        group_name = _group_label(key, show_mode)
        q_data = hierarchy[key]
        all_results = [r for results in q_data.values() for r in results]
        all_m = [r for r in all_results if r.get("scored", True)]
        n_na = len(all_results) - len(all_m)
        total = len(all_m)
        passed = sum(1 for r in all_m if r["passed"])
        failed = total - passed
        pct = passed / total * 100 if total else 0.0

        na_part = f" &nbsp; N/A: **{n_na}**" if n_na else ""
        label = (
            f"**{group_name}** &nbsp;|&nbsp; "
            f"Passed: **{passed}** &nbsp; Failed: **{failed}** &nbsp; "
            f"Total: **{total}**{na_part} &nbsp; Pass Rate: **{pct:.1f}%**"
        )

        with st.expander(label, expanded=False):
            # mean score per metric (in display order)
            by_metric: dict[str, list[float]] = defaultdict(list)
            thresholds: dict[str, float] = {}
            for m in all_m:
                name = m["metric_name"].strip()
                by_metric[name].append(m["score"])
                thresholds[name] = m["threshold"]

            metric_names = sorted(
                by_metric.keys(),
                key=lambda n: (
                    METRIC_DISPLAY_ORDER.index(n)
                    if n in METRIC_DISPLAY_ORDER
                    else len(METRIC_DISPLAY_ORDER)
                ),
            )

            rows_html = ""
            for name in metric_names:
                scores = by_metric[name]
                mean = sum(scores) / len(scores)
                threshold = thresholds[name]
                badge = _score_badge(mean)
                pass_count = sum(1 for s in scores if s >= threshold)
                tooltip = METRIC_TOOLTIPS.get(name, "")
                if tooltip:
                    name_cell = f'<span title="{tooltip}" style="cursor:help;color:#c9d1d9;">{name}</span>'
                else:
                    name_cell = f'<span style="color:#c9d1d9;">{name}</span>'
                rows_html += (
                    f"<tr>"
                    f'<td style="padding:6px 14px;">{name_cell}</td>'
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


def _render_llm_summary_bar(q_data: dict[int, list[dict]]) -> None:
    """header stats for one (llm, chat_mode) group"""
    all_m = [r for results in q_data.values() for r in results if r.get("scored", True)]
    total = len(all_m)
    passed = sum(1 for r in all_m if r["passed"])
    pct = passed / total * 100 if total else 0.0
    colour = "#3fb950" if pct >= 80 else "#f0ad4e" if pct >= 50 else "#f85149"
    st.markdown(
        f"**{passed}/{total}** metrics passed &nbsp;"
        f'<span style="color:{colour};font-weight:600;">{pct:.1f}%</span>',
        unsafe_allow_html=True,
    )


def _metric_row_label(m: dict) -> str:
    """One-line summary of a metric, used as its expander label.

    Carries everything the old summary table's row carried, so opening the row
    is the only step between seeing a score and reading why it came out that
    way. Expander labels take Markdown, including :red[] / :green[] colour.
    """
    name = m["metric_name"].strip()

    if not m.get("scored", True):
        # Every run in this group was a judge error or capture gate, so there
        # is no score or pass/fail status to show.
        return f"**{name}** &nbsp; `N/A` &nbsp; :gray[Not scored]"

    status = ":green[Passed]" if m["passed"] else ":red[Failed]"
    label = (
        f"**{name}** &nbsp; `{m['score']:.3f}` &nbsp; {status} "
        f"&nbsp; :gray[threshold {m['threshold']:.2f}]"
    )

    # Only worth the space when the runs actually disagreed.
    if "min_score" in m and m["min_score"] != m["max_score"]:
        label += f" &nbsp; :gray[runs {m['min_score']:.3f} to {m['max_score']:.3f}]"

    if m.get("not_scored_count"):
        label += f" &nbsp; :orange[{m['not_scored_count']} not scored]"

    return label


def _is_failure(m: dict) -> bool:
    """A scored metric that did not meet its threshold.

    Not-scored metrics are excluded: a judge error is not a failure.
    """
    return m.get("scored", True) and not m["passed"]


def _render_metric_rows(metrics: list[dict], failures_only: bool = False) -> None:
    """One expander per metric: the label is the summary row, the body is that
    metric's per-run detail.

    Failed metrics open by default, passed ones stay shut, so the reasons you
    need are on screen and the rest is one line each.
    """
    shown = [m for m in metrics if _is_failure(m)] if failures_only else metrics
    for m in shown:
        scored = m.get("scored", True)
        with st.expander(
            _metric_row_label(m),
            expanded=scored and not m["passed"],
        ):
            tooltip = METRIC_TOOLTIPS.get(m["metric_name"].strip(), "")
            if tooltip:
                st.caption(tooltip)
            if not scored:
                st.markdown(
                    ":gray[No run produced a quality verdict "
                    "(judge error or capture gate).]"
                )
            _render_metric_body(m)


def _render_metric_body(m: dict) -> None:
    """
    show individual raw eval results for one metric.
    aggregated metrics also show a per-run breakdown.
    """
    if "min_score" not in m:
        # Consistency - single aggregated result, no per-run breakdown
        _render_single_eval_result(m)
        return

    # Mean/threshold are already in the row label; only the run count adds
    # anything here, and only once there is more than one run.
    n = m.get("n_runs", len(m.get("raw_results", [])))
    if n > 1:
        st.caption(f"Mean of {n} runs")

    not_scored_reasons = m.get("not_scored_reasons") or []
    if not_scored_reasons:
        reasons_list = "; ".join(html.escape(reason) for reason in not_scored_reasons)
        st.markdown(
            f":orange[**{len(not_scored_reasons)} run(s) not scored** "
            f"(excluded from the mean above, not a quality verdict): "
            f"{reasons_list}]"
        )

    for idx, raw in enumerate(m.get("raw_results", []), 1):
        _render_single_eval_result(raw, run_label=f"Run {idx}" if n > 1 else None)


def _render_single_eval_result(r: dict, run_label: str | None = None) -> None:
    """one raw eval result entry."""
    prefix = f"{run_label}: " if run_label else ""

    if _is_scored(r):
        passed = r["passed"]
        colour = "#3fb950" if passed else "#f85149"
        label = "Passed" if passed else "Failed"
        score_text = f"{r['score']:.3f}"
    else:
        colour = "#8b949e"
        label = "N/A"
        score_text = "not scored"

    # Escaped, then newlines turned into <br>, so a multi-line reason (e.g.
    # Plan Coverage's per-statement breakdown) renders as line breaks rather
    # than one run-on line, without trusting judge/reason text as raw HTML.
    reason_html = html.escape(r.get("reason", "")).replace("\n", "<br>")
    st.markdown(
        f'<div style="background:#0d1117;border-left:3px solid {colour};'
        f'padding:10px 14px;border-radius:4px;margin:6px 0;">'
        f'<span style="color:#8b949e;font-size:0.8em;">'
        f'{prefix}{r.get("test_name","")}</span>&nbsp;&nbsp;'
        f'<span style="color:{colour};font-size:0.85em;font-weight:600;">{label}</span>'
        f"&nbsp;&nbsp;score: <code>{score_text}</code>"
        f'<div style="color:#8b949e;font-size:0.85em;margin-top:6px;">'
        f'{reason_html}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )

    meta_bits = []
    if r.get("run_at"):
        meta_bits.append(f"scored {r['run_at'][:19]}")
    if r.get("judge_llm"):
        meta_bits.append(f"judge: `{r['judge_llm']}`")
        if r.get("judge_tokens"):
            meta_bits.append(f"{r['judge_tokens']} tokens")
    if meta_bits:
        st.caption(" · ".join(meta_bits))

    tools = r.get("tools_used")
    if tools:
        st.caption(f"Tools used: {', '.join(tools)}")

    if r.get("error"):
        st.error(r["error"])


def _strip_worker_prefix(name: str) -> str:
    """Remove the 'Worker: ' prefix from tool names for display."""
    return name.removeprefix("Worker: ")


# chat_mode -> (icon, label) used for tab labels and the execution context badge.
_CHAT_MODE_DISPLAY = {
    "research": ("🔎", "research"),
    "deep_research": ("🧭", "deep_research"),
    "conversational": ("💬", "conversational"),
}


def _chat_mode_badge(chat_mode: str) -> str:
    icon, label = _CHAT_MODE_DISPLAY.get(chat_mode, ("❔", chat_mode or "unknown"))
    return f"{icon} {label}"


def _render_chat_interaction(records: list[dict]) -> None:
    """
    raw chat interaction(s) for an LLM/question pair.
    A row/record in responses.db is one run.
    """
    if not records:
        st.info("No response records found in responses.db for this combination.")
        return

    run_tabs = st.tabs(
        [
            f"Run {i + 1}  {_chat_mode_badge(r.get('chat_mode', 'research'))}  "
            f"({r['timestamp'][:19]})"
            for i, r in enumerate(records)
        ]
    )

    for tab, rec in zip(run_tabs, records):
        with tab:
            # --- Execution Metadata ---
            st.markdown("##### ⚙️ Execution Context")
            cols = st.columns(5)
            with cols[0]:
                chat_mode = rec.get("chat_mode", "research")
                st.markdown(f"**Research Type:** {_chat_mode_badge(chat_mode)}")
            with cols[1]:
                st.markdown(f"**Research Mode:** `{rec.get('research_mode', 'N/A')}`")
            with cols[2]:
                fallback = rec.get("fallback_used", False)
                st.markdown(f"**Fallback Used:** {'Yes' if fallback else 'No'}")
            with cols[3]:
                summarisation = rec.get("summarisation_used", False)
                summ_llm = rec.get("summarisation_llm", "")
                main_llm = rec.get("llm_name", "")
                if summarisation and summ_llm and summ_llm != main_llm:
                    st.markdown("**Summarisation:** Yes")
                    st.caption(f"Model: `{summ_llm}`")
                elif summarisation:
                    st.markdown("**Summarisation:** Yes *(main model)*")
                else:
                    st.markdown("**Summarisation:** No")
            with cols[4]:
                tool_seq = rec.get("tool_sequence") or []
                st.markdown(f"**Tool Sequence:** `{len(tool_seq)}` steps")
                if tool_seq:
                    display_seq = [_strip_worker_prefix(t) for t in tool_seq]
                    st.caption(" → ".join(display_seq))

            st.divider()

            # --- Deep Research Plan ---
            research_plan = rec.get("research_plan")
            if chat_mode == "deep_research" and research_plan:
                with st.expander("📋 Deep Research Plan (as presented to the user)"):
                    steps = (
                        research_plan.get("steps")
                        if isinstance(research_plan, dict)
                        else None
                    )
                    if steps:
                        for i, step in enumerate(steps):
                            if isinstance(step, dict):
                                title = step.get("title") or f"Step {i + 1}"
                                detail = step.get("brief") or step.get("description")
                                st.markdown(f"**{i + 1}. {title}**")
                                if detail:
                                    st.caption(detail)
                            else:
                                st.markdown(f"**{i + 1}.** {step}")
                    else:
                        st.json(research_plan, expanded=False)
                st.divider()

            # --- LLM Answer ---
            st.markdown("#### LLM Answer")
            actual = rec.get("actual_output", "")
            if actual:
                st.markdown(actual)
            else:
                st.caption("_(no output captured)_")

            st.divider()

            # --- Research Output ---
            research_output = rec.get("research_output", "")
            if research_output:
                st.markdown("#### 📝 Research Output (Worker Findings)")
                st.markdown(research_output)
                st.divider()

            # --- Summarisation Output ---
            summarisation_output: list = rec.get("summarisation_output") or []
            summarisation_used = rec.get("summarisation_used", False)
            if summarisation_output:
                st.markdown(
                    f"#### 🔍 Summarised Context ({len(summarisation_output)} passage(s))"
                )
                for i, summary_text in enumerate(summarisation_output):
                    with st.expander(f"Summarised Passage {i + 1}", expanded=i == 0):
                        st.markdown(summary_text)
                st.divider()
            elif summarisation_used:
                st.info("Summarisation was used but no output was captured.")
                st.divider()

            # --- Tools Called (sorted by tool_sequence start order) ---
            tools_called: list[dict] = [
                t
                for t in (rec.get("tools_called") or [])
                if t.get("name") != "Research Agent"
            ]
            if tools_called:
                # Sort tools_called by their position in tool_sequence
                tool_seq = rec.get("tool_sequence") or []
                order_map = {name: i for i, name in enumerate(tool_seq)}
                tools_called.sort(key=lambda t: order_map.get(t.get("name", ""), 999))

                st.markdown(f"#### Tools Called ({len(tools_called)})")
                for i, tool in enumerate(tools_called):
                    tool_name = tool.get("name", f"tool_{i}")
                    display_name = _strip_worker_prefix(tool_name)
                    is_lex_api = any(
                        k in tool_name
                        for k in (
                            "search_legislation",
                            "get_legislation_text",
                            "get_legislation",
                        )
                    )
                    st.markdown(f"🔧 **{display_name}**")
                    if is_lex_api:
                        params = (
                            tool.get("input_parameters")
                            or tool.get("inputParameters")
                            or {}
                        )
                        output_raw = tool.get("output", "")
                        req_col, _ = st.columns([3, 1])
                        with req_col:
                            method = params.get("method", "POST")
                            url = params.get("url", "")
                            payload = params.get("payload") or {}
                            st.markdown(
                                f'<div style="background:#0d1117;border-left:3px solid #58a6ff;'
                                f"padding:8px 12px;border-radius:4px;margin:4px 0 2px 0;"
                                f'font-size:0.85em;font-family:monospace;color:#58a6ff;">'
                                f"📡 {method} {url}</div>",
                                unsafe_allow_html=True,
                            )
                            if payload:
                                st.json(payload, expanded=True)
                        st.markdown(
                            '<div style="font-size:0.8em;color:#8b949e;margin:4px 0 2px 16px;">'
                            "↩ Response</div>",
                            unsafe_allow_html=True,
                        )
                        with st.container():
                            if isinstance(output_raw, str):
                                # Check for explicit "no results" fallback indicators
                                if output_raw.strip().lower() in (
                                    "done",
                                    "none",
                                    "null",
                                    "",
                                ):
                                    st.info(
                                        "⚠️ No results returned from this API call."
                                    )
                                else:
                                    try:
                                        parsed = json.loads(output_raw)
                                        if (
                                            isinstance(parsed, dict)
                                            and parsed.get("status") == "no_results"
                                        ):
                                            st.info(
                                                f"⚠️ {parsed.get('message', 'No results returned from this API call.')}"
                                            )
                                        else:
                                            st.json(parsed, expanded=False)
                                    except (json.JSONDecodeError, ValueError):
                                        st.code(output_raw, language="text")
                            elif (
                                isinstance(output_raw, dict)
                                and output_raw.get("status") == "no_results"
                            ):
                                st.info(
                                    f"⚠️ {output_raw.get('message', 'No results returned from this API call.')}"
                                )
                            elif isinstance(output_raw, (dict, list)):
                                st.json(output_raw, expanded=False)
                            else:
                                st.text(str(output_raw))
                    elif tool_name == "delegate_research":
                        params = (
                            tool.get("input_parameters")
                            or tool.get("inputParameters")
                            or {}
                        )
                        query = params.get("query", "")
                        if query:
                            st.markdown(
                                f'<div style="background:#0d1117;border-left:3px solid #d29922;'
                                f"padding:8px 12px;border-radius:4px;margin:4px 0 2px 0;"
                                f'font-size:0.85em;font-family:monospace;color:#d29922;">'
                                f"🎯 Manager asked: {html.escape(query)}</div>",
                                unsafe_allow_html=True,
                            )
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

            # --- Case Law Context ---
            case_law_ctx: list[dict] = rec.get("case_law_context") or []
            if case_law_ctx:
                st.markdown(f"#### ⚖️ Case Law Context ({len(case_law_ctx)} items)")
                for i, case_data in enumerate(case_law_ctx):
                    title = case_data.get("title", "Unknown Title")
                    ncn = case_data.get("ncn", "")
                    court = case_data.get("court", "")
                    date = case_data.get("date", "")
                    url = case_data.get("url", "")

                    meta_parts = [p for p in [ncn, court, date] if p]
                    meta_str = f" ({' | '.join(meta_parts)})" if meta_parts else ""
                    st.markdown(f"**{i + 1}. {title}{meta_str}**")
                    if url:
                        st.markdown(f"   🔗 [Link to judgment]({url})")
                st.divider()

            # --- Retrieved Context ---
            contexts: list[str] = rec.get("retrieval_context") or []
            if contexts:
                st.markdown(f"#### 📚 Retrieved Context ({len(contexts)} items)")
                for i, ctx in enumerate(contexts):
                    st.markdown(f"**Context {i + 1}**")
                    with st.container():
                        st.code(ctx, language="text")
            else:
                st.caption("No retrieval context captured.")

            st.divider()
            st.markdown("ℹ️ **Full Record Metadata**")
            st.json(
                {
                    "timestamp": rec.get("timestamp"),
                    "llm_name": rec.get("llm_name"),
                    "summarisation_llm": rec.get("summarisation_llm", ""),
                    "question_id": rec.get("question_id"),
                    "research_mode": rec.get("research_mode"),
                    "chat_mode": rec.get("chat_mode"),
                    "research_plan": rec.get("research_plan"),
                    "fallback_used": rec.get("fallback_used"),
                    "summarisation_used": rec.get("summarisation_used"),
                    "tool_sequence": rec.get("tool_sequence", []),
                },
                expanded=False,
            )


def _render_question_block(
    qid: int,
    question_text: str,
    metrics: list[dict],
    response_records: list[dict],
    chat_key: str,
    failures_only: bool = False,
) -> None:
    """Full block for one question within an LLM section."""
    scored_metrics = [m for m in metrics if m.get("scored", True)]
    n_pass = sum(1 for m in scored_metrics if m["passed"])
    n_total = len(scored_metrics)
    n_na = len(metrics) - n_total
    n_fail = sum(1 for m in metrics if _is_failure(m))

    count = f"{n_pass}/{n_total} passed" if n_total else "nothing scored"
    count = f":green[{count}]" if n_pass == n_total and n_total else f":red[{count}]"
    if n_na:
        count += f" &nbsp; :gray[{n_na} N/A]"

    # In failures-only mode the failing questions are the point, so open them
    # rather than making the reader click through to what they asked to see.
    with st.expander(
        f"**Q{qid}:** {question_text[:120]}{'…' if len(question_text) > 120 else ''}"
        f" &nbsp; {count}",
        expanded=failures_only and bool(n_fail),
    ):
        _render_metric_rows(metrics, failures_only=failures_only)

        # Loaded on demand. Streamlit runs an expander's body whether or not it
        # is open, so rendering every question's transcript on every rerun cost
        # about 1.6 of the 1.8 seconds each filter change used to take. The
        # toggle keeps its own state, so this builds only for a question the
        # reader actually opened.
        if st.toggle("💬 Chat interaction", key=chat_key):
            _render_chat_interaction(response_records)


_ALL_MODES = "All research types"


def _render_mode_filter(container, modes_present: list[str]) -> str:
    """Dropdown selecting which chat_mode to show. Returns the selection.

    Only rendered when the database holds more than one research type, since
    with one it would be a dropdown with a single choice.
    """
    if len(modes_present) < 2:
        return _ALL_MODES
    return container.selectbox(
        "Research type",
        [_ALL_MODES, *modes_present],
        index=0,
        format_func=lambda m: m
        if m == _ALL_MODES
        else _chat_mode_badge(m).replace("_", " "),
        help="Deep research and single-shot runs are scored and averaged "
        "separately, never blended into one number.",
    )


def _render_model_selector(container, llms: list[str]) -> str:
    """Dropdown selecting which model's detail to show.

    A selectbox rather than st.tabs because Streamlit renders every tab's
    contents on every rerun, so tabs made the page cost grow with the number of
    models evaluated even though only one is ever on screen.
    """
    return container.selectbox(
        "Model", llms, index=0, help="Worst pass rate first."
    )


_ALL_RESULTS = "All results"
_FAILURES_ONLY = "Failures only"


def _render_show_filter(container) -> str:
    """Dropdown selecting whether to show every metric or only the failures."""
    return container.selectbox(
        "Show",
        [_ALL_RESULTS, _FAILURES_ONLY],
        index=0,
        help="Failures only hides passing metrics and questions, and opens what "
        "is left, so the reasons are on screen without hunting.",
    )


def _render_mode_comparison(
    llm: str, keys: list[tuple[str, str]], hierarchy: dict, failures_only: bool
) -> None:
    """Metrics passed per question, one column per research type.

    Only shown when a model has been run under more than one research type,
    which is the case where the numbers are otherwise only comparable by
    flipping the filter and remembering what was there.
    """
    qids = sorted({q for k in keys for q in hierarchy[k]})
    rows = []
    for qid in qids:
        row = {"Question": f"Q{qid}"}
        for _llm, mode in keys:
            metrics = hierarchy[(llm, mode)].get(qid, [])
            scored = [m for m in metrics if m.get("scored", True)]
            n_fail = sum(1 for m in metrics if _is_failure(m))
            if not scored:
                row[mode] = "-"
            elif failures_only:
                row[mode] = str(n_fail)
            else:
                row[mode] = f"{len(scored) - n_fail}/{len(scored)}"
        rows.append(row)

    st.caption(
        "Failures per question" if failures_only else "Metrics passed per question"
    )
    st.dataframe(rows, hide_index=True, use_container_width=True)


def main() -> None:
    st.set_page_config(
        page_title="LexChat Eval",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.title("LexChat Evaluation")
    st.markdown("[LexChat](https://github.com/delphium226/lexchat) testing metric \
    results exploration. Explore LLM responses to a set of legal queries.   \
    Currently under development.")

    st.divider()

    _db_mtime = RESPONSES_DB.stat().st_mtime if RESPONSES_DB.exists() else 0.0

    if not RESPONSES_DB.exists():
        st.error(
            "No results found. Run evaluations first: python lex_eval/run_evals.py"
        )
        st.stop()

    raw_results = load_eval_results(_db_mtime=_db_mtime)
    if not raw_results:
        st.warning("No eval_<metric> tables have results yet - run evaluations first.")

    hierarchy = _build_hierarchy(raw_results)
    responses = load_responses(_mtime=_db_mtime) if RESPONSES_DB.exists() else {}

    if not responses:
        st.warning(
            f"responses.db not found at {RESPONSES_DB} - chat interaction will be empty."
        )

    st.markdown(
        """
    <style>
        .block-container { padding-top: 1.8 rem; }
    </style>
""",
        unsafe_allow_html=True,
    )

    modes_present = sorted({mode for _llm, mode in hierarchy})
    multiple_modes = len(modes_present) > 1

    # Every model and research type in the database, rendered above the filters
    # because the filters do not narrow it. They control the per question detail
    # below, and a control that changed something above it would not read that
    # way.
    _render_top_summary(hierarchy, multiple_modes)
    st.divider()

    model_col, mode_col, show_col = st.columns([2, 1, 1])

    # Models worst pass rate first, matching the summary above.
    llms = list(dict.fromkeys(llm for llm, _mode in _sorted_group_keys(hierarchy)))
    llm = _render_model_selector(model_col, llms)
    selected_mode = _render_mode_filter(mode_col, modes_present)
    failures_only = _render_show_filter(show_col) == _FAILURES_ONLY

    # Model and research type are independent axes, so "All research types" for a
    # model that has several shows them one after another rather than picking one.
    keys = [
        k
        for k in _sorted_group_keys(hierarchy)
        if k[0] == llm and (selected_mode == _ALL_MODES or k[1] == selected_mode)
    ]
    if not keys:
        st.info(f"No {selected_mode} results for {llm}.")
        return

    if len(keys) > 1:
        _render_mode_comparison(llm, keys, hierarchy, failures_only)
        st.markdown("")

    for key in keys:
        _llm, mode = key
        q_data = hierarchy[key]
        st.subheader(_group_label(key, show_mode=multiple_modes))
        _render_llm_summary_bar(q_data)
        st.markdown("")

        shown_any = False
        for qid in sorted(q_data.keys()):
            metrics = q_data[qid]
            if failures_only and not any(_is_failure(m) for m in metrics):
                continue
            shown_any = True
            _render_question_block(
                qid,
                metrics[0].get("question", ""),
                metrics,
                responses.get((llm, mode, qid), []),
                chat_key=f"chat::{llm}::{mode}::{qid}",
                failures_only=failures_only,
            )
        if not shown_any:
            st.success("No failures. Every scored metric passed for this model.")


if __name__ == "__main__":
    main()
