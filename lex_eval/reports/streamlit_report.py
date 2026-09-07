from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import streamlit as st

# Ensure the repo root is importable when Streamlit launches this file directly.
_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from lex_eval.reference.store import (
    ANSWERS_DIR,
    MANIFEST_NAME,
    effective_verified,
    load_reference_answers,
    reference_version,
)
from lex_eval.reports.attribution import caveat, worst_attribution
from lex_eval.reports.comparison import compare
from lex_eval.reports.diagnostics import searches, search_summary, plan_steps
from lex_eval.reports.data import (
    apply_scope,
    question_metadata,
    current_reference_rows,
    aggregate_metrics,
    read_database,
    latest_results,
    outcome,
    outcome_counts,
    question_groups,
)
from lex_eval.utils.db import (
    DEFAULT_DB,
    reason_is_not_measured,
)

script_dir = Path(__file__).parent
data_dir = script_dir.parent / "data"

RESPONSES_DB = DEFAULT_DB


DEFAULT_CHAT_MODE = "research"


@st.cache_data
def load_dashboard(db_path: str, mtime: float):
    """Cache by path and modification time; never migrate the source database."""
    return read_database(Path(db_path), METRICS)


# Single source of truth for every metric this dashboard displays: its
# eval_<key> table, its display name, and its tooltip, in display order.
# Keys here must match run_evals.py::METRIC_FILES.
#
# Grouped, in this order: 1) deterministic metrics that run in research or
# deep research mode, 2) AI-judge metrics that run in research or deep
# research mode, 3) deep-research-only deterministic metrics, 4)
# deep-research-only AI-judge metrics. The "(Deep research only)" suffix on
# groups 3-4's display names is what shows that scope on the title bar
# wherever the metric name is rendered.
METRICS: list[tuple[str, str, str]] = [
    # 1. Deterministic, research or deep research
    (
        "tool_usage",
        "Tool Usage",
        "Are all of delegate research, search legislation and search legislation sections used, in the correct order (search legislation - search legislation sections - get legislation text if needed), and does the Worker stick to that order rather than looping back to an earlier step later in the same run?",
    ),
    (
        "mandatory_structure",
        "Research Output Structure",
        "Does the worker agent return the findings to the manager with the requested headers. Not measured in conversational mode, where the worker is told not to use those headers.",
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
        "citation_read",
        "Citation Read",
        "Did the researcher actually read every Act it cites? An Act whose text was pulled counts as read, one that only appeared as a title in a search results list does not. Catches a report making claims about a real, correctly linked source it never opened.",
    ),
    (
        "citation_domain",
        "Citation Domain",
        "Does every citation link in the researcher's report point to a domain the Worker is permitted to cite. That is legislation.gov.uk for legislation only mode, caselaw.nationalarchives.gov.uk for case law only mode, and both for the hybrid mode, matching what each Worker prompt asks for.",
    ),
    (
        "genuine_gap",
        "Genuine Gap",
        "When retrieval found no usable legislation text, does the researcher's report say so plainly instead of answering with unsupported confidence. In research mode the wording is set by the prompt, so a paraphrase scores half. In conversational mode no wording is mandated, so a plain statement of the gap scores full.",
    ),
    (
        "consistency",
        "Consistency (Cosine)",
        "Compare the answers provided when the same question is asked multiple times in the same chat mode, using TF cosine similarity. Research and deep research answers are never compared against each other, and a mode with only one stored run is not scored. Any legislation section cited in one answer but not the other is listed in the detail, but does not decide pass or fail.",
    ),
    (
        "citation_agreement",
        "Citation Agreement",
        "Of the legislation provisions the hand written reference answer cites, how many does the response cite too. No AI judge, it compares the two lists of legislation.gov.uk links. For each Act the reference answer relied on that the response does not cite, the reason says whether any search turned it up, so a search that missed the law reads differently from an answer that had the law and left it out.",
    ),
    # 2. AI judge, research or deep research
    (
        "reference_answer_agreement",
        "Reference Answer Agreement",
        "AI as a judge metric: How many of the question's key statements the response also makes, at most 5 of them. The statements are written once alongside the hand written reference answer and stored with it, so the judge labels a fixed list rather than picking the points afresh on every run. A second judge call looks for contradictions and nothing else, which is what catches a long answer that makes a point correctly in one section and then undoes it in another. A statement the response contradicts fails the metric outright, since a confidently wrong statement of law is worse than a missing one. The reference answers are unverified drafts, so read a flagged contradiction as a prompt to compare the two texts.",
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
    # 3. Deterministic, deep research only
    (
        "step_completion",
        "Step Completion (Deep research only)",
        "Deep research only. Did every step of the approved research plan carry its own retrieved legal text into its own report, rather than a step that retrieved text and then reported nothing (for example, hitting a tool-call budget limit mid-step).",
    ),
    # 4. AI judge, deep research only
    (
        "report_integration",
        "Report Integration (Deep research only)",
        "Deep research only, AI as a judge metric: For every step that reported a real, cited finding of its own, does the final answer reflect that finding, rather than dropping it when the Manager condenses several step reports into one response. A step with nothing of its own to check (empty or uncited retrieval) is not scored here; that is Step Completion's and Genuine Gap's question.",
    ),
    (
        "plan_coverage",
        "Plan Coverage (Deep research only)",
        "Deep research only, AI as a judge metric: Does the approved research plan set out to cover the question's key statements, before any research happens. Reuses the same fixed statement list as Reference Answer Agreement rather than a separately authored golden plan.",
    ),
]

# order shown in streamlit
METRIC_DISPLAY_ORDER: list[str] = [name for _key, name, _tooltip in METRICS]

# hover over tips on app summary tables
METRIC_TOOLTIPS: dict[str, str] = {name: tip for _key, name, tip in METRICS}


# What puts a question in the "needs attention" set. Written once because
# the summary column and the checkbox that filters on it are the same rule.
NEEDS_ATTENTION_HELP = (
    "A question needs attention when a metric failed or could not be scored, "
    "when no metric ran at all, or when a run ended without an answer or hit "
    "the turn limit."
)

# Hover over help for the column headers of every table on the dashboard, one
# dict per table. They are kept separate because the same header means
# different things in different tables: "Error" counts failed attempts in the
# outcomes table and holds one tool call's error message in the searches table.
OUTCOME_COLUMNS: dict[str, str] = {
    "Attempts": "Every captured run of this question, including repeats and runs that failed.",
    "Answer": "Runs that returned an answer.",
    "Clarification": "Runs where LexChat asked the user a clarifying question instead of researching.",
    "Error": "Runs that failed with an error and produced no answer.",
    "No answer": "Runs that ended with no error, no clarifying question and no answer text.",
    "Turn-cap flags": "Runs where at least one research step was cut short at the server's turn limit, so that step returned no findings.",
    "Reformatted": "Runs where the worker's report missed the required headings and LexChat asked the model to rewrite it once.",
}

QUESTION_COLUMNS: dict[str, str] = {
    "Question": "The question id and the start of the question text.",
    "Model": "The model LexChat had active when these runs were gathered.",
    "Chat mode": "The LexChat mode used: research, deep research or conversational.",
    "Research mode": "Which sources the question expects: legislation only, case law only, or both.",
    "Experiment": "The label of the gather run these attempts belong to. Only runs in the same experiment are compared.",
    "Type": "From the question set. A regression question reproduces a past failure, a positive control is one LexChat is expected to get right.",
    "Reference agreement": "Reference Answer Agreement only: how many runs matched the reference answer's key statements, out of the runs it could score.",
    "Attempts": OUTCOME_COLUMNS["Attempts"],
    "Answers": OUTCOME_COLUMNS["Answer"],
    "Clarifications": OUTCOME_COLUMNS["Clarification"],
    "Errors": OUTCOME_COLUMNS["Error"],
    "Turn-cap flags": OUTCOME_COLUMNS["Turn-cap flags"],
    "Needs attention": NEEDS_ATTENTION_HELP,
}

SEARCH_COLUMNS: dict[str, str] = {
    "Response": "Which attempt made this search. One question can have several.",
    "Step": "Which research step made the call. Deep research numbers its steps, the other modes have one.",
    "Tool": "The search tool called, for example search_legislation.",
    "Arguments": "The arguments the tool was called with, including the search terms.",
    "Outcome": "Results returned, Empty, Error, or Unknown / incomplete when the result was cut short or blocked by the tool call budget.",
    "Returned": "How many items came back. This counts items, not relevance, and is blank when the count is unknown.",
    "Cache reused": "True when the result came from a cache rather than a fresh API call.",
    "Error": "The error the tool call returned, if any.",
    "Later nonempty search in this step": "True when this search returned nothing but a repeat of the same tool later in the step did return results.",
}

SEARCH_SUMMARY_COLUMNS: dict[str, str] = {
    "Response": SEARCH_COLUMNS["Response"],
    "Tool": SEARCH_COLUMNS["Tool"],
    "Outcome": SEARCH_COLUMNS["Outcome"],
    "Searches": "How many calls to this tool on this response ended this way.",
    "Items returned": "Total items those calls returned. Blank where no count is available, which is every Error and Unknown / incomplete row.",
}

COMPARISON_SUMMARY_COLUMNS: dict[str, str] = {
    "Matched questions and modes": "How many question and mode combinations appear in both experiments. Only these are compared.",
    "Baseline only": "Combinations gathered in the baseline experiment but not the candidate.",
    "Candidate only": "Combinations gathered in the candidate experiment but not the baseline.",
}

COMPARISON_OUTCOME_COLUMNS: dict[str, str] = {
    "Experiment": "Which side of the comparison the counts on this row come from.",
    **OUTCOME_COLUMNS,
}

COMPARISON_CHANGE_COLUMNS: dict[str, str] = {
    "Question": "The question id.",
    "Question text": "The question as it was asked. Both experiments used this exact wording.",
    "Chat mode": QUESTION_COLUMNS["Chat mode"],
    "Research mode": QUESTION_COLUMNS["Research mode"],
    "Metric": "The metric this row compares.",
    "Baseline": "Passes out of measured runs in the baseline experiment, and how many of its runs the metric could not measure.",
    "Candidate": "Passes out of measured runs in the candidate experiment, and how many of its runs the metric could not measure.",
    "Baseline responses": "The response ids scored on the baseline side.",
    "Candidate responses": "The response ids scored on the candidate side.",
    "Change": "How the candidate compares with the baseline, or why the two cannot be compared.",
}


def _column_help(tooltips: dict[str, str]) -> dict:
    """Turn a column name to description mapping into Streamlit column config."""
    return {
        name: st.column_config.Column(name, help=text)
        for name, text in tooltips.items()
    }


def _is_scored(result: dict) -> bool:
    """Whether this row's score is a verdict, and so belongs in a mean.

    A metric that could not score a response still writes a row, so the gap
    stays visible, carrying score 0.0 because the column is NOT NULL. The
    `measured` column is what marks those. Rows written before that column
    existed fall back to the reason wording, which is the rule the column was
    filled from (see db.backfill_measured_column), so a stale deploy.db still
    aggregates correctly.
    """
    if "measured" in result:
        return bool(result["measured"])
    return not reason_is_not_measured(result.get("reason") or "")


def _metric_sort_key(metric: dict) -> int:
    name = metric["metric_name"]
    try:
        return METRIC_DISPLAY_ORDER.index(name)
    except ValueError:
        return len(METRIC_DISPLAY_ORDER)


def _aggregate_metrics(results: list[dict]) -> list[dict]:
    return sorted(aggregate_metrics(results), key=_metric_sort_key)


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
        return f"**{name}** &nbsp; `N/A` &nbsp; :gray[{m.get('state', 'Not measured')}]"

    status = f"{m['pass_count']}/{m['measured_count']} measured passes, {m['state']}"
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
        st.caption(
            f"{m['measured_count']} measured responses; one selected score per response"
        )

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
    reason_html = html.escape(r.get("reason") or "").replace("\n", "<br>")
    st.markdown(
        f'<div style="background:#0d1117;border-left:3px solid {colour};'
        f'padding:10px 14px;border-radius:4px;margin:6px 0;">'
        f'<span style="color:#8b949e;font-size:0.8em;">'
        f'{prefix}{r.get("test_name","")}</span>&nbsp;&nbsp;'
        f'<span style="color:{colour};font-size:0.85em;font-weight:600;">{label}</span>'
        f"&nbsp;&nbsp;score: <code>{score_text}</code>"
        f'<div style="color:#8b949e;font-size:0.85em;margin-top:6px;">'
        f"{reason_html}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    if r.get("scope_note"):
        st.caption(r["scope_note"])
    if r.get("details"):
        st.json(r["details"], expanded=False)
    if r.get("reference_sha256"):
        refs = r.get("scoring_config", {}).get("references", {})
        reference = refs.get(str(r["question_id"])) or refs.get(r["question_id"])
        if reference:
            with st.expander("Reference used for this score"):
                st.caption(
                    f"Reference status: {r.get('reference_mode', 'unknown')}; fingerprint: {r['reference_sha256']}"
                )
                for statement in reference.get("statements", []):
                    st.write(statement)
    meta_bits = [f"response {r.get('response_id', 'unknown')}"]
    if r.get("scoring_run_id"):
        meta_bits.append(f"scoring run {r['scoring_run_id']}")
        meta_bits.append(f"code {r.get('metric_version', 'unknown')[:12]}")
    else:
        meta_bits.append("legacy scorer version unknown")
    if not _reference_is_current(r):
        meta_bits.append("reference answer changed since, re-run to update")
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


def _log_section(title: str) -> None:
    """A label for one of the log's own sections.

    Deliberately unlike a Markdown heading: a captured answer carries its own
    # and ## headings, which render larger than anything this page could write,
    so a heading here would sit below the content it introduces.
    """
    st.markdown(
        f'<div style="border-left:3px solid #58a6ff;padding:2px 10px;'
        f"margin:18px 0 6px 0;font-size:0.75em;font-weight:700;"
        f'letter-spacing:0.08em;text-transform:uppercase;color:#58a6ff;">'
        f"{html.escape(title)}</div>",
        unsafe_allow_html=True,
    )


def _captured_text(text: str, height: int = 420) -> None:
    """Model output in a bordered box that scrolls once it is taller than the box.

    The border marks where captured text starts and stops, and the fixed height
    keeps the next section on screen instead of pages below.
    """
    with st.container(border=True, height=height):
        st.markdown(text)


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

    for tab, rec in zip(run_tabs, records, strict=True):
        with tab:
            st.caption(f"Response {rec['response_id']} · {outcome(rec)}")
            if rec.get("needs_clarification"):
                st.info(rec.get("clarification_question") or "Clarification requested")
            if rec.get("is_error"):
                st.error(rec.get("error_message") or "Request failed")
            if rec.get("max_turns_halted"):
                st.warning("A research step reached the tool-call limit.")
            st.caption(
                f"Reformatted: {bool(rec.get('reformatted'))}; request attempts: {rec.get('attempts', 'unknown')}"
            )
            # --- Execution Metadata ---
            _log_section("Execution context")
            cols = st.columns(5)
            with cols[0]:
                chat_mode = rec.get("chat_mode", "research")
                st.markdown(f"**Chat mode:** {_chat_mode_badge(chat_mode)}")
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
                    # The opening of the sequence only. Printing all of a long
                    # one here fills the screen in a narrow column, and the
                    # Tools called section below lists every call in order.
                    display_seq = [_strip_worker_prefix(t) for t in tool_seq[:6]]
                    preview = " → ".join(display_seq)
                    if len(tool_seq) > 6:
                        preview += f" → and {len(tool_seq) - 6} more"
                    st.caption(preview)

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
                                detail = (
                                    step.get("detail")
                                    or step.get("brief")
                                    or step.get("description")
                                )
                                st.markdown(f"**{i + 1}. {title}**")
                                if detail:
                                    st.caption(detail)
                            else:
                                st.markdown(f"**{i + 1}.** {step}")
                    else:
                        st.json(research_plan, expanded=False)

            # --- LLM Answer ---
            _log_section("Answer to the user")
            actual = rec.get("actual_output", "")
            if actual:
                _captured_text(actual)
            else:
                st.caption("_(no output captured)_")

            # --- Research Output ---
            research_output = rec.get("research_output", "")
            if research_output:
                _log_section("Research output (worker findings)")
                _captured_text(research_output)

            # --- Summarisation Output ---
            summarisation_output: list = rec.get("summarisation_output") or []
            summarisation_used = rec.get("summarisation_used", False)
            if summarisation_output:
                _log_section(
                    f"Summarised context ({len(summarisation_output)} passages)"
                )
                # A long list of identical rows fills screens; one scrolling
                # box keeps the section after it within reach.
                with st.container(height=320):
                    for i, summary_text in enumerate(summarisation_output):
                        with st.expander(
                            f"Summarised Passage {i + 1}", expanded=i == 0
                        ):
                            st.markdown(summary_text)
            elif summarisation_used:
                st.info("Summarisation was used but no output was captured.")

            # --- Tools Called (sorted by tool_sequence start order) ---
            tools_called: list[dict] = [
                t
                for t in (rec.get("tools_called") or [])
                if t.get("name") != "Research Agent"
            ]
            if tools_called:
                _log_section(f"Tools called ({len(tools_called)})")
                st.caption(
                    "Captured order. The step view above preserves each delegation boundary."
                )
                # A long list of identical rows fills screens; one scrolling
                # box keeps the section after it within reach.
                with st.container(height=360):
                    for index, tool in enumerate(tools_called, 1):
                        name = _strip_worker_prefix(tool.get("name", "unknown"))
                        with st.expander(f"{index}. {name}"):
                            params = (
                                tool.get("input_parameters")
                                or tool.get("inputParameters")
                                or {}
                            )
                            st.caption("Tool arguments")
                            st.json(params, expanded=True)
                            output = tool.get("output")
                            if isinstance(output, str):
                                try:
                                    output = json.loads(output)
                                except (ValueError, TypeError):
                                    pass
                            if isinstance(output, (dict, list)):
                                st.json(output, expanded=False)
                            elif output:
                                st.code(str(output), language="text")
                            else:
                                st.caption(
                                    "No output captured; this is not proof of an empty search."
                                )
            else:
                st.caption("No tool calls captured.")

            # --- Case Law Context ---
            case_law_ctx: list[dict] = rec.get("case_law_context") or []
            if case_law_ctx:
                _log_section(f"Case law context ({len(case_law_ctx)} items)")
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

            # --- Retrieved Context ---
            contexts: list[str] = rec.get("retrieval_context") or []
            if contexts:
                _log_section(f"Retrieved context ({len(contexts)} items)")
                # A long list of identical rows fills screens; one scrolling
                # box keeps the section after it within reach.
                with st.container(height=360):
                    for i, ctx in enumerate(contexts):
                        st.markdown(f"**Context {i + 1}**")
                        st.code(ctx, language="text")
            else:
                st.caption("No retrieval context captured.")

            _log_section("Full record metadata")
            st.json(
                {
                    "response_id": rec.get("response_id"),
                    "experiment_id": rec.get("experiment_id"),
                    "gather_run_id": rec.get("gather_run_id"),
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


# Colour and wording for the attribution flag, one entry per verdict from
# reports/attribution.py. Grey for "cannot say", which is a gap in the
# reference answers rather than a finding about the response.
_ATTRIBUTION_STYLE = {
    "tech": ("#f85149", "Run did not finish"),
    "search": ("#f0ad4e", "The search did not find the law"),
    "model": ("#58a6ff", "Found the law, did not cite it"),
    # Not the pass green used elsewhere: this says no step lost any law, which
    # is narrower than the response being good, and the detail says so.
    "no_law_lost": ("#56d364", "No law lost"),
    "not_attributable": ("#8b949e", "Cannot say"),
}


@st.cache_data
def _reference_answers(mtime: float = 0.0) -> dict:
    """Reference answers keyed by question_id.

    Signed and unsigned, matching tests/eval/test_reference.py: excluding
    drafts would leave every question unattributable until sign off.

    ``_mtime`` busts the cache when the manifest is rewritten, the same way
    ``load_eval_results`` and ``load_responses`` do for the database.
    """
    return load_reference_answers()


def _reference_is_current(r: dict) -> bool:
    """Whether a stored metric row was scored against the reference in use now.

    A row scored against an answer that has since been corrected, or against a
    draft since signed off, is not comparable with the attribution shown beside
    it, so it is labelled rather than quietly averaged in.
    """
    if not r.get("reference_sha256"):
        return True
    reference = _reference_answers(_reference_manifest_mtime()).get(r["question_id"])
    if not reference:
        return False
    return reference_version(reference) == (
        r["reference_sha256"],
        r.get("reference_mode"),
    )


def _reference_manifest_mtime() -> float:
    path = ANSWERS_DIR / MANIFEST_NAME
    return path.stat().st_mtime if path.exists() else 0.0


def _render_attribution_flag(response_records: list[dict]) -> None:
    """One line saying which step lost the law, for one response.

    Always rendered when there are records to judge. An absent flag used to
    mean "no step lost any law" and was read as "nothing to report", so that
    case now has its own label rather than being silence. No score is shown
    here; the question's metric rows above hold those.
    """
    verdict = worst_attribution(
        response_records, _reference_answers(_reference_manifest_mtime())
    )
    if verdict is None:
        return

    colour, label = _ATTRIBUTION_STYLE.get(verdict["stage"], ("#8b949e", "Unclear"))
    ids = ", ".join(verdict["ids"][:4])
    if len(verdict["ids"]) > 4:
        ids += f" and {len(verdict['ids']) - 4} more"
    detail = html.escape(verdict["detail"] + (f": {ids}" if ids else ""))
    st.markdown(
        f'<div style="border-left:3px solid {colour};padding:4px 10px;'
        f'margin:0 0 10px 0;font-size:0.9em;">'
        f'<span style="color:{colour};font-weight:600;">{label}</span>'
        f'<span style="color:#8b949e;"> &nbsp; {detail}</span></div>',
        unsafe_allow_html=True,
    )
    note = caveat(verdict["stage"])
    if note:
        st.caption(note)


def _reference_status_line() -> str:
    """How many reference answers a lawyer has signed off, and how many are drafts.

    The three reference metrics score against both, so the reader needs to know
    how much of what they are looking at is measured against lawyer-approved
    law and how much against a draft.
    """
    references = _reference_answers(_reference_manifest_mtime())
    if not references:
        return "No reference answers yet, so the reference metrics score nothing."
    verified = sum(1 for r in references.values() if effective_verified(r))
    drafts = len(references) - verified
    return (
        f"Reference answers in use: {verified} signed off by a lawyer, "
        f"{drafts} unverified draft(s). A draft's scores measure agreement with "
        "its author, not legal correctness."
    )


def _render_outcomes(records):
    counts = outcome_counts(records)
    st.dataframe(
        [counts],
        hide_index=True,
        width="stretch",
        column_config=_column_help(OUTCOME_COLUMNS),
    )
    st.caption(
        "Outcomes count all attempts. Turn-cap and reformat flags may overlap with answers or errors."
    )


STAGES = {
    "Answer coverage": {"reference_answer_agreement", "citation_agreement"},
    "Planning and search": {"plan_coverage"},
    "Worker evidence": {
        "claim_support",
        "citation_grounding",
        "citation_read",
        "citation_domain",
        "genuine_gap",
        "tool_usage",
        "step_completion",
    },
    "Final synthesis": {
        "response_groundedness",
        "report_integration",
        "citation_passthrough",
        "mandatory_structure",
    },
    "Repeat similarity": {"consistency"},
}


def _question_summary(key, records, rows):
    qid, model, chat, research, question, experiment = key
    ids = {r["response_id"] for r in records}
    metrics = _aggregate_metrics([r for r in rows if r["response_id"] in ids])
    agreement = next(
        (m for m in metrics if m["test_name"] == "reference_answer_agreement"), None
    )
    counts = outcome_counts(records)
    metadata = question_metadata(records[0])
    return {
        "Question": f"Q{qid}: {question[:95]}",
        "Model": model,
        "Chat mode": chat,
        "Research mode": research,
        "Experiment": records[0].get("experiment", {}).get("label", experiment),
        "Type": metadata.get("test_type", ""),
        "Reference agreement": (
            f"{agreement['pass_count']}/{agreement['measured_count']} measured passes"
            if agreement
            else "Not scored"
        ),
        "Attempts": counts["Attempts"],
        "Answers": counts["Answer"],
        "Clarifications": counts["Clarification"],
        "Errors": counts["Error"],
        "Turn-cap flags": counts["Turn-cap flags"],
        "Needs attention": any(
            _is_failure(m) or m.get("not_scored_count") for m in metrics
        )
        or any(outcome(r) != "Answer" or r.get("max_turns_halted") for r in records)
        or not metrics,
    }


def _question_detail(key, group, rows):
    st.subheader(f"Q{key[0]}: {key[4]}")
    metadata = question_metadata(group[0])
    st.caption(metadata["metadata_source"])
    if metadata.get("eval_observation"):
        st.caption(str(metadata["eval_observation"]))
    _render_outcomes(group)
    # Above the known failure and the metric rows: a reviewer checking whether a
    # recorded failure came back needs the answer beside the description of it,
    # not several screens below. Both start closed so the page still opens on
    # the scores.
    with st.expander("Response to user"):
        # One panel per run, all closed when there are several: an answer runs
        # to pages, and the point of this section is comparing the runs, not
        # scrolling through the first to reach the second.
        for index, rec in enumerate(group, 1):
            with st.expander(
                f"Run {index} · Response {rec['response_id']} · {outcome(rec)}"
                f" · {rec['timestamp'][:19]}",
                expanded=len(group) == 1,
            ):
                if rec.get("needs_clarification"):
                    st.info(
                        rec.get("clarification_question") or "Clarification requested"
                    )
                if rec.get("is_error"):
                    st.error(rec.get("error_message") or "Request failed")
                _captured_text(
                    rec.get("actual_output") or "_(no output captured)_", height=500
                )
    with st.expander("Full research log"):
        _render_chat_interaction(group)
    ids = {r["response_id"] for r in group}
    selected_rows = [r for r in rows if r["response_id"] in ids]
    metrics = _aggregate_metrics(selected_rows)
    # Same heading level as the metric groups below, and before them: what this
    # question was recorded as failing is what those scores are checking for.
    if metadata.get("known_gap"):
        st.markdown("#### Previous known failure check")
        st.write(metadata["known_gap"])
    for title, keys in STAGES.items():
        subset = [m for m in metrics if m["test_name"] in keys]
        if subset:
            st.markdown(f"#### {title}")
            _render_metric_rows(subset)
    if key[3] != "case_law_only":
        with st.expander(
            "Did this run find the legislation the reference answer relies on?"
        ):
            st.caption(
                "For each Act the reference answer cites, this says where that Act was "
                "lost: no search in the run found it, or a search found it and the "
                "answer did not cite it. It reads the run's own stored tool calls and "
                "answer, against the reference answer as it stands now, not the version "
                "an older stored score was measured against. Acts only, so it says "
                "nothing about sections, judgments, or whether the answer is legally "
                "correct."
            )
            for rec in group:
                st.caption(f"Response {rec['response_id']}")
                _render_attribution_flag([rec])
    search_rows = [r for rec in group for r in searches(rec)]
    with st.expander("Searches and deep-research steps"):
        st.caption(
            "Counts describe returned items, not relevance. A later nonempty search is evidence of further results, not proof that the question was resolved."
        )
        if search_rows:
            st.dataframe(
                search_summary(search_rows),
                hide_index=True,
                width="stretch",
                column_config=_column_help(SEARCH_SUMMARY_COLUMNS),
            )
            with st.expander(f"Every search call ({len(search_rows)})"):
                st.dataframe(
                    search_rows,
                    hide_index=True,
                    width="stretch",
                    column_config=_column_help(SEARCH_COLUMNS),
                )
        else:
            st.info("No captured search facts for these attempts.")
        for rec in group:
            steps = plan_steps(rec)
            approved = len((rec.get("research_plan") or {}).get("steps", []))
            st.caption(
                f"Response {rec['response_id']}: {approved} approved steps, {len(steps)} captured delegations"
            )
            for step in steps:
                with st.expander(
                    f"Response {rec['response_id']}, step {step['Step']}: {step['Title']}"
                ):
                    st.caption(
                        f"Tools: {step['Tools']}; reformatted: {step['Reformatted']}; error: {step['Error'] or 'none'}"
                    )
                    st.markdown(step["Report"] or "No report")
    with st.expander("Export review evidence"):
        st.caption(
            "Includes answers, Worker reports, selected verdicts, question metadata and current reference statements. Draft agreement is not legal correctness."
        )
        reference = _reference_answers(_reference_manifest_mtime()).get(key[0], {})
        pack = {
            "question": metadata,
            "current_reference_statements": reference.get("statements", []),
            "current_reference_sha256": reference.get("reference_sha256"),
            "scored_reference_snapshots": {
                r["reference_sha256"]: r.get("scoring_config", {})
                .get("references", {})
                .get(str(key[0]))
                for r in selected_rows
                if r.get("reference_sha256") and r.get("scoring_config")
            },
            "responses": [
                {
                    k: r.get(k)
                    for k in (
                        "response_id",
                        "question",
                        "actual_output",
                        "research_output",
                        "experiment_id",
                        "gather_run_id",
                        "needs_clarification",
                        "error_message",
                    )
                }
                for r in group
            ],
            "metrics": [
                {k: v for k, v in r.items() if k != "scoring_config"}
                for r in selected_rows
            ],
            "searches": search_rows,
            "review": {
                "observed_problem": "",
                "evidence_passage": "",
                "metric_caught_it": None,
                "reviewer": "",
            },
        }
        st.download_button(
            "Download review pack",
            json.dumps(pack, ensure_ascii=False, indent=2),
            file_name=f"q{key[0]}_review.json",
            mime="application/json",
        )


def _compare_experiments(records, rows):
    experiments = {
        r["experiment_id"]: r.get("experiment", {})
        for r in records
        if r.get("experiment_id")
    }
    if len(experiments) < 2:
        st.info(
            "Comparison needs two recorded experiments. Legacy responses remain available in the question review; dates alone do not establish matching conditions."
        )
        return
    options = sorted(experiments)
    label = lambda value: f"{experiments[value].get('label', value)} ({value[:8]})"
    left, right = st.columns(2)
    baseline = left.selectbox("Baseline experiment", options, format_func=label)
    candidate = right.selectbox(
        "Candidate experiment", [e for e in options if e != baseline], format_func=label
    )
    sides = [
        [r for r in records if r.get("experiment_id") == exp]
        for exp in (baseline, candidate)
    ]
    st.caption(
        "Matched question wording, snapshot and modes only. The latest shared scoring version is selected per metric; missing measurements remain visible. Two repeats describe observations, not statistical certainty."
    )
    summary, changes, outcomes = compare(*sides, apply_scope(rows))
    st.dataframe(
        [summary],
        hide_index=True,
        width="stretch",
        column_config=_column_help(COMPARISON_SUMMARY_COLUMNS),
    )
    st.dataframe(
        outcomes,
        hide_index=True,
        width="stretch",
        column_config=_column_help(COMPARISON_OUTCOME_COLUMNS),
    )
    if changes:
        st.dataframe(
            changes,
            hide_index=True,
            width="stretch",
            column_config=_column_help(COMPARISON_CHANGE_COLUMNS),
        )
    else:
        st.info("No matched metric results to compare.")
    with st.expander("Experiment conditions"):
        for exp in (baseline, candidate):
            st.write(label(exp))
            st.json(
                {
                    k: v
                    for k, v in experiments[exp].get("config", {}).items()
                    if k != "questions"
                }
            )
    matched_ids = {r["response_id"] for side in sides for r in side}
    options = [r for r in records if r["response_id"] in matched_ids]
    selected = st.selectbox(
        "Inspect a response",
        options,
        format_func=lambda r: f"Response {r['response_id']}: Q{r['question_id']} ({r.get('experiment', {}).get('label', '')})",
    )
    _render_chat_interaction([selected])


def main() -> None:
    st.set_page_config(page_title="LexChat Eval", layout="wide")
    st.title("LexChat evaluation")
    st.caption(_reference_status_line())
    db_path = Path(st.text_input("Results database", str(RESPONSES_DB)))
    if not db_path.exists():
        st.info("No database at this path.")
        return
    records, rows = load_dashboard(str(db_path), db_path.stat().st_mtime_ns)
    if not records:
        st.info("No response attempts in this database.")
        return
    view = st.radio(
        "View", ["Review questions", "Compare experiments"], horizontal=True
    )
    cols = st.columns(3)
    for col, field, label in zip(
        cols,
        ("llm_name", "chat_mode", "research_mode"),
        ("Model", "Chat mode", "Research mode"),
        strict=True,
    ):
        options = sorted({r.get(field) or "unknown" for r in records})
        selected = col.selectbox(label, ["All", *options])
        if selected != "All":
            records = [r for r in records if (r.get(field) or "unknown") == selected]
    selected_questions = st.multiselect(
        "Questions",
        sorted({r["question_id"] for r in records}),
        format_func=lambda q: f"Q{q}",
    )
    if selected_questions:
        records = [r for r in records if r["question_id"] in selected_questions]
    ids = {r["response_id"] for r in records}
    rows = [r for r in rows if r["response_id"] in ids]
    if view == "Compare experiments":
        _compare_experiments(records, rows)
        return
    experiments = {
        r.get("experiment_id")
        or "Legacy": r.get("experiment", {}).get("label", "Legacy, condition unknown")
        for r in records
    }
    exp = st.selectbox(
        "Experiment",
        ["All", *sorted(experiments)],
        format_func=lambda e: e if e == "All" else f"{experiments[e]} ({e[:8]})",
    )
    if exp != "All":
        records = [r for r in records if (r.get("experiment_id") or "Legacy") == exp]
    ids = {r["response_id"] for r in records}
    rows = [r for r in rows if r["response_id"] in ids]
    runs = {
        r["scoring_run_id"]: r.get("scoring_label", r["scoring_run_id"])
        for r in rows
        if r.get("scoring_run_id")
    }
    scoring = st.selectbox(
        "Scoring selection",
        ["Latest stored", *sorted(runs)],
        format_func=lambda r: r if r == "Latest stored" else f"{runs[r]} ({r[:8]})",
    )
    if scoring != "Latest stored":
        rows = [r for r in rows if r.get("scoring_run_id") == scoring]
    else:
        rows = current_reference_rows(
            rows, _reference_answers(_reference_manifest_mtime())
        )
    rows = apply_scope(latest_results(rows))
    st.caption(
        "One selected score per response and metric. Legacy scorer versions are unknown. An explicit scoring run shows its historical reference; the latest view excludes outdated reference verdicts."
    )
    _render_outcomes(records)
    if records:
        st.caption(
            f"Gathered {min(r['timestamp'] for r in records)[:10]} to {max(r['timestamp'] for r in records)[:10]}"
        )
    groups = question_groups(records)
    summaries = {
        key: _question_summary(key, group, rows) for key, group in groups.items()
    }
    attention = st.checkbox(
        "Only questions needing attention",
        help="Hides questions where every run answered and every metric passed. "
        + NEEDS_ATTENTION_HELP,
    )
    keys = [
        key for key, row in summaries.items() if not attention or row["Needs attention"]
    ]
    keys.sort(key=lambda k: (not summaries[k]["Needs attention"], k))
    st.dataframe(
        [summaries[k] for k in keys],
        hide_index=True,
        width="stretch",
        column_config=_column_help(QUESTION_COLUMNS),
    )
    if not keys:
        st.info("No questions match this selection.")
        return
    selected = st.selectbox(
        "Inspect question",
        keys,
        format_func=lambda k: f"Q{k[0]} · {k[1]} · {k[2]} · {k[3]} · {k[5][:8]}",
    )
    _question_detail(selected, groups[selected], rows)


if __name__ == "__main__":
    main()
